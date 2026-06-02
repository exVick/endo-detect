import os
import sys
import argparse
from utils.utils import (
    _extract_gpu_arg_early,
    print_cuda_info,
    _load_input_file,
    get_gpu_name,
    get_gpu_memory_mb,
    reset_peak_gpu_memory,
    get_peak_gpu_memory_mb,
    write_run_stats,
)


# gpu must be specified before cuda initializes — exit early if missing
_EARLY_GPU_ID = _extract_gpu_arg_early()
if not _EARLY_GPU_ID:
    print("Error: --gpu is required for this script", file=sys.stderr)
    sys.exit(1)
os.environ["CUDA_VISIBLE_DEVICES"] = _EARLY_GPU_ID

import time
from pathlib import Path
import shutil
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create MedGemma text embeddings for texts listed in a CSV or Parquet file.")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV or Parquet file with text in pdf_text column")
    parser.add_argument("--output_file", type=str, required=True, help="Output parquet file path")
    parser.add_argument("--gpu", type=str, required=True, help="Physical GPU ID")
    parser.add_argument("--model_id", type=str, default="google/medgemma-4b-it")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=1, help="Save every N batches")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit to the first N samples from the input file")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum token sequence length; longer texts are truncated")
    return parser


def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    computes mean pooling over the token dimension, ignoring padding positions.

    multiplies hidden states by the expanded attention mask so padding tokens
    contribute zero, then divides by the count of real tokens.
    returns a tensor of shape (batch_size, hidden_dim).
    """
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _embed_batch(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int,
) -> np.ndarray:
    """
    tokenizes a list of strings and returns l2-normalized embeddings.

    pads and truncates to max_length, runs a forward pass with no gradient,
    applies mean pooling over non-padding tokens, and l2-normalizes each vector.
    returns a float32 numpy array of shape (batch_size, embedding_dim).
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True)

    # prefer last_hidden_state; fall back to the final entry in hidden_states
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        hidden = output.last_hidden_state
    elif hasattr(output, "hidden_states") and output.hidden_states is not None:
        hidden = output.hidden_states[-1]
    else:
        raise ValueError("could not extract hidden states from model output")

    embeddings = _mean_pool(hidden, inputs["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return embeddings.detach().cpu().numpy()


def _write_chunks_to_parquet(chunk_paths: List[Path], output_file: Path) -> None:
    """merges all per-batch chunk files into a single output parquet file."""
    if not chunk_paths:
        empty_table = pa.Table.from_pandas(pd.DataFrame())
        pq.write_table(empty_table, output_file)
        return

    writer = None
    try:
        for chunk_path in chunk_paths:
            table = pq.read_table(chunk_path)
            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    t_start = time.time()

    parser = _build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = _load_input_file(str(input_path))
    if "pdf_text" not in dataframe.columns:
        raise KeyError("expected a pdf_text column in the input file")
    if args.max_samples is not None:
        dataframe = dataframe.head(args.max_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_cuda_info()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)
    model.to(device)
    model.eval()
    mem_after_model = get_gpu_memory_mb()
    reset_peak_gpu_memory()

    chunk_dir = output_path.parent / f"{output_path.stem}_chunks"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths: List[Path] = []
    batch_buffer = []
    batch_index = 0

    progress_bar = tqdm(total=len(dataframe), desc="Embedding texts", unit="doc")
    try:
        for start_idx in range(0, len(dataframe), args.batch_size):
            batch_index += 1
            batch = dataframe.iloc[start_idx : start_idx + args.batch_size].copy()
            texts = batch["pdf_text"].fillna("").astype(str).tolist()
            batch["medgemma_text_embedding"] = _embed_batch(
                model, tokenizer, texts, device, args.max_length
            ).tolist()
            batch_buffer.append(batch)

            progress_bar.update(len(batch))

            should_save = len(batch_buffer) >= args.save_every or start_idx + args.batch_size >= len(dataframe)
            if should_save:
                buffered = pd.concat(batch_buffer, ignore_index=True)
                chunk_path = chunk_dir / f"batch_{batch_index:06d}.parquet"
                buffered.to_parquet(chunk_path, index=False)
                chunk_paths.append(chunk_path)
                batch_buffer = []
    finally:
        progress_bar.close()

    _write_chunks_to_parquet(chunk_paths, output_path)
    shutil.rmtree(chunk_dir, ignore_errors=True)

    write_run_stats(output_path, {
        "gpu_name": get_gpu_name(),
        "model_id": args.model_id,
        "num_samples": len(dataframe),
        "memory_after_model_load_mb": mem_after_model,
        "peak_memory_embedding_mb": get_peak_gpu_memory_mb(),
        "runtime_seconds": round(time.time() - t_start, 2),
    })


if __name__ == "__main__":
    main()
