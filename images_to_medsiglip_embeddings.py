import os
import sys
import argparse
from utils.utils import (
    _extract_gpu_arg_early,
    print_cuda_info,
    _load_input_file,
    get_gpu_name,
    get_gpu_memory_gb,
    reset_peak_gpu_memory,
    get_peak_gpu_memory_gb,
    write_run_stats,
)

_EARLY_GPU_ID = _extract_gpu_arg_early()
if not _EARLY_GPU_ID:
    print("Error: --gpu is required for this script", file=sys.stderr)
    sys.exit(1)
os.environ["CUDA_VISIBLE_DEVICES"] = _EARLY_GPU_ID
 
import time
from pathlib import Path
import shutil
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm.auto import tqdm 

TARGET_SIZE = 448

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create MedSigLIP embeddings for images listed in a CSV or Parquet file.")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV or Parquet file with image paths in FilePath column")
    parser.add_argument("--output_file", type=str, required=True, help="Output parquet file path")
    parser.add_argument("--gpu", type=str, required=True, help="Physical GPU ID (required)")
    parser.add_argument("--model_id", type=str, default="google/medsiglip-448")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of 2D images (frames) per forward pass")
    parser.add_argument("--save_every", type=int, default=1, help="Save every N model batches")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit to the first N input rows")
    parser.add_argument(
        "--sample_frames",
        type=str,
        choices=["middle", "all"],
        default="middle",
        help="'middle' embeds only the middle frame of each volume; 'all' embeds every frame as a separate row",
    )
    return parser 

def _load_volume(path: str) -> Tuple[np.ndarray, bool, int]:
    """Load one file as a stack with a leading frame axis.
 
    Returns (frames, invert, samples_per_pixel) where:
      * grayscale -> frames has shape (F, H, W)
      * color     -> frames has shape (F, H, W, 3)
      * invert is True for MONOCHROME1 (display-inverted) DICOMs.
    Frame count comes from the decoded pixel data, not from metadata.
    """
    suffix = Path(path).suffix.lower()
 
    if suffix in {".dcm", ""}:
        import pydicom

        dataset = pydicom.dcmread(path)
        array = dataset.pixel_array.astype(np.float32)
 
        samples = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
        photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
 
        # Rescale slope/intercept are harmless no-ops when absent (typical for MR).
        slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
        array = array * slope + intercept

        # Disambiguate frames vs channels using SamplesPerPixel, never the shape.
        if samples == 1:
            if array.ndim == 2:            # (H, W) single frame
                array = array[None, ...]   # -> (1, H, W)
            # else already (F, H, W)
        else:
            if array.ndim == 3:            # (H, W, C) single color frame
                array = array[None, ...]   # -> (1, H, W, C)
            array = array[..., :3]         # drop alpha if present -> (F, H, W, 3)

        invert = photometric == "MONOCHROME1"
        return array, invert, samples

    # Non-DICOM (png/jpg/...): already a normal 8-bit display image.
    image = Image.open(path).convert("RGB")
    array = np.asarray(image).astype(np.float32)[None, ...]  # (1, H, W, 3)
    return array, False, 3


def _prepare_frame(frame: np.ndarray, invert: bool, samples: int) -> Image.Image:
    """Turn one raw frame into a 448x448 RGB uint8 PIL image ready for the processor.
 
    Grayscale frames are min-max normalized per frame (MR has no absolute scale),
    then replicated to 3 identical channels. The (-1, 1) normalization is left to
    the processor, only produce the 0-255 image here.
    """
    f = np.nan_to_num(frame.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if samples == 1:
        if invert:
            f = f.max() - f
        f = f - f.min()
        fmax = f.max()
        if fmax > 0:
            f = f / fmax
        f8 = (f * 255.0).clip(0, 255).astype(np.uint8)
        f8 = np.repeat(f8[:, :, None], 3, axis=2)          # (H, W, 3)
    else:
        f8 = f.clip(0, 255).astype(np.uint8)
        if f8.ndim == 2:
            f8 = np.repeat(f8[:, :, None], 3, axis=2)
        f8 = f8[..., :3]

    tensor = torch.from_numpy(f8).permute(2, 0, 1).float()  # (3, H, W)
    # similar to how Google did it (torch equivalent)
    tensor = TF.resize(
        tensor,
        [TARGET_SIZE, TARGET_SIZE],
        interpolation=InterpolationMode.BILINEAR,
        antialias=False,
    )                                           # (H, W, 3)
    resized = tensor.round().clamp(0, 255).permute(1, 2, 0).contiguous().numpy().astype(np.uint8)
    return Image.fromarray(resized)


def _embed_images(model, processor, images: List[Image.Image], device: torch.device) -> np.ndarray:
    # images are already 448x448, so skip the processor's resize
    # it does the rescale (1/255) + mean/std normalization to (-1, 1).
    inputs = processor(images=images, do_resize=False, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        raw_output = model.get_image_features(**inputs)  # only image embeddings wanted

    embeddings = raw_output.pooler_output
    # L2 norm => dot product is directly cosine similarity
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return embeddings.detach().cpu().numpy()


def _write_chunks_to_parquet(chunk_paths: List[Path], output_file: Path) -> None:
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
    if "FilePath" not in dataframe.columns:
        raise KeyError("expected a FilePath column in the input file")
    if args.max_samples is not None:
        dataframe = dataframe.head(args.max_samples)
    dataframe = dataframe.reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_cuda_info()
    processor = AutoImageProcessor.from_pretrained(args.model_id, use_fast=False)
    model = AutoModel.from_pretrained(args.model_id) 
    model.to(device)
    model.eval()
    mem_after_model = get_gpu_memory_gb()
    reset_peak_gpu_memory()

    chunk_dir = output_path.parent / f"{output_path.stem}_chunks"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    errors: List[dict] = []
    chunk_paths: List[Path] = []
    batch_buffer: List[pd.DataFrame] = []   # per-model-batch frames awaiting a chunk write

    # Image-level buffers, filled across volumes until they reach batch_size.
    buf_idx: List[int] = []
    buf_frame: List[int] = []
    buf_total: List[int] = []
    buf_imgs: List[Image.Image] = []

    state = {"batch_index": 0, "total_embeddings": 0}
 
    def write_chunk() -> None:
        if not batch_buffer:
            return
        buffered = pd.concat(batch_buffer, ignore_index=True)
        chunk_path = chunk_dir / f"batch_{len(chunk_paths) + 1:06d}.parquet"
        buffered.to_parquet(chunk_path, index=False)
        chunk_paths.append(chunk_path)
        batch_buffer.clear()
 
    def flush_batch() -> None:
        if not buf_imgs:
            return
        state["batch_index"] += 1
        embeddings = _embed_images(model, processor, buf_imgs, device)
        rows = dataframe.iloc[buf_idx].reset_index(drop=True).copy()
        rows["frame_index"] = buf_frame
        rows["n_frames"] = buf_total
        rows["medsiglip_embedding"] = embeddings.tolist()
        batch_buffer.append(rows)
        state["total_embeddings"] += len(rows)
 
        buf_idx.clear()
        buf_frame.clear()
        buf_total.clear()
        buf_imgs.clear()

        if len(batch_buffer) >= args.save_every:
            write_chunk()

    progress = tqdm(total=len(dataframe), desc=f"Embedding ({args.sample_frames})", unit="file")
    try:
        for row_index in range(len(dataframe)):
            path = str(dataframe["FilePath"].iloc[row_index])
            try:
                frames, invert, samples = _load_volume(path)
                n_frames = int(frames.shape[0])

                # a decoded volume with no frames is treated as a failure rather
                # than silently skipped, so it is recorded in the error list
                if n_frames == 0:
                    raise ValueError("decoded volume contains no frames")

                if args.sample_frames == "middle":
                    frame_indices = [(n_frames - 1) // 2]  # first of the two middle frames when even
                else:
                    frame_indices = list(range(n_frames))

                # all selected frames are prepared into a local list before the
                # shared buffers are touched, so a failure partway through a file
                # leaves no partial rows behind for it
                prepared = [_prepare_frame(frames[k], invert, samples) for k in frame_indices]

                # the whole file decoded and prepared cleanly, so its frames are committed
                for k, image in zip(frame_indices, prepared):
                    buf_idx.append(row_index)
                    buf_frame.append(k + 1)    # 1-based
                    buf_total.append(n_frames)
                    buf_imgs.append(image)
                    if len(buf_imgs) >= args.batch_size:
                        flush_batch()
            except Exception as exc:  
                errors.append({"row_index": int(row_index), "FilePath": path, "error": repr(exc)})
            finally:
                progress.update(1)
                progress.set_postfix(embedded=state["total_embeddings"], errors=len(errors))
 
        flush_batch()  # final partial model batch
    finally:
        progress.close()
 
    write_chunk()  # write any batches still buffered below save_every
    _write_chunks_to_parquet(chunk_paths, output_path)
    shutil.rmtree(chunk_dir, ignore_errors=True)
 
    write_run_stats(output_path, {
        "gpu_name": get_gpu_name(),
        "model_id": args.model_id,
        "sample_frames": args.sample_frames,
        "batch_size": args.batch_size,
        "num_input_files": len(dataframe),
        "num_embeddings": state["total_embeddings"],
        "num_failed_files": len(errors),
        "failed_files": errors,
        "memory_after_model_load_gb": mem_after_model,
        "peak_memory_embedding_gb": get_peak_gpu_memory_gb(),
        "runtime_seconds": round(time.time() - t_start, 2),
    })
 
 
if __name__ == "__main__":
    main()
 