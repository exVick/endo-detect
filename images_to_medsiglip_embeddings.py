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
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm.auto import tqdm 

TARGET_SIZE = 448
CLIP_PERCENTILES = (1.0, 99.0)
REQUIRED_COLUMNS = ["FilePath", "SOPInstanceUID", "frame_index", "SeriesInstanceUID", "AccessionNumber"]
KEY_COLUMNS = ["input_row", "SOPInstanceUID", "frame_index", "n_frames"]
MAX_LISTED_FRAME_ERRORS = 500

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create MedSigLIP embeddings for images listed in a CSV or Parquet file.")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV or Parquet frame manifest, one row per 2D frame") 
    parser.add_argument("--output_file", type=str, required=True, help="Output .npy file path; keys are written next to it as <stem>_keys.parquet")
    parser.add_argument("--gpu", type=str, required=True, help="Physical GPU ID (required)")
    parser.add_argument("--model_id", type=str, default="google/medsiglip-448")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of 2D images (frames) per forward pass")
    parser.add_argument("--save_every", type=int, default=1, help="Save every N model batches")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit to the first N input rows")
    return parser


def _is_dicom(path: str) -> bool:
    """Detect DICOM from file content rather than from the filename.

    Some exports name files after a UID component, so Path.suffix reports
    something like '.30000024050212003631800000258' and a suffix-based test
    would misroute a valid DICOM to the image loader. The DICM marker at
    offset 128 is part of the standard file preamble.
    """
    try:
        with open(path, "rb") as handle:
            return handle.read(132)[128:132] == b"DICM"
    except OSError:
        return False


def _load_volume(path: str) -> Tuple[np.ndarray, bool, int]:
    """Load one file as a stack with a leading frame axis.
 
    Returns (frames, invert, samples_per_pixel) where:
      * grayscale -> frames has shape (F, H, W)
      * color     -> frames has shape (F, H, W, 3)
      * invert is True for MONOCHROME1 (display-inverted) DICOMs.
    Frame count comes from the decoded pixel data, not from metadata.
    """
    if _is_dicom(path):
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
 
    Grayscale frames are windowed to a robust percentile range per frame (MR has no
    absolute scale), then replicated to 3 identical channels. The (-1, 1) normalization
    is left to the processor, only produce the 0-255 image here.
    """
    f = np.nan_to_num(frame.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if samples == 1:
        if invert:
            f = f.max() - f
        # percentile window instead of raw min-max: a single hot voxel or a noisy
        # edge slice would otherwise set the scale for the whole frame, so the same
        # tissue lands at different intensities across slices of one volume
        low, high = np.percentile(f, CLIP_PERCENTILES)
        if high <= low:                    # near-flat frame, fall back to full range
            low, high = float(f.min()), float(f.max())
        f = np.clip(f, low, high) - low
        span = float(high - low)
        if span > 0:
            f = f / span
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


def _summarize_errors(errors: List[dict]) -> dict:
    """Collapse per-frame failures into one entry per series, plus a capped flat list.

    One unreadable multi-frame file fails every frame it holds, so the flat list
    alone would be unreadable; the grouped view keeps that as a single line and
    names the accession and series to look at.
    """
    if not errors:
        return {"failed_series": [], "failed_frames": [], "failed_frames_truncated": 0}

    frame = pd.DataFrame(errors)
    grouped: List[dict] = []
    for (accession, series), rows in frame.groupby(["AccessionNumber", "SeriesInstanceUID"], sort=False):
        grouped.append({
            "AccessionNumber": str(accession),
            "SeriesInstanceUID": str(series),
            "num_failed_frames": int(len(rows)),
            "num_files_affected": int(rows["FilePath"].nunique()),
            "distinct_errors": sorted(set(rows["error"].tolist()))[:5],
            "example_file": str(rows["FilePath"].iloc[0]),
        })
    grouped.sort(key=lambda item: item["num_failed_frames"], reverse=True)

    return {
        "failed_series": grouped,
        "failed_frames": errors[:MAX_LISTED_FRAME_ERRORS],
        "failed_frames_truncated": max(0, len(errors) - MAX_LISTED_FRAME_ERRORS),
    }


def _write_chunks_to_outputs(
    chunk_paths: List[Tuple[Path, Path]], embedding_file: Path, keys_file: Path
) -> None:
    """Concatenate the per-batch chunks into one embedding matrix and one key table.

    Frames are embedded grouped by file, so both are finally sorted back into input
    row order: row i of the matrix is then row i of the input manifest, and row i of
    the key table always identifies it regardless of what failed.
    """
    if not chunk_paths:
        np.save(embedding_file, np.zeros((0, 0), dtype=np.float32))
        pd.DataFrame(columns=KEY_COLUMNS).to_parquet(keys_file, index=False)
        return

    embeddings = np.concatenate([np.load(path) for path, _ in chunk_paths], axis=0)
    keys = pd.concat([pd.read_parquet(path) for _, path in chunk_paths], ignore_index=True)

    order = np.argsort(keys["input_row"].to_numpy(), kind="stable")
    embeddings = embeddings[order]
    keys = keys.iloc[order].reset_index(drop=True)

    np.save(embedding_file, embeddings.astype(np.float32))
    keys.to_parquet(keys_file, index=False)


def main() -> None:
    t_start = time.time()

    parser = _build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file).with_suffix(".npy")
    keys_path = output_path.with_name(f"{output_path.stem}_keys.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = _load_input_file(str(input_path))
    for column in REQUIRED_COLUMNS:
        if column not in dataframe.columns:
            raise KeyError(f"expected a {column} column in the input file")
    if args.max_samples is not None:
        dataframe = dataframe.head(args.max_samples)
    # only the identity of each frame is needed; every other manifest column stays
    # in the manifest and is joined back on input_row later
    dataframe = dataframe[REQUIRED_COLUMNS].reset_index(drop=True)
    dataframe["input_row"] = np.arange(len(dataframe), dtype=np.int64)
    sop_uids = dataframe["SOPInstanceUID"].to_numpy()
    frame_numbers = dataframe["frame_index"].to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # the siglip vision tower contains a single patch-embedding conv, for which no cudnn engine is available 
    # on some gpu architectures (eg volta / gv100) under the current cudnn build, where the forward pass 
    # fails with "unable to find an engine to execute this computation"
    # cudnn is disabled so that this one conv is routed to a native kernel
    # overall performance is unaffected because the rest of the model is matmul/attention, not convolution
    torch.backends.cudnn.enabled = False

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
    chunk_paths: List[Tuple[Path, Path]] = []
    batch_buffer: List[Tuple[np.ndarray, pd.DataFrame]] = []   # per-model-batch results awaiting a chunk write

    # Image-level buffers, filled across volumes until they reach batch_size.
    buf_row: List[int] = []
    buf_total: List[int] = []
    buf_imgs: List[Image.Image] = []

    state = {"batch_index": 0, "total_embeddings": 0}
 
    def write_chunk() -> None:
        if not batch_buffer:
            return
        embeddings = np.concatenate([item[0] for item in batch_buffer], axis=0)
        keys = pd.concat([item[1] for item in batch_buffer], ignore_index=True)
        stem = f"batch_{len(chunk_paths) + 1:06d}"
        embedding_chunk = chunk_dir / f"{stem}.npy"
        keys_chunk = chunk_dir / f"{stem}.parquet"
        np.save(embedding_chunk, embeddings.astype(np.float32))
        keys.to_parquet(keys_chunk, index=False)
        chunk_paths.append((embedding_chunk, keys_chunk))
        batch_buffer.clear()
 
    def flush_batch() -> None:
        if not buf_imgs:
            return
        state["batch_index"] += 1
        embeddings = _embed_images(model, processor, buf_imgs, device)
        keys = pd.DataFrame({
            "input_row": buf_row,
            "SOPInstanceUID": sop_uids[buf_row],
            "frame_index": frame_numbers[buf_row],
            "n_frames": buf_total,
        })
        batch_buffer.append((embeddings, keys))
        state["total_embeddings"] += len(keys)
 
        buf_row.clear()
        buf_total.clear()
        buf_imgs.clear()

        if len(batch_buffer) >= args.save_every:
            write_chunk()

    def record_error(record, message: str) -> None:
        errors.append({
            "input_row": int(record.input_row),
            "AccessionNumber": str(record.AccessionNumber),
            "SeriesInstanceUID": str(record.SeriesInstanceUID),
            "SOPInstanceUID": str(record.SOPInstanceUID),
            "frame_index": int(record.frame_index),
            "FilePath": str(record.FilePath),
            "error": message,
        })

    progress = tqdm(total=len(dataframe), desc="Embedding", unit="frame")
    try:
        # grouping by file means a multi-frame instance is read and decoded once,
        # no matter how many of its frames the manifest asks for
        for path, group in dataframe.groupby("FilePath", sort=False):
            try:
                frames, invert, samples = _load_volume(str(path))
                n_frames = int(frames.shape[0])

                # a decoded volume with no frames is treated as a failure rather
                # than silently skipped, so it is recorded in the error list
                if n_frames == 0:
                    raise ValueError("decoded volume contains no frames")
            except Exception as exc:
                # the file could not be decoded, so every frame requested from it fails
                for record in group.itertuples():
                    record_error(record, repr(exc))
                progress.update(len(group))
                progress.set_postfix(embedded=state["total_embeddings"], errors=len(errors))
                continue

            for record in group.itertuples():
                try:
                    frame_number = int(record.frame_index)
                    # 1-based and bounded, so a stray 0 cannot silently select the
                    # last frame through negative indexing
                    if frame_number < 1 or frame_number > n_frames:
                        raise IndexError(f"frame_index {frame_number} outside 1..{n_frames}")
                    image = _prepare_frame(frames[frame_number - 1], invert, samples)
                except Exception as exc:  
                    record_error(record, repr(exc))
                    continue

                buf_row.append(int(record.input_row))
                buf_total.append(n_frames)
                buf_imgs.append(image)
                if len(buf_imgs) >= args.batch_size:
                    flush_batch()

            progress.update(len(group))
            progress.set_postfix(embedded=state["total_embeddings"], errors=len(errors))
 
        flush_batch()  # final partial model batch
    finally:
        progress.close()
 
    write_chunk()  # write any batches still buffered below save_every
    _write_chunks_to_outputs(chunk_paths, output_path, keys_path)
    shutil.rmtree(chunk_dir, ignore_errors=True)
 
    write_run_stats(output_path, {
        "gpu_name": get_gpu_name(),
        "model_id": args.model_id,
        "batch_size": args.batch_size,
        "embedding_file": str(output_path),
        "keys_file": str(keys_path),
        "num_input_frames": len(dataframe),
        "num_input_files": int(dataframe["FilePath"].nunique()),
        "num_embeddings": state["total_embeddings"],
        "num_failed_frames": len(errors),
        **_summarize_errors(errors),
        "memory_after_model_load_gb": mem_after_model,
        "peak_memory_embedding_gb": get_peak_gpu_memory_gb(),
        "runtime_seconds": round(time.time() - t_start, 2),
    })
 
 
if __name__ == "__main__":
    main()