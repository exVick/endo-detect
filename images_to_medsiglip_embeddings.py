import os
import sys
import argparse


def _extract_gpu_arg_early(default: str = "0") -> str:
	"""
	Extract GPU early, to limit cuda to only one physical ID (for shared servers with multiple GPUs)
	"""
	gpu_id = default
	argv = sys.argv[1:]

	for i, tok in enumerate(argv):
		if tok == "--gpu" and i + 1 < len(argv):
			gpu_id = argv[i + 1]
			break
		if tok.startswith("--gpu="):
			gpu_id = tok.split("=", 1)[1]
			break

	return gpu_id


_EARLY_GPU_ID = _extract_gpu_arg_early()
os.environ["CUDA_VISIBLE_DEVICES"] = _EARLY_GPU_ID

from pathlib import Path
import shutil
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Create MedSigLIP embeddings for images listed in a CSV file.")
	parser.add_argument("--csv_file", type=str, required=True, help="Input CSV file with image paths in FilePath column")
	parser.add_argument("--output_file", type=str, required=True, help="Output parquet file path")
	parser.add_argument("--gpu", type=str, default=_EARLY_GPU_ID, help="Physical GPU ID")
	parser.add_argument("--model_id", type=str, default="google/medsiglip-448")
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--save_every", type=int, default=1, help="Save every N batches")
	parser.add_argument("--max_samples", type=int, default=None, help="Optional limit to the first N samples from the CSV")
	return parser


def _load_image(path: str) -> Image.Image:
	suffix = Path(path).suffix.lower()

	if suffix in {".dcm", ""}:
		try:
			import pydicom
		except ImportError as exc:
			raise ImportError("pydicom is required to read DICOM images") from exc

		dataset = pydicom.dcmread(path)
		array = dataset.pixel_array.astype(np.float32)

		if array.ndim > 2:
			if array.shape[-1] in {3, 4}:
				array = array[..., :3]
			else:
				array = array[0]

		slope = float(getattr(dataset, "RescaleSlope", 1.0))
		intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
		array = array * slope + intercept

		if getattr(dataset, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
			array = array.max() - array

		array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
		array = array - array.min()
		max_value = array.max()
		if max_value > 0:
			array = array / max_value

		array = (array * 255.0).clip(0, 255).astype(np.uint8)
		if array.ndim == 2:
			array = np.repeat(array[:, :, None], 3, axis=2)
		return Image.fromarray(array).convert("RGB")

	return Image.open(path).convert("RGB")


def _load_batch_images(paths: List[str]) -> List[Image.Image]:
	return [_load_image(path) for path in paths]


def _extract_embedding_tensor(output):
	if torch.is_tensor(output):
		return output

	if hasattr(output, "image_embeds") and output.image_embeds is not None:
		return output.image_embeds

	if hasattr(output, "pooler_output") and output.pooler_output is not None:
		return output.pooler_output

	if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
		return output.last_hidden_state[:, 0]

	if isinstance(output, (tuple, list)) and len(output) > 0:
		for item in output:
			if torch.is_tensor(item):
				return item

	raise ValueError("Could not extract a tensor embedding from model output")


def _embed_batch(model, processor, image_paths: List[str], device: torch.device) -> np.ndarray:
	images = _load_batch_images(image_paths)
	inputs = processor(images=images, return_tensors="pt")
	inputs = {key: value.to(device) for key, value in inputs.items()}

	with torch.no_grad():
		if hasattr(model, "get_image_features"):
			raw_output = model.get_image_features(**inputs)
		else:
			raw_output = model(**inputs)

	embeddings = _extract_embedding_tensor(raw_output)

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
	parser = _build_parser()
	args = parser.parse_args()

	csv_path = Path(args.csv_file)
	output_path = Path(args.output_file)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	dataframe = pd.read_csv(csv_path, low_memory=False)
	if "FilePath" not in dataframe.columns:
		raise KeyError("Expected a FilePath column in the input CSV")
	if args.max_samples is not None:
		dataframe = dataframe.head(args.max_samples)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	processor = AutoImageProcessor.from_pretrained(args.model_id, trust_remote_code=True, use_fast=False)
	model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)
	model.to(device)
	model.eval()

	chunk_dir = output_path.parent / f"{output_path.stem}_chunks"
	if chunk_dir.exists():
		shutil.rmtree(chunk_dir)
	chunk_dir.mkdir(parents=True, exist_ok=True)

	chunk_paths: List[Path] = []
	batch_buffer = []
	batch_index = 0

	for start_idx in range(0, len(dataframe), args.batch_size):
		batch_index += 1
		batch = dataframe.iloc[start_idx : start_idx + args.batch_size].copy()
		image_paths = batch["FilePath"].astype(str).tolist()
		batch["medsiglip_embedding"] = _embed_batch(model, processor, image_paths, device).tolist()
		batch_buffer.append(batch)

		should_save = len(batch_buffer) >= args.save_every or start_idx + args.batch_size >= len(dataframe)
		if should_save:
			buffered = pd.concat(batch_buffer, ignore_index=True)
			chunk_path = chunk_dir / f"batch_{batch_index:06d}.parquet"
			buffered.to_parquet(chunk_path, index=False)
			chunk_paths.append(chunk_path)
			batch_buffer = []

	_write_chunks_to_parquet(chunk_paths, output_path)
	shutil.rmtree(chunk_dir, ignore_errors=True)


if __name__ == "__main__":
	main()