import json
import os
import sys
import pandas as pd
from pathlib import Path


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


def print_cuda_info() -> None:
	import torch

	visible = os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")
	print(f"Using GPU with ID:\t{visible}")
	print(f"CUDA Available:\t\t{torch.cuda.is_available()}")

	if torch.cuda.is_available():
		cur = torch.cuda.current_device()
		print(f"GPU name:\t\t{torch.cuda.get_device_name(cur)}")
	else:
		print("WARNING: No CUDA device available - inference will be very slow or fail.")
	print()


def get_gpu_name():
	"""returns the name of the current cuda device, or None if cuda is unavailable."""
	import torch
	if not torch.cuda.is_available():
		return None
	return torch.cuda.get_device_name(torch.cuda.current_device())


def get_gpu_memory_mb() -> dict:
	"""
	returns currently allocated and reserved gpu memory in megabytes.

	returns a dict with keys allocated_mb and reserved_mb, both None when cuda is unavailable.
	"""
	import torch
	if not torch.cuda.is_available():
		return {"allocated_mb": None, "reserved_mb": None}
	dev = torch.cuda.current_device()
	return {
		"allocated_mb": round(torch.cuda.memory_allocated(dev) / 1024 ** 2, 2),
		"reserved_mb": round(torch.cuda.memory_reserved(dev) / 1024 ** 2, 2),
	}


def reset_peak_gpu_memory() -> None:
	"""resets the peak gpu memory counter so the next measurement starts from zero."""
	import torch
	if torch.cuda.is_available():
		torch.cuda.reset_peak_memory_stats()


def get_peak_gpu_memory_mb():
	"""returns peak gpu memory allocated since the last reset in megabytes, or None if cuda is unavailable."""
	import torch
	if not torch.cuda.is_available():
		return None
	return round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)


def write_run_stats(output_file, stats: dict) -> None:
	"""
	writes a stats dictionary as formatted json next to output_file.

	the output filename is derived by stripping the extension from output_file
	and appending _run_stats.json. prints the path on completion.
	"""
	output_file = Path(output_file)
	stats_path = output_file.parent / f"{output_file.stem}_run_stats.json"
	with open(stats_path, "w") as fh:
		json.dump(stats, fh, indent=2)
	print(f"run stats written to {stats_path}")


def _load_input_file(path: str) -> pd.DataFrame:
    """
    loads a csv or parquet file based on file extension.

    accepts .csv, .tsv, or .parquet files and returns a dataframe.
    raises ValueError for unsupported formats.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"unsupported file format: {suffix!r} — use .csv, .tsv, or .parquet")
