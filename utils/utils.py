import os
import sys


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
