"""Shared torch helpers for reproducible fitting.

Both the attention pooling in :mod:`attention` and the multi-layer perceptron in
:mod:`classifiers` train small modules that have to give the same answer on
every run. The determinism machinery they need is identical, so it is kept here
rather than duplicated. The module imports nothing from the package, which is
what allows :mod:`classifiers` to stay a leaf and be imported by
:mod:`evaluate` without a cycle.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import torch

# the two settings cuBLAS accepts for reproducible matmuls. it has to be read
# from the environment before torch is imported, so it cannot be set here.
_CUBLAS_OK = (":4096:8", ":16:8")


@contextmanager
def _deterministic(flag):
    """Temporarily switch deterministic kernels on or off.

    The setting is global to torch, so the previous value is restored on exit
    and nothing outside the block is affected.

    Args:
        flag (bool): Whether deterministic algorithms should be enforced.
    """
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(flag)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)


def _check_deterministic(deterministic, dev):
    """Verify that deterministic execution is actually available.

    Args:
        deterministic (bool): Whether bit-reproducible results were requested.
        dev (torch.device): Device the work will run on.

    Raises:
        RuntimeError: If deterministic execution is requested on CUDA without
            CUBLAS_WORKSPACE_CONFIG having been set before torch was imported.
    """
    if (deterministic and dev.type == "cuda"
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in _CUBLAS_OK):
        raise RuntimeError(
            "deterministic=True on CUDA needs CUBLAS_WORKSPACE_CONFIG set "
            "before torch is imported:\n"
            "    import os\n"
            "    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'\n"
            "    import torch\n"
            "Pass deterministic=False to accept run-to-run drift of ~1e-5 "
            "instead.")
