"""Capture the environment needed to reproduce a CUDA benchmark."""

from __future__ import annotations

import platform
import sys
from typing import Any

import torch


def collect_environment() -> dict[str, Any]:
    """Return software and GPU properties without running a benchmark."""

    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        return report

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    report["device"] = {
        "index": device_index,
        "name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device_index)),
        "multiprocessor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
    }
    report["nvrtc_route"] = (
        "PyTorch-bundled NVRTC -> PTX -> CUDA Driver API -> current PyTorch stream"
    )
    return report

