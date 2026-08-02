"""NVTX-annotated workload for Nsight Systems/Compute."""

from __future__ import annotations

import argparse

import torch

from cudarepo.extension import extension_available, fused_bias_relu_extension
from cudarepo.kernels import get_kernels


def run_range(name: str, operation, iterations: int) -> None:
    torch.cuda.nvtx.range_push(name)
    try:
        for _ in range(iterations):
            operation()
    finally:
        torch.cuda.nvtx.range_pop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    inputs = torch.randn(4096, 1024, device="cuda")
    bias = torch.randn(1024, device="cuda")
    kernels = get_kernels()
    run_range("eager_bias_relu", lambda: torch.relu(inputs + bias), args.iterations)
    run_range("nvrtc_fused_bias_relu", lambda: kernels.fused_bias_relu(inputs, bias), args.iterations)
    if extension_available():
        run_range(
            "extension_fused_bias_relu",
            lambda: fused_bias_relu_extension(inputs, bias),
            args.iterations,
        )
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
