"""Compare Eager, NVRTC and compiled extension paths across shapes/dtypes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from cudarepo.benchmark import benchmark_cuda
from cudarepo.environment import collect_environment
from cudarepo.extension import extension_available, fused_bias_relu_extension
from cudarepo.kernels import get_kernels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/extension_latest.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not extension_available():
        raise SystemExit("Build the extension before running this benchmark")

    torch.manual_seed(7)
    kernels = get_kernels()
    cases = []
    for rows, columns in ((1024, 256), (4096, 1024), (8192, 1024)):
        for dtype in (torch.float32, torch.float16):
            inputs = torch.randn(rows, columns, device="cuda", dtype=dtype)
            bias = torch.randn(columns, device="cuda", dtype=dtype)
            reference = torch.relu(inputs + bias)
            extension_output = fused_bias_relu_extension(inputs, bias)
            torch.testing.assert_close(extension_output, reference)
            operations = {
                "eager": lambda: torch.relu(inputs + bias),
                "extension": lambda: fused_bias_relu_extension(inputs, bias),
            }
            if dtype == torch.float32:
                torch.testing.assert_close(kernels.fused_bias_relu(inputs, bias), reference)
                operations["nvrtc"] = lambda: kernels.fused_bias_relu(inputs, bias)
            timings = {
                name: benchmark_cuda(operation, warmup=args.warmup, iterations=args.iterations).to_dict()
                for name, operation in operations.items()
            }
            cases.append(
                {
                    "shape": [rows, columns],
                    "dtype": str(dtype),
                    "timings": timings,
                    "extension_speedup_vs_eager_p50": (
                        timings["eager"]["p50_ms"] / timings["extension"]["p50_ms"]
                    ),
                }
            )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": collect_environment(),
        "methodology": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "timer": "CUDA Event",
            "scope": "kernel-only; compilation excluded",
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
