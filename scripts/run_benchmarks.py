"""Run a compact, reproducible benchmark suite and save JSON results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from cudarepo.benchmark import benchmark_cuda, effective_bandwidth_gbps
from cudarepo.environment import collect_environment
from cudarepo.kernels import get_kernels


def benchmark_transpose(kernels, size: int) -> dict[str, Any]:
    matrix = torch.randn(size, size, device="cuda")
    torch.testing.assert_close(kernels.transpose_tiled(matrix), matrix.t())
    bytes_moved = matrix.numel() * matrix.element_size() * 2
    report: dict[str, Any] = {"shape": [size, size], "bytes_moved": bytes_moved}
    for name, operation in {
        "naive": lambda: kernels.transpose_naive(matrix),
        "tiled": lambda: kernels.transpose_tiled(matrix),
    }.items():
        result = benchmark_cuda(operation, warmup=10, iterations=50)
        values = result.to_dict()
        values["effective_bandwidth_gbps_p50"] = effective_bandwidth_gbps(
            bytes_moved, result.p50_ms
        )
        report[name] = values
    return report


def benchmark_fusion(kernels, rows: int, columns: int) -> dict[str, Any]:
    inputs = torch.randn(rows, columns, device="cuda")
    bias = torch.randn(columns, device="cuda")
    torch.testing.assert_close(kernels.fused_bias_relu(inputs, bias), torch.relu(inputs + bias))
    eager = benchmark_cuda(lambda: torch.relu(inputs + bias), warmup=10, iterations=50)
    fused = benchmark_cuda(
        lambda: kernels.fused_bias_relu(inputs, bias),
        warmup=10,
        iterations=50,
    )
    return {
        "shape": [rows, columns],
        "eager": eager.to_dict(),
        "fused": fused.to_dict(),
        "p50_speedup": eager.p50_ms / fused.p50_ms,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA不可用，benchmark未运行。")

    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    kernels = get_kernels()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": collect_environment(),
        "methodology": {
            "warmup": 10,
            "iterations": 50,
            "timer": "CUDA Event on current PyTorch stream",
            "correctness": "torch.testing.assert_close against PyTorch reference",
            "performance_threshold_in_tests": False,
        },
        "transpose_2048": benchmark_transpose(kernels, 2048),
        "fused_bias_relu": benchmark_fusion(kernels, 4096, 1024),
        "scope": [
            "single NVIDIA GPU",
            "kernel-only timing",
            "Windows WDDM scheduling may affect tails",
            "no NCCL, RDMA, RoCE or multi-node measurement",
        ],
    }

    output = Path(__file__).resolve().parents[1] / "results" / "latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()

