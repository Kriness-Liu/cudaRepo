"""Reproducible CUDA microbenchmark helpers.

CUDA launches are asynchronous.  A CPU wall-clock measurement around a launch
mostly measures enqueue overhead, so this module records CUDA Events on the
current stream and synchronizes only when reading each sample.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Callable

import torch


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""

    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class BenchmarkResult:
    """Distribution summary for device-side execution time."""

    iterations: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def benchmark_cuda(
    operation: Callable[[], object],
    *,
    warmup: int = 10,
    iterations: int = 50,
) -> BenchmarkResult:
    """Measure ``operation`` on the current CUDA stream with CUDA Events.

    Compilation and allocator warm-up should happen before samples are kept.
    The helper intentionally reports a distribution rather than a single best
    number.  It measures device work bracketed by the events, not Python setup
    performed before the first event.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iterations):
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))

    return BenchmarkResult(
        iterations=iterations,
        p50_ms=_percentile(samples, 50.0),
        p95_ms=_percentile(samples, 95.0),
        mean_ms=fmean(samples),
        min_ms=min(samples),
        max_ms=max(samples),
    )


def effective_bandwidth_gbps(bytes_moved: int, elapsed_ms: float) -> float:
    """Convert a known logical byte count and latency to decimal GB/s."""

    if bytes_moved < 0:
        raise ValueError("bytes_moved must be non-negative")
    if elapsed_ms <= 0:
        raise ValueError("elapsed_ms must be positive")
    return bytes_moved / (elapsed_ms / 1000.0) / 1e9

