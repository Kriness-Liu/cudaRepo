"""实验02：矩阵转置中的合并访存与shared-memory tile。

Naive Kernel按行读取输入，但向转置后的输出做跨步写入。Tiled Kernel先把
一个32x32 tile搬到shared memory，再交换block坐标连续写回；shared-memory
数组声明为32x33，额外一列padding用于避免按列读取时的bank conflict。

性能受GPU、shape和WDDM调度影响，因此这里只报告P50/P95和有效带宽，不把
某个固定加速倍数写进断言。
"""

from __future__ import annotations

import torch

from cudarepo.benchmark import benchmark_cuda, effective_bandwidth_gbps
from cudarepo.kernels import get_kernels


def print_result(name: str, result, bytes_moved: int) -> None:
    bandwidth = effective_bandwidth_gbps(bytes_moved, result.p50_ms)
    print(
        f"{name:16s} p50={result.p50_ms:.4f} ms, "
        f"p95={result.p95_ms:.4f} ms, effective={bandwidth:.2f} GB/s"
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA不可用。")
        return

    kernels = get_kernels()
    # 非tile整数倍尺寸验证边界；大尺寸用于benchmark。
    boundary_case = torch.randn(1003, 769, device="cuda")
    torch.testing.assert_close(kernels.transpose_naive(boundary_case), boundary_case.t())
    torch.testing.assert_close(kernels.transpose_tiled(boundary_case), boundary_case.t())

    matrix = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
    reference = matrix.t().contiguous()
    torch.testing.assert_close(kernels.transpose_naive(matrix), reference)
    torch.testing.assert_close(kernels.transpose_tiled(matrix), reference)

    naive = benchmark_cuda(lambda: kernels.transpose_naive(matrix), warmup=10, iterations=50)
    tiled = benchmark_cuda(lambda: kernels.transpose_tiled(matrix), warmup=10, iterations=50)
    bytes_moved = matrix.numel() * matrix.element_size() * 2  # 一次读+一次写
    print_result("naive", naive, bytes_moved)
    print_result("shared tiled", tiled, bytes_moved)
    print("PASS: 两种Kernel均与PyTorch reference一致。")


if __name__ == "__main__":
    main()

