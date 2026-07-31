"""实验04：用Bias+ReLU融合减少中间Tensor和Kernel launch。

Eager表达式 ``relu(inputs + bias)`` 通常包含加法、ReLU两个GPU Kernel，并
把中间结果写回全局内存。自定义Kernel在寄存器中完成加法和激活，只写一次
最终结果。

本Kernel仅支持连续FP32二维Tensor，不接入Autograd；接口边界由Python包装
显式检查。生产算子还需要dtype dispatch、Backward和完整的错误处理。
"""

from __future__ import annotations

import torch

from cudarepo.benchmark import benchmark_cuda
from cudarepo.kernels import get_kernels


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA不可用。")
        return

    rows, columns = 4096, 1024
    inputs = torch.randn(rows, columns, device="cuda", dtype=torch.float32)
    bias = torch.randn(columns, device="cuda", dtype=torch.float32)
    kernels = get_kernels()

    reference = torch.relu(inputs + bias)
    fused_output = kernels.fused_bias_relu(inputs, bias)
    torch.testing.assert_close(fused_output, reference)

    eager = benchmark_cuda(
        lambda: torch.relu(inputs + bias),
        warmup=10,
        iterations=50,
    )
    fused = benchmark_cuda(
        lambda: kernels.fused_bias_relu(inputs, bias),
        warmup=10,
        iterations=50,
    )

    print(f"eager add+relu: p50={eager.p50_ms:.4f} ms, p95={eager.p95_ms:.4f} ms")
    print(f"fused kernel:   p50={fused.p50_ms:.4f} ms, p95={fused.p95_ms:.4f} ms")
    print("PASS: 融合Kernel与PyTorch reference一致。")
    print("解释性能时需同时考虑kernel launch数和近似global-memory traffic。")


if __name__ == "__main__":
    main()

