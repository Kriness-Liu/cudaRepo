"""实验01：理解Grid、Block、Thread与grid-stride loop。

对于一维问题，线程的第一个全局索引为：

    blockIdx.x * blockDim.x + threadIdx.x

当数据规模大于一次Grid能覆盖的元素时，每个线程再按
``blockDim.x * gridDim.x``步进。这个grid-stride loop让同一个Kernel既能
处理小Tensor，也能处理非常大的Tensor。
"""

from __future__ import annotations

import torch

from cudarepo.kernels import get_kernels


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA不可用。")
        return

    count = 1031  # 故意不取256的整数倍，验证尾部边界判断。
    block_size = 256
    grid_size = (count + block_size - 1) // block_size
    print(f"count={count}, block={block_size}, minimum grid={grid_size}")

    # 手算几个线程的global index。
    for block_index, thread_index in [(0, 0), (0, 255), (1, 0), (4, 6)]:
        global_index = block_index * block_size + thread_index
        print(
            f"blockIdx.x={block_index}, threadIdx.x={thread_index}"
            f" -> global index={global_index}"
        )

    output = get_kernels().write_indices(count)
    reference = torch.arange(count, device="cuda", dtype=torch.int64)
    torch.testing.assert_close(output, reference)
    print("tail:", output[-8:].cpu().tolist())
    print("PASS: 非block整除尺寸下，索引与边界判断均正确。")


if __name__ == "__main__":
    main()

