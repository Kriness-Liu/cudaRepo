"""实验00：分清Driver、Runtime、Toolkit与NVRTC。

常见误区是看到 ``nvidia-smi`` 顶部的“CUDA Version”就认为本机已安装
对应版本的CUDA Toolkit。实际上：

- NVIDIA Driver负责与GPU通信；
- PyTorch wheel可以自带CUDA Runtime和NVRTC；
- CUDA Toolkit才包含nvcc、头文件、静态库和开发工具。

本仓库使用PyTorch自带NVRTC即时编译CUDA C，并通过Driver API加载PTX。
"""

from __future__ import annotations

import json

from cudarepo.environment import collect_environment
from cudarepo.kernels import get_kernels


def main() -> None:
    report = collect_environment()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["cuda_available"]:
        print("SKIP: CUDA不可用。")
        return

    # 第一次构造会触发NVRTC编译；后续调用复用进程内缓存的CUDA Module。
    kernels = get_kernels()
    output = kernels.write_indices(8)
    print("NVRTC/Driver API smoke result:", output.cpu().tolist())
    print("PASS: 自定义CUDA Kernel已经在PyTorch当前stream上执行。")


if __name__ == "__main__":
    main()

