# cudarepo：CUDA Kernel 与性能工程实验

这是一个面向 AI Infra 初学者的 CUDA 工程仓库。它从两条路径解释 PyTorch 算子如何落到 GPU：

```text
路径 A：CUDA C -> NVRTC -> PTX -> CUDA Driver API -> 当前 PyTorch Stream
路径 B：CUDA/C++ -> NVCC + CMake/Extension -> PyTorch Dispatcher -> Autograd/torch.compile
```

路径 A 在没有完整 CUDA Toolkit 时也能学习 Kernel 启动、访存和计时；路径 B 则补齐标准工程中的编译、Dispatcher 注册、FakeTensor、Autograd 与 Profiler 接入。

## 学习目标

- 理解 Grid/Block/Thread、边界判断和 grid-stride loop。
- 分析合并访存、Shared Memory、Bank Conflict 和 Tiled Transpose。
- 使用 CUDA Stream/Event、Pinned Memory 与异步 H2D 构造传输—计算流水线。
- 通过 Kernel Fusion 减少中间 Tensor 与全局内存往返。
- 使用 CUDA Event、Warm-up、P50/P95 和有效带宽建立可复现 benchmark。
- 编写 PyTorch C++/CUDA Extension，并为自定义算子注册 Autograd 与 FakeTensor。
- 使用 PyTorch Profiler、NVTX、Nsight Systems/Compute 区分调度、访存和计算瓶颈。

## 目录

```text
cudarepo/
├─ csrc/                  # C++ Dispatcher 注册与 CUDA Kernel
├─ src/cudarepo/
│  ├─ nvrtc.py            # NVRTC、PTX、Driver API 与 Kernel Launch
│  ├─ kernels.py          # 输入校验及 NVRTC Kernel 包装
│  ├─ extension.py        # Extension、FakeTensor 与 Autograd 注册
│  └─ benchmark.py        # CUDA Event、P50/P95、有效带宽
├─ labs/                  # 从执行模型到 C++/CUDA Extension
├─ scripts/               # 环境检查、基准和 Profiler 入口
├─ tests/                 # Reference、边界与梯度测试
├─ docs/                  # Extension 构建和 Nsight 操作说明
└─ results/               # 带硬件环境的实测结果
```

## 环境与安装

```powershell
conda create -n cudarepo python=3.12 -y
conda activate cudarepo
# 先按显卡驱动安装 CUDA 版 PyTorch
python -m pip install -e .
python scripts/check_env.py
```

PyTorch wheel 携带 CUDA Runtime/NVRTC，不代表机器已经安装 NVCC、CMake 和 MSVC。运行前五个实验只需要 CUDA 版 PyTorch；构建原生 Extension 需要完整工具链。

## 推荐学习顺序

```powershell
python labs/00_environment.py
python labs/01_execution_model.py
python labs/02_memory_coalescing.py
python labs/03_streams_events.py
python labs/04_kernel_fusion.py
python -m unittest discover -s tests -v
python scripts/run_benchmarks.py
```

安装 CUDA Toolkit、CMake、Ninja 和兼容的 C++ 编译器后，再进入标准 Extension 路径：

```powershell
python -m pip install -e ".[extension]"
python setup_extension.py build_ext --inplace
python labs/05_cpp_cuda_extension.py
python scripts/run_extension_benchmarks.py
python scripts/profile_extension.py
```

也可以使用 `CMakeLists.txt` 构建；具体命令和 Windows 注意事项见 [`docs/CPP_EXTENSION.md`](docs/CPP_EXTENSION.md)。Nsight Systems/Compute 的采集与解读见 [`docs/NSIGHT.md`](docs/NSIGHT.md)。

## 已实现内容

| 实验/算子 | 核心问题 | 路径 |
|---|---|---|
| `write_indices` | Thread 到数据元素的映射、非整除边界 | NVRTC |
| `vector_add` | 基础合并访存与 Kernel Launch | NVRTC |
| `transpose_naive/tiled` | 32×33 Tile、Shared Memory、Bank Conflict | NVRTC |
| `fused_bias_relu` | 两算子融合、减少中间 Tensor 访存 | NVRTC |
| `cudarepo_ext::fused_bias_relu` | Dispatcher、FP32/FP16、Autograd、FakeTensor | C++/CUDA Extension |
| Extension benchmark | shape/dtype 扫描、Eager/NVRTC/Extension 对照 | CUDA Event |
| Extension profile | NVTX 区间与 Nsight/PyTorch Profiler | Profiling |

## Benchmark 规则

1. JIT 编译、CUDA Context 和 allocator 冷启动不计入稳态 Kernel 基准。
2. GPU 时间使用 CUDA Event；未同步的 CPU wall time主要表示 enqueue 开销。
3. 报告 P50/P95、shape、dtype、预热、采样次数及软硬件版本。
4. Kernel-only 与包含 H2D/D2H 的端到端时间分开解释。
5. 先用 PyTorch Reference 和梯度检查验证正确性，再讨论性能。
6. 不跨机器设置固定性能断言，也不根据单次最好成绩写加速倍数。

仓库保留了一份本机 NVRTC 实测结果：[`results/gtx1650_windows_cu128.json`](results/gtx1650_windows_cu128.json)。`results/latest.json` 与 Extension 临时结果不纳入版本控制。

## 当前边界

- 本机若没有 NVCC/CMake/MSVC，只能执行 NVRTC 路径；源码存在不等于原生 Extension 已在该机器编译验证。
- 当前实验为单机单卡，不覆盖 NCCL、多机、InfiniBand、RoCE、RDMA 或 GPUDirect。
- `Pinned Memory + non_blocking=True` 只是异步传输的必要条件；是否与计算重叠仍取决于 Copy Engine、数据规模和调度。
