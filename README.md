# cudarepo

一个从真实 CUDA Kernel 出发学习 GPU 执行与性能工程的实验仓库。仓库不把
普通 PyTorch 调用包装成“CUDA 项目”，而是走下面这条可检查的数据路径：

```text
CUDA C source
  -> PyTorch wheel 内置 NVRTC
  -> PTX
  -> CUDA Driver API
  -> PyTorch 当前 CUDA Context / Stream
  -> torch.Tensor.data_ptr()
```

因此，即使 Windows 机器暂时没有完整 CUDA Toolkit、NVCC、CMake 和 MSVC，
也可以编译并启动自定义 CUDA Kernel。后续安装标准工具链后，再把这些
Kernel 迁移到 PyTorch C++/CUDA Extension。

## 学习目标

- 手算 `blockIdx`、`threadIdx`、global index 和 grid-stride loop。
- 理解连续全局内存访问、合并访存、shared memory 和 bank conflict。
- 使用 CUDA Stream、Event、Pinned Memory 和异步 H2D 拷贝构建流水线。
- 用 Kernel Fusion 减少中间 Tensor 和全局内存往返。
- 用 warm-up、CUDA Event、同步边界、P50/P95 和有效带宽做可复现基准。
- 先用 PyTorch Reference 验证正确性，再讨论性能。

## 仓库结构

```text
cudarepo/
├─ src/cudarepo/
│  ├─ nvrtc.py          # NVRTC编译、PTX加载、Driver API Kernel启动
│  ├─ kernels.py        # 输入校验和自定义Kernel封装
│  ├─ benchmark.py      # CUDA Event、P50/P95、有效带宽
│  └─ environment.py    # 可复现实验环境快照
├─ labs/                # 按顺序阅读和修改的教学实验
├─ tests/               # PyTorch Reference与边界测试
├─ scripts/             # 环境检查和统一benchmark
└─ results/             # 带硬件环境的实测结果
```

## 当前实验

| 实验 | 主要内容 | 需要GPU | 需要NVCC |
|---|---|---:|---:|
| `00_environment.py` | Driver、Runtime、NVRTC、Compute Capability | 是 | 否 |
| `01_execution_model.py` | Grid、Block、Thread、边界判断、grid-stride loop | 是 | 否 |
| `02_memory_coalescing.py` | Naive/Tiled矩阵转置、shared-memory padding | 是 | 否 |
| `03_streams_events.py` | Pinned Memory、异步拷贝、双Stream流水 | 是 | 否 |
| `04_kernel_fusion.py` | Fused Bias+ReLU与Eager两算子路径 | 是 | 否 |

## 1. 创建独立环境

不要把依赖安装进默认 Anaconda 环境：

```powershell
conda create -n cudarepo python=3.12 -y
conda activate cudarepo
```

按照 PyTorch 官网与你的显卡驱动匹配的命令安装 CUDA 版 PyTorch，然后：

```powershell
python -m pip install -e .
python scripts/check_env.py
```

仅安装 CPU 版 PyTorch 无法运行这些实验。PyTorch 显示的 CUDA Runtime
版本也不等于本机安装了完整 CUDA Toolkit。

## 2. 按顺序运行

```powershell
python labs/00_environment.py
python labs/01_execution_model.py
python labs/02_memory_coalescing.py
python labs/03_streams_events.py
python labs/04_kernel_fusion.py
python -m unittest discover -s tests -v
python scripts/run_benchmarks.py
```

每次只改变一个变量，例如矩阵尺寸、block大小、dtype或stream数量；先预测
结果，再运行断言，并记录环境、P50/P95、正确性和实验边界。

仓库内保留了一份本机可复现结果：
[`results/gtx1650_windows_cu128.json`](results/gtx1650_windows_cu128.json)。
它记录了GPU、PyTorch/CUDA Runtime、shape、预热次数、采样次数、P50/P95
和实验范围；`results/latest.json`用于每次重跑，不纳入版本控制。

## 已实现的 Kernel

- `write_indices`：展示线程到数据元素的映射和 grid-stride loop。
- `vector_add`：基础合并访存 Kernel。
- `transpose_naive`：直接转置，输出写入呈跨步访问。
- `transpose_tiled`：`32 x 33` shared-memory tile，通过 `+1` padding 避免
  转置读取时的 32-way bank conflict。
- `fused_bias_relu`：在单个 Kernel 内完成 Bias Add 与 ReLU，避免中间
  Tensor 的一次写回和再次读取。

所有公开包装都检查 CUDA device、FP32 dtype、连续性和 shape；测试覆盖
非 tile 整除尺寸，并与 PyTorch Reference 使用 `torch.testing.assert_close`
对齐。

## 性能实验规则

1. JIT 编译、CUDA Context 初始化和 allocator 冷启动不计入 Kernel 基准。
2. 设备端耗时使用 CUDA Event；未同步的 CPU wall time只代表enqueue成本。
3. 报告 P50/P95，不使用单次最好成绩。
4. 端到端传输时间和 Kernel-only 时间分开解释。
5. 性能不设置跨机器单元测试阈值；硬件、驱动和 WDDM 调度都会影响结果。
6. 没有正确性对照、原始结果和环境快照，就不写“加速倍数”。

## 真实性边界

- 这是 NVRTC + CUDA Driver API 学习运行时，不是完整 PyTorch Dispatcher
  或 C++ Extension；原始指针启动不会自动接入 Autograd。
- 当前单卡实验不覆盖 NCCL、多机、InfiniBand、RoCE、RDMA 或 GPUDirect。
- NVRTC DLL 来自 PyTorch wheel，其路径和名称可能随版本变化；启动时会
  自动发现并在缺失时明确报错。
- `Pinned Memory + non_blocking=True` 只提供异步传输的必要条件；是否真正
  与计算重叠取决于硬件 copy engine、数据规模和调度，实验不会假定必然加速。
