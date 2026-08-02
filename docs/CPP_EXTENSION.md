# PyTorch C++/CUDA Extension

## Toolchain

The NVRTC labs only require a CUDA-enabled PyTorch wheel and an NVIDIA driver. The compiled extension additionally requires:

- a CUDA Toolkit containing `nvcc`;
- a compiler supported by that Toolkit (MSVC on Windows or GCC/Clang on Linux);
- CMake or the Python `BuildExtension` path;
- Ninja for the default fast build backend.

Build in place:

```powershell
python -m pip install -e ".[extension]"
python setup_extension.py build_ext --inplace
python labs/05_cpp_cuda_extension.py
python -m unittest tests.test_extension -v
```

The operator is registered as `cudarepo_ext::fused_bias_relu`. C++ defines the schema and CPU/CUDA kernels; Python registers FakeTensor and Autograd formulas. Tests separately cover numerical correctness, gradients, dispatcher metadata and `torch.compile` composition.

## CMake path

```powershell
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake --config Release
```

Copy the generated `_C.pyd`/`_C.so` beside `src/cudarepo/extension.py`, or load it explicitly with `torch.ops.load_library`.

The checked-in source is portable, but a successful build is intentionally not claimed until the local compiler and CUDA Toolkit are recorded.
