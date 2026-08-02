"""Build the optional PyTorch C++/CUDA extension in place."""

from __future__ import annotations

import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


if os.name == "nt":
    cxx_flags = ["/O2"]
else:
    cxx_flags = ["-O3"]

setup(
    name="cudarepo-extension",
    version="0.2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=[
        CUDAExtension(
            name="cudarepo._C",
            sources=[
                "csrc/fused_bias_relu.cpp",
                "csrc/fused_bias_relu_cuda.cu",
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
