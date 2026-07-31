"""Validated Python wrappers around the learning CUDA kernels."""

from __future__ import annotations

from functools import lru_cache

import torch

from .nvrtc import CudaModule, compile_to_ptx, int32, int64


CUDA_SOURCE = r"""
extern "C" __global__
void write_indices(long long* output, long long count) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; index < count; index += stride) {
        output[index] = index;
    }
}

extern "C" __global__
void vector_add(
    const float* left,
    const float* right,
    float* output,
    long long count
) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; index < count; index += stride) {
        output[index] = left[index] + right[index];
    }
}

extern "C" __global__
void transpose_naive(
    const float* input,
    float* output,
    int rows,
    int columns
) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && column < columns) {
        output[column * rows + row] = input[row * columns + column];
    }
}

#define TILE_DIM 32
#define BLOCK_ROWS 8

extern "C" __global__
void transpose_tiled(
    const float* input,
    float* output,
    int rows,
    int columns
) {
    // +1 padding changes the shared-memory row stride and avoids a 32-way
    // bank conflict when threads later read the tile by columns.
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int input_column = blockIdx.x * TILE_DIM + threadIdx.x;
    int input_row = blockIdx.y * TILE_DIM + threadIdx.y;
    #pragma unroll
    for (int offset = 0; offset < TILE_DIM; offset += BLOCK_ROWS) {
        if (input_column < columns && input_row + offset < rows) {
            tile[threadIdx.y + offset][threadIdx.x] =
                input[(input_row + offset) * columns + input_column];
        }
    }
    __syncthreads();

    int output_column = blockIdx.y * TILE_DIM + threadIdx.x;
    int output_row = blockIdx.x * TILE_DIM + threadIdx.y;
    #pragma unroll
    for (int offset = 0; offset < TILE_DIM; offset += BLOCK_ROWS) {
        if (output_column < rows && output_row + offset < columns) {
            output[(output_row + offset) * rows + output_column] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

extern "C" __global__
void fused_bias_relu(
    const float* input,
    const float* bias,
    float* output,
    int rows,
    int columns
) {
    long long count = (long long)rows * columns;
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; index < count; index += stride) {
        float value = input[index] + bias[index % columns];
        output[index] = value > 0.0f ? value : 0.0f;
    }
}
"""


def _cuda_device(device: int | torch.device | None) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    result = torch.device(device)
    if result.type != "cuda":
        raise ValueError("device must be CUDA")
    if result.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return result


class CudaKernels:
    """Small set of FP32 kernels with strict interface validation."""

    def __init__(self, device: int | torch.device | None = None) -> None:
        self.device = _cuda_device(device)
        ptx = compile_to_ptx(CUDA_SOURCE, name="cudarepo_kernels.cu", device=self.device)
        self.module = CudaModule(ptx, device=self.device)

    def _check_float_tensor(self, tensor: torch.Tensor, *, name: str) -> None:
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    @staticmethod
    def _grid_1d(count: int, block: int = 256) -> tuple[int]:
        # Cap the number of blocks because every 1-D kernel uses a grid-stride
        # loop.  This keeps launch geometry reasonable for very large tensors.
        return (max(1, min(4096, (count + block - 1) // block)),)

    def write_indices(self, count: int) -> torch.Tensor:
        if count < 0:
            raise ValueError("count must be non-negative")
        output = torch.empty(count, device=self.device, dtype=torch.int64)
        if count:
            self.module.launch(
                "write_indices",
                grid=self._grid_1d(count),
                block=(256,),
                arguments=(output, int64(count)),
            )
        return output

    def vector_add(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        self._check_float_tensor(left, name="left")
        self._check_float_tensor(right, name="right")
        if left.shape != right.shape:
            raise ValueError("left and right must have the same shape")
        output = torch.empty_like(left)
        if left.numel():
            self.module.launch(
                "vector_add",
                grid=self._grid_1d(left.numel()),
                block=(256,),
                arguments=(left, right, output, int64(left.numel())),
            )
        return output

    def transpose_naive(self, matrix: torch.Tensor) -> torch.Tensor:
        self._check_matrix(matrix)
        rows, columns = matrix.shape
        output = torch.empty((columns, rows), device=self.device, dtype=torch.float32)
        if matrix.numel():
            self.module.launch(
                "transpose_naive",
                grid=((columns + 31) // 32, (rows + 7) // 8),
                block=(32, 8),
                arguments=(matrix, output, int32(rows), int32(columns)),
            )
        return output

    def transpose_tiled(self, matrix: torch.Tensor) -> torch.Tensor:
        self._check_matrix(matrix)
        rows, columns = matrix.shape
        output = torch.empty((columns, rows), device=self.device, dtype=torch.float32)
        if matrix.numel():
            self.module.launch(
                "transpose_tiled",
                grid=((columns + 31) // 32, (rows + 31) // 32),
                block=(32, 8),
                arguments=(matrix, output, int32(rows), int32(columns)),
            )
        return output

    def _check_matrix(self, matrix: torch.Tensor) -> None:
        self._check_float_tensor(matrix, name="matrix")
        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional")
        if matrix.shape[0] > 2**31 - 1 or matrix.shape[1] > 2**31 - 1:
            raise ValueError("matrix dimensions exceed the int32 kernel interface")

    def fused_bias_relu(self, inputs: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        self._check_float_tensor(inputs, name="inputs")
        self._check_float_tensor(bias, name="bias")
        if inputs.ndim != 2:
            raise ValueError("inputs must be two-dimensional")
        if bias.ndim != 1 or bias.numel() != inputs.shape[1]:
            raise ValueError("bias must be one-dimensional with inputs.shape[1] elements")
        rows, columns = inputs.shape
        output = torch.empty_like(inputs)
        if inputs.numel():
            self.module.launch(
                "fused_bias_relu",
                grid=self._grid_1d(inputs.numel()),
                block=(256,),
                arguments=(inputs, bias, output, int32(rows), int32(columns)),
            )
        return output


@lru_cache(maxsize=None)
def _get_kernels_by_index(device_index: int) -> CudaKernels:
    return CudaKernels(torch.device("cuda", device_index))


def get_kernels(device: int | torch.device | None = None) -> CudaKernels:
    """Return one compiled module per CUDA device for the current process."""

    selected = _cuda_device(device)
    assert selected.index is not None
    return _get_kernels_by_index(selected.index)

