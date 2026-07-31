"""Minimal NVRTC and CUDA Driver API bridge.

This module deliberately avoids a C++ extension so the experiments can run on
a Windows machine that has a CUDA-enabled PyTorch wheel and NVIDIA driver but
does not yet have the full CUDA Toolkit/MSVC toolchain.

It is a learning runtime, not a production operator framework: raw pointer
launches bypass PyTorch dispatch, dtype dispatch, version counters and
Autograd.  The higher-level wrappers in :mod:`cudarepo.kernels` therefore
validate every input and compare results with PyTorch references in tests.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable, Sequence

import torch


class NvrtcError(RuntimeError):
    """NVRTC compilation or resource-management failure."""


class CudaDriverError(RuntimeError):
    """CUDA Driver API failure."""


def _load_nvrtc() -> ctypes.CDLL:
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if os.name == "nt":
        candidates = sorted(
            path
            for path in torch_lib.glob("nvrtc64_*.dll")
            if "builtins" not in path.name and ".alt." not in path.name
        )
        if not candidates:
            raise NvrtcError(f"PyTorch-bundled NVRTC was not found under {torch_lib}")
        return ctypes.WinDLL(str(candidates[-1]))

    candidates = sorted(torch_lib.glob("libnvrtc.so*"))
    if not candidates:
        raise NvrtcError(f"NVRTC was not found under {torch_lib}")
    return ctypes.CDLL(str(candidates[-1]))


def _load_driver() -> ctypes.CDLL:
    if os.name == "nt":
        return ctypes.WinDLL("nvcuda.dll")
    return ctypes.CDLL("libcuda.so.1")


class _Apis:
    """Typed function tables; explicit signatures prevent 64-bit truncation."""

    def __init__(self) -> None:
        self.nvrtc = _load_nvrtc()
        self.cuda = _load_driver()
        self._declare_nvrtc()
        self._declare_driver()

    def _declare_nvrtc(self) -> None:
        library = self.nvrtc
        library.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        library.nvrtcCreateProgram.restype = ctypes.c_int
        library.nvrtcCompileProgram.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        library.nvrtcCompileProgram.restype = ctypes.c_int
        library.nvrtcGetProgramLogSize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        library.nvrtcGetProgramLogSize.restype = ctypes.c_int
        library.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        library.nvrtcGetProgramLog.restype = ctypes.c_int
        library.nvrtcGetPTXSize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        library.nvrtcGetPTXSize.restype = ctypes.c_int
        library.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        library.nvrtcGetPTX.restype = ctypes.c_int
        library.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        library.nvrtcDestroyProgram.restype = ctypes.c_int

    def _declare_driver(self) -> None:
        library = self.cuda
        library.cuInit.argtypes = [ctypes.c_uint]
        library.cuInit.restype = ctypes.c_int
        library.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        library.cuCtxGetCurrent.restype = ctypes.c_int
        library.cuModuleLoadData.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        library.cuModuleLoadData.restype = ctypes.c_int
        library.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        library.cuModuleGetFunction.restype = ctypes.c_int
        library.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.cuLaunchKernel.restype = ctypes.c_int
        library.cuModuleUnload.argtypes = [ctypes.c_void_p]
        library.cuModuleUnload.restype = ctypes.c_int
        library.cuGetErrorName.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        library.cuGetErrorName.restype = ctypes.c_int
        library.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        library.cuGetErrorString.restype = ctypes.c_int


_APIS: _Apis | None = None


def _apis() -> _Apis:
    global _APIS
    if _APIS is None:
        _APIS = _Apis()
    return _APIS


def _check_nvrtc(code: int, operation: str) -> None:
    if code != 0:
        raise NvrtcError(f"{operation} failed with NVRTC status {code}")


def _driver_error_text(code: int) -> str:
    api = _apis().cuda
    name = ctypes.c_char_p()
    message = ctypes.c_char_p()
    api.cuGetErrorName(code, ctypes.byref(name))
    api.cuGetErrorString(code, ctypes.byref(message))
    name_text = name.value.decode(errors="replace") if name.value else "CUDA_ERROR"
    message_text = message.value.decode(errors="replace") if message.value else "unknown error"
    return f"{name_text}: {message_text}"


def _check_driver(code: int, operation: str) -> None:
    if code != 0:
        raise CudaDriverError(f"{operation} failed ({code}, {_driver_error_text(code)})")


def compile_to_ptx(
    source: str,
    *,
    name: str = "cudarepo_kernel.cu",
    device: int | torch.device | None = None,
    extra_options: Iterable[str] = (),
) -> bytes:
    """Compile CUDA C source to PTX for the selected device capability."""

    if not torch.cuda.is_available():
        raise NvrtcError("CUDA is unavailable")
    device_object = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    if device_object.type != "cuda":
        raise ValueError("device must be a CUDA device")
    index = torch.cuda.current_device() if device_object.index is None else device_object.index
    major, minor = torch.cuda.get_device_capability(index)

    api = _apis().nvrtc
    program = ctypes.c_void_p()
    _check_nvrtc(
        api.nvrtcCreateProgram(
            ctypes.byref(program),
            source.encode("utf-8"),
            name.encode("utf-8"),
            0,
            None,
            None,
        ),
        "nvrtcCreateProgram",
    )
    try:
        option_text = [
            f"--gpu-architecture=compute_{major}{minor}",
            "--std=c++14",
            *extra_options,
        ]
        encoded = [option.encode("utf-8") for option in option_text]
        options = (ctypes.c_char_p * len(encoded))(*encoded)
        status = api.nvrtcCompileProgram(program, len(encoded), options)
        if status != 0:
            log_size = ctypes.c_size_t()
            _check_nvrtc(
                api.nvrtcGetProgramLogSize(program, ctypes.byref(log_size)),
                "nvrtcGetProgramLogSize",
            )
            log = ctypes.create_string_buffer(max(1, log_size.value))
            _check_nvrtc(api.nvrtcGetProgramLog(program, log), "nvrtcGetProgramLog")
            raise NvrtcError(
                f"NVRTC compilation failed with status {status}:\n"
                f"{log.value.decode('utf-8', errors='replace')}"
            )

        ptx_size = ctypes.c_size_t()
        _check_nvrtc(api.nvrtcGetPTXSize(program, ctypes.byref(ptx_size)), "nvrtcGetPTXSize")
        ptx = ctypes.create_string_buffer(ptx_size.value)
        _check_nvrtc(api.nvrtcGetPTX(program, ptx), "nvrtcGetPTX")
        return bytes(ptx.raw)
    finally:
        if program.value:
            _check_nvrtc(api.nvrtcDestroyProgram(ctypes.byref(program)), "nvrtcDestroyProgram")


def int32(value: int) -> ctypes.c_int:
    """Create an explicitly typed 32-bit scalar kernel argument."""

    return ctypes.c_int(value)


def int64(value: int) -> ctypes.c_longlong:
    """Create an explicitly typed 64-bit scalar kernel argument."""

    return ctypes.c_longlong(value)


KernelArgument = torch.Tensor | ctypes._SimpleCData  # type: ignore[attr-defined]


class CudaModule:
    """PTX module loaded into PyTorch's current primary CUDA context."""

    def __init__(self, ptx: bytes, *, device: int | torch.device | None = None) -> None:
        if not torch.cuda.is_available():
            raise CudaDriverError("CUDA is unavailable")
        device_object = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else torch.device(device)
        )
        if device_object.type != "cuda":
            raise ValueError("device must be CUDA")
        self.device_index = (
            torch.cuda.current_device() if device_object.index is None else device_object.index
        )
        torch.cuda.set_device(self.device_index)
        # A real allocation retains the primary context on the calling thread.
        torch.empty(1, device=torch.device("cuda", self.device_index))

        api = _apis().cuda
        _check_driver(api.cuInit(0), "cuInit")
        context = ctypes.c_void_p()
        _check_driver(api.cuCtxGetCurrent(ctypes.byref(context)), "cuCtxGetCurrent")
        if not context.value:
            raise CudaDriverError("no current CUDA context after a PyTorch allocation")

        self._ptx_buffer = ctypes.create_string_buffer(ptx)
        self._module = ctypes.c_void_p()
        _check_driver(
            api.cuModuleLoadData(
                ctypes.byref(self._module),
                ctypes.cast(self._ptx_buffer, ctypes.c_void_p),
            ),
            "cuModuleLoadData",
        )
        self._functions: dict[str, ctypes.c_void_p] = {}
        self._closed = False

    def _function(self, name: str) -> ctypes.c_void_p:
        if self._closed:
            raise CudaDriverError("CUDA module is already closed")
        if name not in self._functions:
            function = ctypes.c_void_p()
            _check_driver(
                _apis().cuda.cuModuleGetFunction(
                    ctypes.byref(function),
                    self._module,
                    name.encode("utf-8"),
                ),
                f"cuModuleGetFunction({name})",
            )
            self._functions[name] = function
        return self._functions[name]

    def launch(
        self,
        name: str,
        *,
        grid: Sequence[int],
        block: Sequence[int],
        arguments: Sequence[KernelArgument],
        shared_memory_bytes: int = 0,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Launch a kernel asynchronously on a PyTorch CUDA stream."""

        if len(grid) not in (1, 2, 3) or len(block) not in (1, 2, 3):
            raise ValueError("grid and block must have one to three dimensions")
        grid_xyz = tuple(grid) + (1,) * (3 - len(grid))
        block_xyz = tuple(block) + (1,) * (3 - len(block))
        if any(value <= 0 for value in (*grid_xyz, *block_xyz)):
            raise ValueError("grid and block dimensions must be positive")
        if shared_memory_bytes < 0:
            raise ValueError("shared_memory_bytes must be non-negative")

        torch.cuda.set_device(self.device_index)
        selected_stream = (
            torch.cuda.current_stream(self.device_index) if stream is None else stream
        )
        keepalive: list[ctypes._SimpleCData] = []  # type: ignore[attr-defined]
        pointers: list[ctypes.c_void_p] = []
        for argument in arguments:
            if isinstance(argument, torch.Tensor):
                if argument.device.type != "cuda" or argument.device.index != self.device_index:
                    raise ValueError("all tensor arguments must be on the module CUDA device")
                scalar = ctypes.c_void_p(argument.data_ptr())
            elif isinstance(argument, ctypes._SimpleCData):  # type: ignore[attr-defined]
                scalar = argument
            else:
                raise TypeError(
                    "kernel scalars require an explicit type such as int32(value) or int64(value)"
                )
            keepalive.append(scalar)
            pointers.append(ctypes.cast(ctypes.byref(scalar), ctypes.c_void_p))
        parameter_array = (ctypes.c_void_p * len(pointers))(*pointers)

        _check_driver(
            _apis().cuda.cuLaunchKernel(
                self._function(name),
                *grid_xyz,
                *block_xyz,
                shared_memory_bytes,
                ctypes.c_void_p(selected_stream.cuda_stream),
                parameter_array,
                None,
            ),
            f"cuLaunchKernel({name})",
        )

    def close(self) -> None:
        """Synchronize and unload the module explicitly."""

        if self._closed:
            return
        torch.cuda.synchronize(self.device_index)
        _check_driver(_apis().cuda.cuModuleUnload(self._module), "cuModuleUnload")
        self._closed = True

