"""PyTorch-facing contract for the optional C++/CUDA extension.

The compiled module registers ``cudarepo_ext::fused_bias_relu`` with the
dispatcher.  Python adds FakeTensor and Autograd registrations so the operator
can participate in ``torch.compile`` and training without hiding its gradient
formula inside the CUDA kernel.
"""

from __future__ import annotations

import torch


_EXTENSION_ERROR: Exception | None = None

try:
    from . import _C  # type: ignore[attr-defined]  # noqa: F401
except (ImportError, OSError) as error:
    _EXTENSION_ERROR = error
else:

    @torch.library.register_fake("cudarepo_ext::fused_bias_relu")
    def _fake_fused_bias_relu(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        if input.ndim != 2 or bias.ndim != 1 or bias.shape[0] != input.shape[1]:
            raise ValueError("expected input [rows, columns] and bias [columns]")
        return torch.empty_like(input)

    def _setup_context(context, inputs, output) -> None:
        input, bias = inputs
        context.save_for_backward(input, bias)

    def _backward(context, grad_output: torch.Tensor):
        input, bias = context.saved_tensors
        grad_pre_activation = grad_output * ((input + bias) > 0).to(grad_output.dtype)
        return grad_pre_activation, grad_pre_activation.sum(dim=0)

    torch.library.register_autograd(
        "cudarepo_ext::fused_bias_relu",
        _backward,
        setup_context=_setup_context,
    )


def extension_available() -> bool:
    """Return whether the compiled dispatcher extension was imported."""

    return _EXTENSION_ERROR is None


def extension_error() -> Exception | None:
    """Return the build/import error for diagnostics."""

    return _EXTENSION_ERROR


def fused_bias_relu_extension(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Execute the registered operator or explain how to build it."""

    if _EXTENSION_ERROR is not None:
        raise RuntimeError(
            "cudarepo C++/CUDA extension is unavailable; run "
            "`python setup_extension.py build_ext --inplace` after installing "
            "a C++ compiler and the CUDA Toolkit"
        ) from _EXTENSION_ERROR
    return torch.ops.cudarepo_ext.fused_bias_relu.default(input, bias)
