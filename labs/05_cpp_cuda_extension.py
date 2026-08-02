"""Experiment 05: dispatcher, Autograd and compile integration."""

from __future__ import annotations

import torch

from cudarepo.extension import extension_available, extension_error, fused_bias_relu_extension


def main() -> None:
    if not extension_available():
        print("SKIP: extension is not built:", extension_error())
        print("Build with: python setup_extension.py build_ext --inplace")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = torch.randn(64, 256, device=device, requires_grad=True)
    bias = torch.randn(256, device=device, requires_grad=True)
    actual = fused_bias_relu_extension(inputs, bias)
    reference = torch.relu(inputs + bias)
    torch.testing.assert_close(actual, reference)

    actual.sum().backward()
    actual_input_grad = inputs.grad.detach().clone()
    actual_bias_grad = bias.grad.detach().clone()
    inputs.grad = None
    bias.grad = None
    reference.sum().backward()
    torch.testing.assert_close(actual_input_grad, inputs.grad)
    torch.testing.assert_close(actual_bias_grad, bias.grad)

    torch.library.opcheck(
        torch.ops.cudarepo_ext.fused_bias_relu.default,
        (inputs.detach(), bias.detach()),
    )
    compiled = torch.compile(fused_bias_relu_extension)
    torch.testing.assert_close(compiled(inputs.detach(), bias.detach()), reference.detach())
    print("PASS: forward, Autograd, opcheck and torch.compile agree with PyTorch.")


if __name__ == "__main__":
    main()
