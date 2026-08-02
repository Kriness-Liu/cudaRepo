from __future__ import annotations

import unittest

import torch

from cudarepo.extension import extension_available, fused_bias_relu_extension


@unittest.skipUnless(extension_available(), "C++/CUDA extension is not built")
class ExtensionTests(unittest.TestCase):
    def test_forward_matches_reference_for_supported_dtypes(self) -> None:
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float32, torch.float16):
                with self.subTest(device=device, dtype=dtype):
                    inputs = torch.randn(17, 33, device=device, dtype=dtype)
                    bias = torch.randn(33, device=device, dtype=dtype)
                    torch.testing.assert_close(
                        fused_bias_relu_extension(inputs, bias),
                        torch.relu(inputs + bias),
                    )

    def test_autograd_matches_reference(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = torch.randn(8, 16, device=device, requires_grad=True)
        bias = torch.randn(16, device=device, requires_grad=True)
        output = fused_bias_relu_extension(inputs, bias)
        output.sum().backward()
        actual_grads = (inputs.grad.detach().clone(), bias.grad.detach().clone())
        inputs.grad = None
        bias.grad = None
        torch.relu(inputs + bias).sum().backward()
        torch.testing.assert_close(actual_grads[0], inputs.grad)
        torch.testing.assert_close(actual_grads[1], bias.grad)


if __name__ == "__main__":
    unittest.main()
