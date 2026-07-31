from __future__ import annotations

import unittest

import torch

from cudarepo.kernels import get_kernels


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class CudaKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kernels = get_kernels()

    def test_indices_cover_non_block_multiple(self) -> None:
        actual = self.kernels.write_indices(1031)
        expected = torch.arange(1031, device="cuda", dtype=torch.int64)
        torch.testing.assert_close(actual, expected)

    def test_vector_add_matches_torch(self) -> None:
        left = torch.randn(1009, device="cuda")
        right = torch.randn_like(left)
        torch.testing.assert_close(self.kernels.vector_add(left, right), left + right)

    def test_transpose_handles_non_tile_shape(self) -> None:
        matrix = torch.randn(65, 97, device="cuda")
        expected = matrix.t().contiguous()
        torch.testing.assert_close(self.kernels.transpose_naive(matrix), expected)
        torch.testing.assert_close(self.kernels.transpose_tiled(matrix), expected)

    def test_fused_bias_relu_matches_torch(self) -> None:
        inputs = torch.randn(17, 33, device="cuda")
        bias = torch.randn(33, device="cuda")
        expected = torch.relu(inputs + bias)
        torch.testing.assert_close(self.kernels.fused_bias_relu(inputs, bias), expected)

    def test_interface_rejects_wrong_dtype_and_non_contiguous(self) -> None:
        with self.assertRaises(TypeError):
            self.kernels.vector_add(
                torch.ones(8, device="cuda", dtype=torch.float64),
                torch.ones(8, device="cuda", dtype=torch.float64),
            )
        with self.assertRaises(ValueError):
            self.kernels.transpose_tiled(torch.randn(8, 8, device="cuda").t())


if __name__ == "__main__":
    unittest.main()

