#include <algorithm>
#include <cstdint>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t>
__global__ void fused_bias_relu_kernel(
    const scalar_t* input,
    const scalar_t* bias,
    scalar_t* output,
    int64_t count,
    int64_t columns
) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; index < count; index += stride) {
        scalar_t value = input[index] + bias[index % columns];
        output[index] = value > scalar_t(0) ? value : scalar_t(0);
    }
}

}  // namespace

torch::Tensor fused_bias_relu_cuda(const torch::Tensor& input, const torch::Tensor& bias) {
    TORCH_CHECK(input.is_cuda() && bias.is_cuda(), "CUDA tensors are required");
    TORCH_CHECK(input.dim() == 2 && bias.dim() == 1, "expected a 2-D input and 1-D bias");
    TORCH_CHECK(bias.numel() == input.size(1), "bias length must equal input.size(1)");
    TORCH_CHECK(input.device() == bias.device(), "input and bias must be on one CUDA device");
    TORCH_CHECK(input.scalar_type() == bias.scalar_type(), "input and bias dtypes must match");
    TORCH_CHECK(input.is_contiguous() && bias.is_contiguous(), "inputs must be contiguous");

    c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty_like(input);
    const int64_t count = input.numel();
    if (count == 0) {
        return output;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>(4096, (count + threads - 1) / threads));
    const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fused_bias_relu_cuda", [&] {
        fused_bias_relu_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            input.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            count,
            input.size(1)
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
