#include <torch/extension.h>
#include <torch/library.h>

namespace {

void check_inputs(const torch::Tensor& input, const torch::Tensor& bias) {
    TORCH_CHECK(input.dim() == 2, "input must be a 2-D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1-D tensor");
    TORCH_CHECK(bias.numel() == input.size(1), "bias length must equal input.size(1)");
    TORCH_CHECK(input.device() == bias.device(), "input and bias must be on one device");
    TORCH_CHECK(input.scalar_type() == bias.scalar_type(), "input and bias dtypes must match");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat16,
        "only float32 and float16 are supported"
    );
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
}

torch::Tensor fused_bias_relu_cpu(const torch::Tensor& input, const torch::Tensor& bias) {
    check_inputs(input, bias);
    return torch::relu(input + bias);
}

}  // namespace

torch::Tensor fused_bias_relu_cuda(const torch::Tensor& input, const torch::Tensor& bias);

TORCH_LIBRARY(cudarepo_ext, m) {
    m.def("fused_bias_relu(Tensor input, Tensor bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(cudarepo_ext, CPU, m) {
    m.impl("fused_bias_relu", TORCH_FN(fused_bias_relu_cpu));
}

TORCH_LIBRARY_IMPL(cudarepo_ext, CUDA, m) {
    m.impl("fused_bias_relu", TORCH_FN(fused_bias_relu_cuda));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
