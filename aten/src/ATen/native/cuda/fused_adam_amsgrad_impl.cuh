#pragma once
#include <ATen/core/Tensor.h>
#include <vector>

namespace at { namespace native {

void _fused_adam_amsgrad_cuda_impl_(
    std::vector<std::vector<at::Tensor>>& nested_tensorlists,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf
);

} } // namespace at::native
