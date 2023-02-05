#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TypeDefault.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/fused_adamw_amsgrad_impl.cuh>
#include <ATen/native/cuda/fused_adamw_impl.cuh>
#include <ATen/native/cuda/fused_adam_utils.cuh>
#include <c10/util/Exception.h>
#include <c10/core/DeviceGuard.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_sub_native.h>
#endif

#include <vector>


namespace at { namespace native {

// note(crcrpar): To observe the CI rules, i.e. 20 minutes per file to compile, defensively split instantiations into _impl files.
// this is only for CUDA 11.3 for which it took about 20 minutes and 28 minutes in my workstation and CI, respectively.
// As a data point, it took about 20 seconds for CUDA 11.7 installed in my environment.
// See https://github.com/pytorch/pytorch/pull/81705 for details.
void _fused_adamw_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf
) {
  auto grad_scale_map = init_device_tensor_map(grad_scale);
  auto found_inf_map = init_device_tensor_map(found_inf);
  OptionalDeviceGuard device_guard;

  auto nested_tensorlists = [&]() -> ArrayRef<TensorList> {
    if (amsgrad) {
      return {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs};
    }
    return {params, grads, exp_avgs, exp_avg_sqs};
  }();

  for (const auto & pair : group_tensorlists(nested_tensorlists, /* has_step_list */ true)) {
    const auto cur_device = pair.first;
    device_guard.reset_device(cur_device);
    const auto cur_grad_scale = get_optional_tensor(grad_scale_map, grad_scale, cur_device);
    const auto cur_found_inf = get_optional_tensor(found_inf_map, found_inf, cur_device);

    for (const auto & dtype_nested_tensorlists : pair.second) {
      const auto tensors = dtype_nested_tensorlists.second;
      std::vector<std::vector<at::Tensor>> nested_tensorlists{tensors.begin(), tensors.begin() + 3 + static_cast<int>(amsgrad)};
      auto cur_state_steps = tensors[4 + static_cast<int>(amsgrad)];
      at::native::foreach_tensor_add_scalar_kernel_cuda_(cur_state_steps, 1);
      if (amsgrad) {
        _fused_adamw_amsgrad_cuda_impl_(
            nested_tensorlists, cur_state_steps, lr, beta1, beta2, weight_decay, eps, maximize, cur_grad_scale, cur_found_inf);
      } else {
        _fused_adamw_cuda_impl_(
            nested_tensorlists, cur_state_steps, lr, beta1, beta2, weight_decay, eps, maximize, cur_grad_scale, cur_found_inf);
      }
      if (found_inf.has_value()) {
        at::native::foreach_tensor_sub_list_kernel_cuda_(cur_state_steps, std::vector<at::Tensor>(cur_state_steps.size(), cur_found_inf.value()));
      }
    }
  }
}

}} // namespace at::native
