#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TypeDefault.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/fused_adam_amsgrad_impl.cuh>
#include <ATen/native/cuda/fused_adam_impl.cuh>
#include <ATen/native/cuda/fused_adam_utils.cuh>
#include <c10/util/Exception.h>
#include <c10/core/DeviceGuard.h>

#include <unordered_map>


namespace at::native {

namespace {
using device_tensor_map_t = std::unordered_map<at::Device, at::Tensor>;
} // namespace (anonymous)

// note(crcrpar): To observe the CI rules, i.e. 20 minutes per file to compile, defensively split instantiations into _impl files.
// this is only for CUDA 11.3 for which it took about 20 minutes and 28 minutes in my workstation and CI, respectively.
// As a data point, it took about 20 seconds for CUDA 11.7 installed in my environment.
// See https://github.com/pytorch/pytorch/pull/81705 for details.
void _fused_adam_kernel_cuda_(
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
  auto grad_scale_map = [&](){
    if (grad_scale.has_value()) {
      return device_tensor_map_t{{grad_scale->device(), grad_scale.value()}};
    } else {
      return device_tensor_map_t{};
    }
  }();
  auto found_inf_map = [&](){
    if (found_inf.has_value()) {
      return device_tensor_map_t{{found_inf->device(), found_inf.value()}};
    } else {
      return device_tensor_map_t{};
    }
  }();
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
    const auto cur_grad_scale = [&](){
      if (grad_scale.has_value() && !grad_scale_map.count(cur_device)) {
        grad_scale_map[cur_device] = grad_scale->to(cur_device);
        return c10::optional<at::Tensor>(grad_scale_map[cur_device]);
      }
      return c10::optional<at::Tensor>();
    }();
    const auto cur_found_inf = [&](){
      if (found_inf.has_value() && !found_inf_map.count(cur_device)) {
        found_inf_map[cur_device] = found_inf->to(cur_device);
        return c10::optional<at::Tensor>(found_inf_map[cur_device]);
      }
      return c10::optional<at::Tensor>();
    }();
    for (const auto & dtype_nested_tensorlists : pair.second) {
      const auto tensors = dtype_nested_tensorlists.second;
      if (amsgrad) {
        _fused_adam_amsgrad_cuda_impl_(
            tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], lr, beta1, beta2, weight_decay, eps, maximize, cur_grad_scale, cur_found_inf);
      } else {
        _fused_adam_cuda_impl_(
            tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], lr, beta1, beta2, weight_decay, eps, maximize, cur_grad_scale, cur_found_inf);
      }
    }
  }
}

} // namespace at::native
