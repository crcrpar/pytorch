#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TypeDefault.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/fused_adam_amsgrad_impl.cuh>
#include <ATen/native/cuda/fused_adam_impl.cuh>
#include <c10/util/Exception.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_sub_native.h>
#endif

#include <unordered_map>
#include <vector>

namespace at { namespace native {

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
  std::unordered_map<at::Device, at::Tensor> dev_grad_scale, dev_found_inf;
  const auto has_grad_scale{grad_scale.has_value()}, has_found_inf{found_inf.has_value()};
  TORCH_CHECK_EQ(has_grad_scale, has_found_inf);
  if (has_grad_scale) {
    dev_grad_scale.insert({grad_scale->device(), *grad_scale});
  }
  if (has_found_inf) {
    dev_found_inf.insert({found_inf->device(), *found_inf});
  }
  if (amsgrad) {
    const auto & device_dtype_tensorlists = group_tensors_by_device_and_dtype({params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs}, state_steps);
    for (const auto & pair : device_dtype_tensorlists) {
      const auto & cur_device = pair.first;
      if (has_grad_scale) {
        if (!dev_grad_scale.count(cur_device)) {
          dev_grad_scale.insert({cur_device, grad_scale->to(grad_scale->options().device(cur_device), /* non_blocking */true, /* copy */true)});
        }
        if (!dev_found_inf.count(cur_device)) {
          dev_found_inf.insert({cur_device, found_inf->to(found_inf->options().device(cur_device), /* non_blocking */true, /* copy */true)});
        }
      }
      for (const auto & dtype_tensorlists : pair.second) {
        const auto & tensorlists = dtype_tensorlists.second;
        foreach_tensor_add_scalar_kernel_cuda_(tensorlists[5], 1);
        _fused_adam_cuda_impl_(
            /*params*/tensorlists[0],
            /*grads*/tensorlists[1],
            /*exp_avgs*/tensorlists[2],
            /*exp_avg_sqs*/tensorlists[3],
            /*max_exp_avg_sqs*/tensorlists[4],
            /*state_steps*/tensorlists[5],
            lr, beta1, beta2, weight_decay, eps, amsgrad, maximize,
            has_grad_scale ? dev_grad_scale[cur_device] : grad_scale,
            has_found_inf ? dev_found_inf[cur_device] : found_inf);
        if (has_found_inf) {
          foreach_tensor_sub_list_kernel_cuda_(tensorlists[5], std::vector<at::Tensor>(tensorlists[5].size(), dev_found_inf[cur_device]));
        }
      }
    }
  } else {
    const auto & device_dtype_tensorlists = group_tensors_by_device_and_dtype({params, grads, exp_avgs, exp_avg_sqs}, state_steps);
    for (const auto & pair : device_dtype_tensorlists) {
      const auto & cur_device = pair.first;
      if (has_grad_scale) {
        if (!dev_grad_scale.count(cur_device)) {
          dev_grad_scale.insert({cur_device, grad_scale->to(grad_scale->options().device(cur_device), /* non_blocking */true, /* copy */true)});
        }
        if (!dev_found_inf.count(cur_device)) {
          dev_found_inf.insert({cur_device, found_inf->to(found_inf->options().device(cur_device), /* non_blocking */true, /* copy */true)});
        }
      }
      for (const auto & dtype_tensorlists : pair.second) {
        const auto & tensorlists = dtype_tensorlists.second;
        foreach_tensor_add_scalar_kernel_cuda_(tensorlists[4], 1);
        _fused_adam_cuda_impl_(
            /*params*/tensorlists[0],
            /*grads*/tensorlists[1],
            /*exp_avgs*/tensorlists[2],
            /*exp_avg_sqs*/tensorlists[3],
            /*state_steps*/tensorlists[4],
            lr, beta1, beta2, weight_decay, eps, amsgrad, maximize,
            has_grad_scale ? dev_grad_scale[cur_device] : grad_scale,
            has_found_inf ? dev_found_inf[cur_device] : found_inf);
        if (has_found_inf) {
          foreach_tensor_sub_list_kernel_cuda_(tensorlists[4], std::vector<at::Tensor>(tensorlists[4].size(), dev_found_inf[cur_device]));
        }
      }
    }
  }
}

}} // namespace at::native
