#include <ATen/native/cuda/fused_adam_amsgrad_impl.cuh>

#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/fused_adam_utils.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <vector>

namespace at::native {

void _fused_adam_amsgrad_cuda_impl_(
    std::vector<std::vector<at::Tensor>>& nested_tensorlists,
    TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf
) {
  TORCH_CHECK(nested_tensorlists.size() == 5, "nested_tensorlists must consists of params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs");

  float* grad_scale_ptr = grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr = found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, nested_tensorlists[0][0].scalar_type(),
      "fused_adam_kernel_cuda", [&]() {
        multi_tensor_apply_for_fused_optimizer<5>(
            nested_tensorlists,
            state_steps,
            FusedAdamMathFunctor<scalar_t, 5>(),
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            /* amsgrad */true,
            grad_scale_ptr,
            found_inf_ptr,
            ADAM_MODE::ORIGINAL);
        });
}

} // namespace at::native
