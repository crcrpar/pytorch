#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/DeviceGuard.h>
#include <ATen/OpMathType.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <vector>


namespace at::native {

namespace {

constexpr uint8_t kParamIdx = 0;
constexpr uint8_t kGradIdx = 1;
constexpr uint8_t kMomIdx = 2;
constexpr uint8_t kDepth = 3;

template <typename scalar_type>
C10_DEVICE __forceinline__ void sgd_math(
    scalar_type r_args[kDepth][kILP],
    const double lr,
    const double weight_decay,
    const double momentum,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const float* grad_scale_ptr,
    const float* found_inf_ptr
) {
  using opmath_t = at::opmath_type<scalar_type>;
#pragma unroll
  for (int ii = 0; ii < kILP; ii++) {
    opmath_t param = static_cast<opmath_t>(r_args[kParamIdx][ii]);
    opmath_t grad = static_cast<opmath_t>(r_args[kGradIdx][ii]) ;
    opmath_t momentum_buffer = static_cast<opmath_t>(r_args[kMomIdx][ii]);
    if (grad_scale_ptr) {
      grad /= static_cast<double>(*grad_scale_ptr);
    }
    const opmath_t grad_to_store = grad;
    if (maximize) {
      grad = -grad;
    }
    if (is_first_step) {
      momentum_buffer = grad;
    } else {
      momentum_buffer = momentum * momentum_buffer + (1 - dampening) * grad;
    }
    if (nesterov) {
      grad += momentum * momentum_buffer;
    } else {
      grad = momentum_buffer;
    }
    param += -lr * grad;

    r_args[kParamIdx][ii] = param;
    r_args[kGradIdx][ii] = grad_to_store;
    r_args[kMomIdx][ii] = momentum_buffer;
  }
}

template <typename scalar_type>
struct FusedSGDMathFunctor {
  C10_DEVICE __forceinline__ void operator()(
      int chunk_size,
      TensorListMetadata<kDepth>& tl,
      const double lr,
      const double weight_decay,
      const double momentum,
      const double dampening,
      const bool nesterov,
      const bool maximize,
      const bool is_first_step,
      const float* grad_scale_ptr,
      const float* found_inf_ptr
  ) {
    if (found_inf_ptr && *found_inf_ptr == 1) {
      return;
    }
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.numel_for_tensor[tensor_loc];

    scalar_type* args[kDepth];
    const bool all_aligned{init_args<kDepth>(args, tl, chunk_idx, chunk_size, tensor_loc)};
    n -= tl.numel_for_tensor[tensor_loc];
    scalar_type r_args[kDepth][kILP];

    if ((n % kILP == 0) && (chunk_size % kILP == 0) && all_aligned) {
      for (int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
#pragma unroll
        for (int i = 0; i < kDepth; i++) {
          load_store(r_args[i], args[i], 0, i_start);
        }
        sgd_math(r_args, lr, weight_decay, momentum, dampening, nesterov, maximize, is_first_step, grad_scale_ptr, found_inf_ptr);
#pragma unroll
        for (int i = 0; i < kDepth; i++) {
          if (i != kGradIdx || grad_scale_ptr) {
            load_store(args[i], r_args[i], i_start, 0);
          }
        }
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * kILP) {
        load_args<kDepth>(r_args, args, i_start, chunk_size, n);
        sgd_math(r_args, lr, weight_decay, momentum, dampening, nesterov, maximize, is_first_step, grad_scale_ptr, found_inf_ptr);
#pragma unroll
        for (int i = 0; i < kDepth; i++) {
          if (i != kGradIdx || grad_scale_ptr) {
            store_args(args[i], r_args[i], i_start, chunk_size, n);
          }
        }
      }
    }
  }
};

} // namespace (anonymous)

void _fused_sgd_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double lr,
    const double weight_decay,
    const double momentum,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf
) {
  std::vector<std::vector<at::Tensor>> tensor_lists{params.vec(), grads.vec(), momentum_buffer_list.vec()};
  float* grad_scale_ptr = grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr = found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, params[0].scalar_type(),
      "fused_sgd_kernel_cuda", [&]() {
        multi_tensor_apply<kDepth>(
            tensor_lists,
            FusedSGDMathFunctor<scalar_t>(),
            lr, weight_decay, momentum, dampening,
            nesterov, maximize, is_first_step,
            grad_scale_ptr, found_inf_ptr);
        });
}

} // namespace at::native
