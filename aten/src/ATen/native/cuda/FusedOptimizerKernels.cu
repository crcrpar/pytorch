#include <cstdint>

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <c10/macros/Macros.h>


namespace at { namespace native {

enum ArgsIndex {
  ParamIdx = 0,
  GradIdx = 1,
  ExpAvgIdx = 2,
  ExpAvgSqIdx = 3,
  // MaxExpAvgSqIdx = 4,
};

template<typename T, int Depth>
struct FusedAdamFunctor {
  /// Depth up to 5
  /// args[0]: param
  /// args[1]: grad
  /// args[2]: exp_avg
  /// args[3]: exp_avg_sq
  /// args[4]: max_exp_avg_sq
  TORCH_INTERNAL_ASSERT(Depth < 5, "AMSGrad is currently not supported.");
  using opmath_t = at::opmath_type<T>;
  C10_DEVICE __forceinline__ void operator() (
      int chunk_size,
      TensorListMetadata<Depth> &tl,
      const float *inverse_grad_scaler,
      const float *lr,
      const double &beta1,
      const double &beta2,
      const double &epsilon,
      const double &weight_decay,
      const bool maximize
  ) {
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.numel_for_tensor[tensor_loc];

    T* args[Depth];
    bool all_aligned{init_args<Depth>(args, tl, chunk_idx, chunk_size, tensor_loc)};
    n -= chunk_idx * chunk_size;
    opmath_t r_args[Depth][kILP];

    for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * kILP) {
      load_args<Depth, opmath_t, T>(r_args, args, i_start, chunk_size, n);
#pragma unroll
      for (int ii = 0; ii < kILP; ii++) {
        if (inverse_grad_scaler) {
          r_args[GradIdx] *= inverse_grad_scaler;
        }
        r_args[ExpAvgIdx][ii] = beta1 * r_args[ExpAvgIdx][ii] + (1 - beta1) * r_args[ExpAvgIdx][ii];
        r_args[ExpAvgSqIdx][ii] = beta2 * r_args[ExpAvgSqIdx][ii] + (1 - beta2) * r_args[ExpAvgSqIdx] * r_args[ExpAvgSqIdx];
        opmath_t denom = sqrtf(r_args[ExpAvgSqIdx][ii]) + epsilon;
        opmath_t update = (r_args[ExpAvgIdx][ii] / denom) + (weight_decay * r_args[ParamIdx][ii]);
        opmath_t step = lr * update;
        r_args[ParamIdx][ii] = maximize ? r_args[ParamIdx][ii] + step : r_args[ParamIdx][ii] - step;
      }
      store_args<T, opmath_t>(args[ParamIdx], r_args[ParamIdx], i_start, chunk_size, n);
    }
  }
};
// TODO(mkozuki): Support AdamW.
// TODO(mkozuki): Support AMSGrad.
void _fused_adam_cuda(
    TensorList params,
    TensorList grads,
    TensorList exp_avgs,
    TensorList exp_avg_sqs,
    TensorList max_exp_avg_sqs,
    TensorList state_steps,
    at::optional<at::Tensor> inverse_grad_scaler,
    at::Tensor &lr,
    const at::Tensor &lr,
    const double &beta1,
    cosnt double &beta2,
    const double &weight_decay,
    const double &eps,
    const bool amsgrad,
    const bool maximize
) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(params.vec());
  tensor_lists.emplace_back(grads.vec());
  tensor_lists.emplace_back(exp_avgs.vec());
  tensor_lists.emplace_back(exp_avg_sqs.vec());

  if (!amsgrad) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, c10::kBFloat16, kHalf, params[0].scalar_type(), "fused_adam_cuda", [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<4>(
            tensor_lists,
            FusedAdamFunctor<scalar_t, /* depth */ 4>(),
            inverse_grad_scaler.has_value() ? inverse_grad_scaler.data_ptr<at::Float>() : nullptr,
            lr.data_ptr<at::Float>(),
            beta1,
            beta2,
            weight_decay,
            eps,
            amsgrad,
            maximize
        );
    });
  } else {
    TORCH_CHECK(false, "FusedAdam does not support AMSGrad");
  }

}

}} // namespace at::native
