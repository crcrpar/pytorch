#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>

#include <iostream>
#include <tuple>

namespace at {
namespace native {

template<typename T, int NormType, int depth=1, int r_args_depth=1, int res_arg_index=0>
struct LpNormFunctor {
  static_assert(NormType == 1 || NormType == 2, "LpNormFunctor supports only L1 and L2 norm");
  using opmath_t = typename at::opmath_type<T>;
  __device__ __forceinline__ void operator() (
      int chunk_size,
      TensorListMetadata<depth>& tl,
      opmath_t* output_per_tensor,
      opmath_t* output,
      const int max_chunks_per_tensor,
      const bool per_tensor
  ) {
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.numel_for_tensor[tensor_loc];

    T* x = (T*)tl.addresses[0][tensor_loc];
    x += chunk_idx * chunk_size;
    n -= chunk_idx * chunk_size;

    __shared__ opmath_t s_vals[512];
    opmath_t vals[kILP];
    T r_x[kILP];
    for (int i = 0; i < kILP; i++) {
      vals[i] = opmath_t(0);
      r_x[i] = T(0);
    }

    if (n % kILP == 0 && chunk_size & kILP == 0 && is_aligned(x)) {
      for (int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
        // load
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          const opmath_t next = static_cast<opmath_t>(r_x[ii]);
          vals[ii] += NormType == 1 ? ::abs(next)  : next * next;
        }
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * kILP) {
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            const opmath_t next = static_cast<opmath_t>(x[i]);
            vals[ii] += NormType == 1 ? ::abs(next) : next * next;
          }
        }
      }
    }

    auto val = opmath_t(0);
    for (int i = 0; i < kILP; i++) {
      val += vals[i];
    }
    auto final = at::native::cuda_utils::BlockReduceSum<opmath_t>(val, s_vals);

    if (threadIdx.x == 0) {
      output_per_tensor[(tl.start_tensor_this_launch + tensor_loc) * max_chunks_per_tensor + chunk_idx] = final;
    }
  }
};

template<typename T, int NormType, typename opmath_t = at::opmath_type<T>>
__global__ void lpnorm_cleanup(
    opmath_t* output_per_tensor,
    T* ret_per_tensor,
    opmath_t* output,
    T* ret,
    int max_chunks_per_tensor,
    const bool per_tensor
  ) {
  __shared__ opmath_t vals[512];

  opmath_t* output_this_tensor = output_per_tensor + blockIdx.x*max_chunks_per_tensor;
  opmath_t val = 0;
  for (int i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x) {
    val += output_this_tensor[i];
  }
  opmath_t final = at::native::cuda_utils::BlockReduceSum<opmath_t>(val, vals);
  if(threadIdx.x == 0) {
    ret_per_tensor[blockIdx.x] = NormType == 1 ? final : ::sqrt(final);
  }
}

// note(mkozuki): Why excluding Int and Complex from fast path
// - Int: at::norm does not support.
// - Complex: __shfl_down_sync does not support complex and foreach does not support functions whose inputs dtypes and output dtype are different.
std::tuple<Tensor, Tensor> foreach_tensor_norm_cuda_impl(TensorList tensors, const double p, const bool per_tensor) {
  const int ntensors = tensors.size();
  int max_chunks_per_tensor = -1;

  for (int t = 0; t < ntensors; t++) {
    int max_chunks_this_tensor = (tensors[0][t].numel() + kChunkSize - 1) / kChunkSize;
    if(max_chunks_this_tensor > max_chunks_per_tensor) {
      max_chunks_per_tensor = max_chunks_this_tensor;
    }
  }
  const auto options = tensors[0].options();
  const auto opmath_type_options = options.dtype(toOpMathType(tensors[0].scalar_type()));
  Tensor output_per_tensor, ret_per_tensor;
  if (per_tensor) {
    output_per_tensor = at::zeros({ntensors*max_chunks_per_tensor}, opmath_type_options);
    ret_per_tensor = at::empty({ntensors}, options);
  }
  Tensor output, ret;
  if (!per_tensor) {
    output = at::empty({320}, opmath_type_options);
    auto ret = at::empty({}, options);
  }

  auto tensor_lists = std::vector<std::vector<Tensor>>{tensors.vec()};
  if (p == static_cast<double>(1)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, tensor_lists[0][0].scalar_type(), "foreach_tensor_norm_cuda", [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        std::cout << "output's dtype " << CppTypeToScalarType<opmath_t>::value << "\n";
        std::cout << "inputs's dytpe " << CppTypeToScalarType<scalar_t>::value << "\n";
        std::cout << "getting output_per_tensor\n";
        auto output_per_tensor_ptr = output_per_tensor.data_ptr<opmath_t>();
        std::cout << "getting output\n";
        auto output_ptr = output.data_ptr<opmath_t>();
        std::cout << "Calling multi_tensor_apply with `p` of " << p << "\n";
        multi_tensor_apply<1>(
          tensor_lists,
          LpNormFunctor<scalar_t, 1>(),
          // output_per_tensor.data_ptr<opmath_t>(),
          output_per_tensor_ptr,
          // output.data_ptr<opmath_t>(),
          output_ptr,
          max_chunks_per_tensor,
          per_tensor);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const at::cuda::OptionalCUDAGuard device_guard(device_of(output_per_tensor));
        auto stream = at::cuda::getCurrentCUDAStream();
        std::cout << "Calling lpnorm_cleanup\n";
        lpnorm_cleanup<scalar_t, 1><<<ntensors, 512, 0, stream>>>(
          output_per_tensor.data_ptr<opmath_t>(),
          ret_per_tensor.data_ptr<scalar_t>(),
          output.data_ptr<opmath_t>(),
          ret.data_ptr<scalar_t>(),
          max_chunks_per_tensor,
          per_tensor);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  } else if (p == static_cast<double>(2)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, tensor_lists[0][0].scalar_type(), "foreach_tensor_norm_cuda", [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        std::cout << "Calling multi_tensor_apply with `p` of " << p << "\n";
        multi_tensor_apply<1>(
          tensor_lists,
          LpNormFunctor<scalar_t, 2>(),
          output_per_tensor.data_ptr<opmath_t>(),
          output.data_ptr<opmath_t>(),
          max_chunks_per_tensor,
          per_tensor);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const at::cuda::OptionalCUDAGuard device_guard(device_of(output_per_tensor));
        auto stream = at::cuda::getCurrentCUDAStream();
        std::cout << "Calling lpnorm_cleanup\n";
        lpnorm_cleanup<scalar_t, 2><<<ntensors, 512, 0, stream>>>(
          output_per_tensor.data_ptr<opmath_t>(),
          ret_per_tensor.data_ptr<scalar_t>(),
          output.data_ptr<opmath_t>(),
          ret.data_ptr<scalar_t>(),
          max_chunks_per_tensor,
          per_tensor);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  } else {
    AT_ERROR("foreach_tensor_norm_cuda fast path got unexpected ord value: ", p);
  }

  return {ret_per_tensor, ret};
}

double ord_to_p(const Scalar &ord) {
  double p;
  if (ord.isIntegral(false)) {
    p = ord.to<int64_t>();
  } else if (ord.isFloatingPoint()) {
    p = ord.to<double>();
  } else {
    AT_ERROR("foreach_tensor_norm_cuda expects ord to be integer or float");
  }
  return p;
}

bool has_int_or_complex(TensorList tensors) {
  return std::any_of(tensors.begin(), tensors.end(), [](const auto &t) -> bool {
      const auto scalar_type = t.scalar_type();
      return at::isIntegralType(scalar_type, /*includeBool*/true) || at::isComplexType(scalar_type);
  });
}

bool should_go_slowpath(TensorList tensors, const double p) {
  return !can_use_fast_route(tensors) || has_int_or_complex(tensors) || !(p == static_cast<double>(1) || p == static_cast<double>(2));
}

std::vector<Tensor> foreach_tensor_norm_per_tensor_cuda(TensorList tensors, const Scalar &ord) {
  const auto p{ord_to_p(ord)};
  check_foreach_api_restrictions(tensors);
  if (should_go_slowpath(tensors, p)) {
    return foreach_tensor_norm_per_tensor_slow(tensors, ord);
  }
  const size_t ntensors = tensors.size();
  const Tensor norm_per_tensor = std::get<0>(foreach_tensor_norm_cuda_impl(tensors, p, /*per_tensor*/true));
  std::vector<Tensor> result;
  result.reserve(ntensors);
  for (const auto &i : c10::irange(ntensors)) {
    result.emplace_back(norm_per_tensor[i]);
  }
  return result;
}

Tensor foreach_tensor_norm_cuda(TensorList tensors, const Scalar &ord) {
  const auto p{ord_to_p(ord)};
  check_foreach_api_restrictions(tensors);
  if (should_go_slowpath(tensors, p)) {
    return foreach_tensor_norm_slow(tensors, ord);
  }
  return std::get<1>(foreach_tensor_norm_cuda_impl(tensors, p, /*per_tensor*/true));
}

} // namespace native
} // namespace at
