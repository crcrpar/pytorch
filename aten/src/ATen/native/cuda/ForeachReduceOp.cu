#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>

namespace at {
namespace native {

enum class CalcNormType { PerTensor, GlobalNorm };
#define check_calc_norm_type(X) static_assert((X) == 1 || (X) == 2, "foreach_norm supports only L1 and L2")

template<CalcNormType cnt> struct RetType { using value = Tensor; };
template<> struct RetType<CalcNormType::PerTensor> { using value = std::vector<Tensor>; };

template<CalcNormType cnt> using ret_type = typename RetType<cnt>::value;


template<typename T, int NormType>
struct NormElemCalcFunctor {
  check_calc_norm_type(NormType);
  using opmath_t = at::opmath_type<T>;
  __device__ __forceinline__ opmath_t operator()(const T x) { return ::abs(x); };
};

template<typename T>
struct NormElemCalcFunctor<T, 2> {
  using opmath_t = at::opmath_type<T>;
  __device__ __forceinline__ opmath_t operator()(const T x) { return x * x; }
};

template<CalcNormType calc_norm_type, typename T, int NormType, int depth=1, int r_args_depth=1, int res_arg_index=0>
struct LpNormFunctor {
  check_calc_norm_type(NormType);
  using opmath_t = typename at::opmath_type<T>;
  __device__ __forceinline__ void operator() (
      int chunk_size,
      TensorListMetadata<depth>& tl,
      opmath_t* output,
      opmath_t* output_per_tensor,
      const int max_chunks_per_tensor
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
          opmath_t next = static_cast<opmath_t>(r_x[ii]);
          vals[ii] += NormElemCalcFunctor<opmath_t, NormType>()(next);
        }
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * kILP) {
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            opmath_t next = static_cast<opmath_t>(x[i]);
            vals[ii] += NormElemCalcFunctor<opmath_t, NormType>()(next);
          }
        }
      }
    }

    auto val = opmath_t(0);
    for (int i = 0; i < kILP; i++) {
      val += vals[i];
    }
    const auto final = at::native::cuda_utils::BlockReduceSum<opmath_t>(val, s_vals);

    if (threadIdx.x == 0) {
      if (calc_norm_type == CalcNormType::PerTensor) {
        output_per_tensor[(tl.start_tensor_this_launch + tensor_loc) * max_chunks_per_tensor + chunk_idx] = final;
      }
      if (calc_norm_type == CalcNormType::GlobalNorm) {
        output[blockIdx.x] += final;
      }
    }
  }
};

template<typename T, int NormType>
struct NormCleanupFunctor {
  check_calc_norm_type(NormType);
  using opmath_t = at::opmath_type<T>;
  __device__ __forceinline__ T operator()(const opmath_t x) { return static_cast<T>(x); };
};

template<typename T>
struct NormCleanupFunctor<T, 2> {
  using opmath_t = at::opmath_type<T>;
  __device__ __forceinline__ T operator()(const opmath_t x) { return static_cast<T>(::sqrt(x)); };
};

template<CalcNormType calc_norm_type, typename T, int NormType, typename opmath_t>
__global__ void lpnorm_cleanup(
    opmath_t *output,
    T *ret,
    opmath_t* output_per_tensor,
    T* ret_per_tensor,
    int max_chunks_per_tensor) {
  __shared__ opmath_t vals[512];
  if (calc_norm_type == CalcNormType::GlobalNorm) {
    if (blockIdx.x == 0) {
      opmath_t val = 0;
      if (threadIdx.x < 320) {
        val = output[threadIdx.x];
      }
      opmath_t final = at::native::cuda_utils::BlockReduceSum<opmath_t>(val, vals);
      if (threadIdx.x == 0) {
        *ret = NormCleanupFunctor<T, NormType>()(final);
      }
    }
  }
  if (calc_norm_type == CalcNormType::PerTensor) {
    opmath_t val = 0;
    opmath_t *output_this_tensor = output_per_tensor + blockIdx.x * max_chunks_per_tensor;
    for (int i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x) {
      val += output_this_tensor[i];
    }
    opmath_t final = at::native::cuda_utils::BlockReduceSum<opmath_t>(val, vals);
    if(threadIdx.x == 0) {
      ret_per_tensor[blockIdx.x] = NormCleanupFunctor<T, NormType>()(final);
    }
  }
}

template<CalcNormType calc_norm_type> ret_type<calc_norm_type> foreach_norm_slow_path(TensorList tensors, const Scalar &ord) {
  return foreach_tensor_norm_slow(tensors, ord);
}

template<> ret_type<CalcNormType::PerTensor>
foreach_norm_slow_path<CalcNormType::PerTensor>(TensorList tensors, const Scalar &ord) {
  return foreach_tensor_norm_per_tensor_slow(tensors, ord);
}

template<CalcNormType calc_norm_type> ret_type<calc_norm_type>
choose_return_value(const Tensor &ret, const Tensor &ret_per_tensor, const int ntensors) {
  return ret;
}

template<> ret_type<CalcNormType::PerTensor>
choose_return_value<CalcNormType::PerTensor>(const Tensor &ret, const Tensor &ret_per_tensor, const int ntensors) {
  std::vector<Tensor> result;
  result.reserve(ntensors);
  for (const auto &i : c10::irange(ntensors)) {
    result.emplace_back(ret_per_tensor[i]);
  }
  return result;
}

#undef check_calc_norm_type

template<CalcNormType calc_norm_type>
void init_tensors(
    const int ntensors,
    const int max_chunks_per_tensor,
    const TensorOptions options,
    const ScalarType scalar_type,
    Tensor &output,
    Tensor &ret,
    Tensor &output_per_tensor,
    Tensor &ret_per_tensor) {
  const auto opmath_type_options = options.dtype(toOpMathType(scalar_type));
  output = at::zeros({320}, opmath_type_options);
  ret = at::empty({}, options);
}

template<>
void init_tensors<CalcNormType::PerTensor>(
    const int ntensors,
    const int max_chunks_per_tensor,
    const TensorOptions options,
    const ScalarType scalar_type,
    Tensor &output,
    Tensor &ret,
    Tensor &output_per_tensor,
    Tensor &ret_per_tensor) {
  const auto opmath_type_options = options.dtype(toOpMathType(scalar_type));
  output_per_tensor = at::zeros({ntensors*max_chunks_per_tensor}, opmath_type_options);
  ret_per_tensor = at::empty({ntensors}, options);
}

// note(mkozuki): Why excluding Int and Complex from fast path
// - Int: at::norm does not support.
// - Complex: __shfl_down_sync does not support complex and foreach does not support functions whose inputs dtypes and output dtype are different.
template<CalcNormType calc_norm_type>
ret_type<calc_norm_type> foreach_tensor_norm_cuda_impl(TensorList tensors, const Scalar& ord) {
  double p;
  if (ord.isIntegral(false)) {
    p = ord.to<int64_t>();
  } else if (ord.isFloatingPoint()) {
    p = ord.to<double>();
  } else {
    AT_ERROR("foreach_tensor_norm_cuda expects ord to be integer or float");
  }
  check_foreach_api_restrictions(tensors);
  const bool has_int_or_complex = std::any_of(tensors.begin(), tensors.end(), [](const auto & t) {
      const auto scalar_type = t.scalar_type();
      return at::isIntegralType(scalar_type, /*includeBool*/true) || at::isComplexType(scalar_type);
  });
  if (!can_use_fast_route(tensors) ||
      has_int_or_complex ||
      !(p == static_cast<double>(1) || p == static_cast<double>(2))) {
    return foreach_norm_slow_path<calc_norm_type>(tensors, ord);
  }

  const int ntensors = tensors.size();
  int max_chunks_per_tensor = -1;

  for (int t = 0; t < ntensors; t++) {
    int max_chunks_this_tensor = (tensors[0][t].numel() + kChunkSize - 1) / kChunkSize;
    if(max_chunks_this_tensor > max_chunks_per_tensor) {
      max_chunks_per_tensor = max_chunks_this_tensor;
    }
  }
  Tensor output, ret, output_per_tensor, ret_per_tensor;
  init_tensors<calc_norm_type>(ntensors, max_chunks_per_tensor, tensors[0].options(), tensors[0].scalar_type(), output, ret, output_per_tensor, ret_per_tensor);

  auto tensor_lists = std::vector<std::vector<Tensor>>{tensors.vec()};
  if (p == static_cast<double>(1)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, tensor_lists[0][0].scalar_type(), "foreach_tensor_norm_cuda", [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<1>(
          tensor_lists,
          LpNormFunctor<calc_norm_type, scalar_t, 1>(),
          calc_norm_type == CalcNormType::GlobalNorm ? output.data_ptr<opmath_t>() : nullptr,
          calc_norm_type == CalcNormType::PerTensor ? output_per_tensor.data_ptr<opmath_t>() : nullptr,
          max_chunks_per_tensor);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const at::cuda::OptionalCUDAGuard device_guard(device_of(output_per_tensor));
        auto stream = at::cuda::getCurrentCUDAStream();
        lpnorm_cleanup<calc_norm_type, scalar_t, 1><<<ntensors, 512, 0, stream>>>(
          calc_norm_type == CalcNormType::GlobalNorm ? output.data_ptr<opmath_t>() : nullptr,
          calc_norm_type == CalcNormType::GlobalNorm ? ret.data_ptr<scalar_t>() : nullptr,
          calc_norm_type == CalcNormType::PerTensor ? output_per_tensor.data_ptr<opmath_t>() : nullptr,
          calc_norm_type == CalcNormType::PerTensor ? ret_per_tensor.data_ptr<scalar_t>() : nullptr,
          max_chunks_per_tensor);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  } else if (p == static_cast<double>(2)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, tensor_lists[0][0].scalar_type(), "foreach_tensor_norm_cuda", [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<1>(
          tensor_lists,
          LpNormFunctor<calc_norm_type, scalar_t, 2>(),
          calc_norm_type == CalcNormType::GlobalNorm ? output.data_ptr<opmath_t>() : nullptr,
          calc_norm_type == CalcNormType::PerTensor ? output_per_tensor.data_ptr<opmath_t>() : nullptr,
          max_chunks_per_tensor);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const at::cuda::OptionalCUDAGuard device_guard(device_of(output_per_tensor));
        auto stream = at::cuda::getCurrentCUDAStream();
        lpnorm_cleanup<calc_norm_type, scalar_t, 2><<<ntensors, 512, 0, stream>>>(
          calc_norm_type == CalcNormType::GlobalNorm ? output.data_ptr<opmath_t>() : nullptr,
          calc_norm_type == CalcNormType::GlobalNorm ? ret.data_ptr<scalar_t>() : nullptr,
          calc_norm_type == CalcNormType::PerTensor ? output_per_tensor.data_ptr<opmath_t>() : nullptr,
          calc_norm_type == CalcNormType::PerTensor ? ret_per_tensor.data_ptr<scalar_t>() : nullptr,
          max_chunks_per_tensor);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  } else {
    AT_ERROR("foreach_tensor_norm_cuda fast path got unexpected ord value: ", p);
  }

  return choose_return_value<calc_norm_type>(ret, ret_per_tensor, ntensors);
}

ret_type<CalcNormType::PerTensor> foreach_tensor_norm_per_tensor_cuda(TensorList tensors, const Scalar &ord) {
  return foreach_tensor_norm_cuda_impl<CalcNormType::PerTensor>(tensors, ord);
}

ret_type<CalcNormType::GlobalNorm> foreach_tensor_norm_cuda(TensorList tensors, const Scalar &ord) {
  return foreach_tensor_norm_cuda_impl<CalcNormType::GlobalNorm>(tensors, ord);
}

} // namespace native
} // namespace at
