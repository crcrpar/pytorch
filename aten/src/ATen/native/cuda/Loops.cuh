
#pragma once

#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>

#include <thrust/tuple.h>

namespace at { namespace native {

#define NUM_THREADS (C10_WARP_SIZE * 2)
#define THREAD_WORK_SIZE 4
#define BLOCK_WORK_SIZE (THREAD_WORK_SIZE * num_threads)

constexpr int num_threads = NUM_THREADS;
constexpr int thread_work_size = THREAD_WORK_SIZE;
constexpr int block_work_size = BLOCK_WORK_SIZE;

// `needs_dynamic_casting` compares the types expected by iterator
// (i.e. dtypes of the operands) with the actual type of the arguments
// of func_t
template<typename func_t, int nargs=function_traits<func_t>::arity>
struct needs_dynamic_casting {
  static bool check(TensorIterator& iter) {
    using traits = function_traits<func_t>;
    if (iter.dtype(nargs) != c10::impl::CPPTypeToScalarType<typename traits::template arg<nargs - 1>::type>::value) {
      return true;
    }
    return needs_dynamic_casting<func_t, nargs - 1>::check(iter);
  }
};

template<typename func_t>
struct needs_dynamic_casting<func_t, 0> {
  static bool check(TensorIterator& iter) {
    using traits = function_traits<func_t>;
    return iter.dtype(0) != c10::impl::CPPTypeToScalarType<typename traits::result_type>::value;
  }
};

template<int N>
static OffsetCalculator<N> make_input_offset_calculator(const TensorIterator& iter) {
  // array size can not be 0, this happens when N == 0
  constexpr int array_size = std::max<int>(N, 1);
  TORCH_INTERNAL_ASSERT(N == iter.ntensors() - iter.noutputs());
  std::array<const int64_t*, array_size> strides;
  int64_t element_sizes[array_size];
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i + iter.noutputs()).data();
    element_sizes[i] = iter.element_size(i + iter.noutputs());
  }
  return OffsetCalculator<N>(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <int num_outputs = 1>
static OffsetCalculator<num_outputs> make_output_offset_calculator(const TensorIterator& iter) {
  std::array<const int64_t*, num_outputs> strides;
  int64_t element_sizes[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    strides[i] = iter.strides(i).data();
    element_sizes[i] = iter.element_size(i);
  }
  return OffsetCalculator<num_outputs>(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

}}  // namespace at::native

// Note:
// CUDA and ROCm get diverged in this PR:
//   https://github.com/pytorch/pytorch/pull/32383
// Because for some reason trying to enable vectorized
// memory access introduce regression on ROCm.

#ifndef __HIP_PLATFORM_HCC__
#include <ATen/native/cuda/CUDALoops.cuh>
#else
#include <ATen/native/cuda/ROCmLoops.cuh>
#endif

namespace at { namespace native {

template <typename func_t>
void gpu_kernel(TensorIterator& iter, const func_t& f) {
  ASSERT_HOST_DEVICE_LAMBDA(func_t);

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(iter.device(arg).is_cuda());
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel(sub_iter, f);
    }
    return;
  }

  gpu_kernel_impl(iter, f);
}

template <typename func_t>
void gpu_kernel_with_scalars(TensorIterator& iter, const func_t& f) {
  ASSERT_HOST_DEVICE_LAMBDA(func_t);
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");

  if (iter.is_cpu_scalar(1)) {
    using arg1_t = typename traits::template arg<0>::type;
    using arg2_t = typename traits::template arg<1>::type;
    auto a = iter.scalar_value<arg1_t>(1);
    iter.remove_operand(1);
    gpu_kernel(iter, [=]GPU_LAMBDA(arg2_t b) {
      return f(a, b);
    });
  } else if (iter.is_cpu_scalar(2)) {
    using arg1_t = typename traits::template arg<0>::type;
    using arg2_t = typename traits::template arg<1>::type;
    auto b = iter.scalar_value<arg2_t>(2);
    iter.remove_operand(2);
    gpu_kernel(iter, [=]GPU_LAMBDA(arg1_t a) {
      return f(a, b);
    });
  } else {
    gpu_kernel(iter, f);
  }
}

namespace { // functions for `gpu_kernel_multiple_outputs`.

// check the return type is `thrust::tuple`, not `std::tuple`.
template <typename T> struct is_tuple: std::false_type {};

template <typename ...T> struct is_tuple<thrust::tuple<T...>>: std::true_type {};

template <int num_outputs, typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t>
C10_LAUNCH_BOUNDS_1(num_threads)
__global__ void unrolled_elementwise_kernel_for_multi_outputs(int N, func_t f, array_t data, inp_calc_t ic, out_calc_t oc) {
  int remaining = N - block_work_size * blockIdx.x;
  modern::elementwise_kernel_helper(f, memory::policies::multi_outputs_unroll<array_t, inp_calc_t, out_calc_t, num_outputs>(data, remaining, ic, oc));
}

template <int num_outputs, typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t>
static inline void launch_unrolled_kernel_for_multi_outputs(int64_t N, const func_t& f, array_t data, inp_calc_t ic, out_calc_t oc) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size - 1) / block_work_size;
  auto stream = at::cuda::getCurrentCUDAStream();
  unrolled_elementwise_kernel_for_multi_outputs<num_outputs, func_t, array_t><<<grid, num_threads, 0, stream>>>(N, f, data, ic, oc);
  AT_CUDA_CHECK(cudaGetLastError());
}

template <typename func_t>
void gpu_kernel_multiple_outputs_impl(TensorIterator& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using output_t = typename traits::result_type;
  static_assert(memory::detail::is_tuple<output_t>::value);
  constexpr int num_outputs = thrust::tuple_size<output_t>::value;
  constexpr int num_inputs = traits::arity;
  constexpr int ntensors = num_outputs + num_inputs;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ntensors() == ntensors);

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  if (iter.is_contiguous()) {
    auto input_calc = TrivialOffsetCalculator<num_inputs>();
    auto output_calc = TrivialOffsetCalculator<num_outputs>();
    launch_unrolled_kernel_for_multi_outputs<num_outputs>(numel, f, data, input_calc, output_calc);
  } else {
    auto input_calc = make_input_offset_calculator<num_inputs>(iter);
    auto output_calc = make_output_offset_calculator<num_outputs>(iter);
    launch_unrolled_kernel_for_multi_outputs<num_outputs>(numel, f, data, input_calc, output_calc);
  }
}
} // namespace

template <typename func_t>
void gpu_kernel_multiple_outputs(TensorIterator& iter, const func_t& f) {
  ASSERT_HOST_DEVICE_LAMBDA(func_t);

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(iter.device(arg).is_cuda());
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel_multiple_outputs(sub_iter, f);
    }
    return;
  }

  gpu_kernel_multiple_outputs_impl(iter, f);
}

}} //namespace at::native
