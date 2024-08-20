#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/Lerp.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_lerp_native.h>

#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

template <typename T>
struct LerpFunctor {
  inline C10_DEVICE T operator()(const T self, const T end, const T weight) {
    return lerp(self, end, weight);
  }
};

std::vector<at::Tensor> foreach_tensor_lerp_ternary_cuda(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  if (!can_use_fast_route({tensors1, tensors2, tensors3}, {}, true)) {
    return foreach_tensor_ternary_lerp_slow(tensors1, tensors2, tensors3);
  }

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), tensors3.vec(), vec_res};

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_ternary_cuda",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<4>(
            tensor_lists,
            TernaryOpListFunctor<
                scalar_t,
                /* depth */ 4,
                /* r_args_depth */ 3,
                /* res_arg_index */ 3>(),
            LerpFunctor<opmath_t>());
      });

  return tensor_lists[3];
}

void foreach_tensor_lerp_ternary_cuda_(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  if (!can_use_fast_route({tensors1, tensors2, tensors3}, {}, true)) {
    return foreach_tensor_ternary_lerp_slow_(tensors1, tensors2, tensors3);
  }

  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), tensors3.vec()};
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_ternary_cuda_",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<3>(
            tensor_lists,
            TernaryOpListFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 3,
                /* res_arg_index */ 0>(),
            LerpFunctor<opmath_t>());
      });
  increment_version(tensors1);
}

std::vector<at::Tensor> foreach_tensor_lerp_list_cuda(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight) {
  check_foreach_api_restrictions(tensors1, tensors2);
  if (!can_use_fast_route({tensors1, tensors2}, {}, true)) {
    return foreach_tensor_lerp_list_kernel_slow(tensors1, tensors2, weight);
  }

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), vec_res};

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_scalar_cuda",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<3>(
            tensor_lists,
            TernaryOpScalarFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 2,
                /* res_arg_index */ 2>(),
            LerpFunctor<opmath_t>(),
            weight.to<opmath_t>());
      });

  return tensor_lists[2];
}

void foreach_tensor_lerp_list_cuda_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight) {
  check_foreach_api_restrictions(tensors1, tensors2);
  if (!can_use_fast_route({tensors1, tensors2}, {}, true)) {
    return foreach_tensor_lerp_list_kernel_slow_(tensors1, tensors2, weight);
  }

  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec()};
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_scalar_cuda_",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<2>(
            tensor_lists,
            TernaryOpScalarFunctor<
                scalar_t,
                /* depth */ 2,
                /* r_args_depth */ 2,
                /* res_arg_index */ 0>(),
            LerpFunctor<opmath_t>(),
            weight.to<opmath_t>());
      });
}

namespace { // anonymous namespace
template <typename T, int depth, int r_args_depth, int res_arg_index>
struct LerpScalarListFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename Op>
  __device__ __forceinline__ void operator()(
      int64_t chunk_size,
      TensorListScalarListMetadata<opmath_t, depth>& tl,
      Op op) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];

    T* args[depth];
    opmath_t scalar = tl.scalar_vals[tensor_loc];
    T r_args[r_args_depth][kILP];
    const bool all_aligned =
        init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc);
    const auto n = tl.numel_for_tensor[tensor_loc] - chunk_idx * chunk_size;

    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
        // load
#pragma unroll
        for (int ii = 0; ii < r_args_depth; ++ii) {
          load_store<T>(r_args[ii], args[ii], 0, i_start);
        }
        // math
#pragma unroll
        for (int ii = 0; ii < kILP; ++ii) {
          r_args[0][ii] = static_cast<T>(
              op(static_cast<opmath_t>(r_args[0][ii]),
                 static_cast<opmath_t>(r_args[1][ii]),
                 scalar));
        }
        // store
        load_store<T>(args[res_arg_index], r_args[0], i_start, 0);
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
        load_args<depth, T>(r_args, args, i_start, chunk_size, n);
        // math
#pragma unroll
        for (int ii = 0; ii < kILP; ++ii) {
          r_args[0][ii] = static_cast<T>(
              op(static_cast<opmath_t>(r_args[0][ii]),
                 static_cast<opmath_t>(r_args[1][ii]),
                 scalar));
        }
        store_args<T>(args[res_arg_index], r_args[0], i_start, chunk_size, n);
      }
    }
  }
};
} // namespace

std::vector<at::Tensor> foreach_tensor_ternary_lerp_scalarlist_cuda(
    TensorList tensors1,
    TensorList tensors2,
    const ArrayRef<Scalar>& weight) {
  check_foreach_api_restrictions(tensors1, tensors2);
  if (!can_use_fast_route({tensors1, tensors2}, weight, true)) {
    return foreach_tensor_ternary_lerp_scalarlist_slow(
        tensors1, tensors2, weight);
  }

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), vec_res};

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_scalar_cuda",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<3, scalar_t>(
            tensor_lists,
            weight,
            LerpScalarListFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 2,
                /* res_arg_index */ 2>(),
            LerpFunctor<opmath_t>());
      });

  return tensor_lists[2];
}

std::vector<at::Tensor> foreach_tensor_ternary_lerp_scalarlist_cuda_(
    TensorList tensors1,
    TensorList tensors2,
    const ArrayRef<Scalar>& weight) {
  check_foreach_api_restrictions(tensors1, tensors2);
  if (!can_use_fast_route({tensors1, tensors2}, weight, true)) {
    return foreach_tensor_ternary_lerp_scalarlist_slow(
        tensors1, tensors2, weight);
  }

  std::vector<at::Tensor> vec_res;
  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec()};

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_scalar_cuda",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<2, scalar_t>(
            tensor_lists,
            weight,
            LerpScalarListFunctor<
                scalar_t,
                /* depth */ 2,
                /* r_args_depth */ 2,
                /* res_arg_index */ 0>(),
            LerpFunctor<opmath_t>());
      });

  return tensor_lists[2];
}
} // namespace at::native
