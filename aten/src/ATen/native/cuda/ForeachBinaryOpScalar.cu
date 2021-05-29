#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/BinaryOps.h>
namespace at { namespace native {

template<template<class> class Op>
std::vector<Tensor> foreach_binary_op(TensorList tensors, const Scalar& scalar) {
    std::vector<std::vector<Tensor>> tensor_lists{tensors.vec()};
    std::vector<Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto & t : tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kBool, kBFloat16, kHalf,
        tensors[0].scalar_type(),
        "foreach_binary_op_scalar_cuda",
        [&]() {
            using opmath_t = get_opmath_t<scalar_t>::opmath_t;
            multi_tensor_apply<2>(
                tensor_lists,
                BinaryOpScalarFunctor<scalar_t, /* depth */ 2, /* r_args_depth */ 1, /* res_arg_index */ 1>(),
                Op<opmath_t>(),
                scalar.to<scalar_t>());
        });
    return tensor_lists[1];
}

template<template<class> class Op>
std::vector<Tensor> foreach_binary_op(std::vector<std::vector<Tensor>>& mta_input_tensors, const Scalar& scalar) {
    std::vector<Tensor> vec_res;
    vec_res.reserve(mta_input_tensors[0].size());
    for (const auto & t : mta_input_tensors[0]) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    mta_input_tensors.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kBool, kBFloat16, kHalf,
        mta_input_tensors[0][0].scalar_type(),
        "foreach_binary_op_scalar_cuda",
        [&]() {
            using opmath_t = get_opmath_t<scalar_t>::opmath_t;
            multi_tensor_apply<2>(
                mta_input_tensors,
                BinaryOpScalarFunctor<scalar_t, /* depth */ 2, /* r_args_depth */ 1, /* res_arg_index */ 1>(),
                Op<opmath_t>(),
                scalar.to<scalar_t>());
        });
    return std::move(mta_input_tensors[1]);
}

template<template<class> class Op>
void foreach_binary_op_(TensorList tensors, const Scalar& scalar) {
    std::vector<std::vector<Tensor>> tensor_lists{tensors.vec()};
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kBool, kBFloat16, kHalf,
        tensor_lists[0].scalar_type(),
        "foreach_binary_op_scalar_cuda",
        [&]() {
            using opmath_t = get_opmath_t<scalar_t>::opmath_t;
            multi_tensor_apply<1>(
                tensor_lists,
                BinaryOpScalarFunctor<scalar_t, /* depth */ 1, /* r_args_depth */ 1, /* res_arg_index */ 0>(),
                Op<opmath_t>(),
                scalar.to<scalar_t>());
        });
}

template<template<class> class Op>
void foreach_binary_op_(std::vector<std::vector<Tensor>>& mta_input_tensors, const Scalar& scalar) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kBool, kBFloat16, kHalf,
        mta_input_tensors[0][0].scalar_type(),
        "foreach_binary_op_scalar_cuda",
        [&]() {
            using opmath_t = get_opmath_t<scalar_t>::opmath_t;
            multi_tensor_apply<1>(
                mta_input_tensors,
                BinaryOpScalarFunctor<scalar_t, /* depth */ 1, /* r_args_depth */ 1, /* res_arg_index */ 0>(),
                Op<opmath_t>(),
                scalar.to<scalar_t>());
        });
}

#define FOREACH_BINARY_OP_SCALAR(NAME, OP, DIVISION_OP)                                             \
void foreach_tensor_##NAME##_scalar_kernel_cuda_(                                                   \
        TensorList tensors,                                                                         \
        const Scalar& scalar) {                                                                     \
    check_foreach_api_restrictions(tensors);                                                        \
    if (can_use_fast_route(tensors)) {                                                              \
        foreach_binary_op_<OP>(tensors, scalar);                                                    \
        return;                                                                                     \
    }                                                                                               \
    OptionalDeviceGuard g;                                                                          \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors)) {               \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                       \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                  \
            auto & mta_input_tensors = dtype_mtainputs.second.mta_input_tensors;                    \
            if (!can_use_fast_route(mta_input_tensors, scalar, DIVISION_OP)) {                      \
                at::native::foreach_tensor_##NAME##_scalar_kernel_slow_(                            \
                    TensorList{mta_input_tensors[0]}, scalar);                                      \
            } else {                                                                                \
                foreach_binary_op_<OP>(mta_input_tensors, scalar);                                  \
            }                                                                                       \
        }                                                                                           \
    }                                                                                               \
}                                                                                                   \
                                                                                                    \
std::vector<Tensor> foreach_tensor_##NAME_scalar_kernel_cuda(                                       \
        TensorList tensors,                                                                         \
        const Scalar& scalar) {                                                                     \
    check_foreach_api_restrictions(tensors);                                                        \
    if (can_use_fast_route(tensors)) {                                                              \
        return foreach_binary_op<OP>(tensors, scalar);                                              \
    }                                                                                               \
    std::vector<Tensor> result;                                                                     \
    result.reserve(tensors.size());                                                                 \
    std::vector<size_t> index_map(tensors.size(), 0);                                               \
                                                                                                    \
    size_t cur_index = 0;                                                                           \
    OptionalDeviceGuard g;                                                                          \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors)) {               \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                       \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                  \
            auto & mta_input_tensors =  dtype_mtainputs.second.mta_input_tensors;                   \
            if (!can_use_fast_route(mta_input_tensors, scalar, DIVISION_OP)) {                      \
                for (const auto & t : at::native::foreach_tensor_##NAME##_scalar_kernel_slow(       \
                        TensorList{mta_input_tensors[0]}, scalar)) {                                \
                    result.emplace_back(std::move(t));                                              \
                }                                                                                   \
            } else {                                                                                \
                for (const auto & t : foreach_binary_op_<OP>(mta_input_tensors, scalar)) {          \
                    result.emplace_back(std::move(t));                                              \
                }                                                                                   \
            }                                                                                       \
            const auto & indices = dtype_mtainputs.second.indices;                                  \
            for (const auto & i : c10::irange(indices.size())) {                                    \
                index_map[indices[i]] = cur_index + i;                                              \
            }                                                                                       \
            cur_index += indices.size();                                                            \
        }                                                                                           \
    }                                                                                               \
    std::vector<Tensor> sorted_result;                                                              \
    sorted_result.reserve(tensors.size());                                                          \
    for (const auto & i : index_map) {                                                              \
        sorted_result.emplace_back(std::move(result[i]));                                           \
    }                                                                                               \
    return sorted_result;                                                                           \
}

FOREACH_BINARY_OP_SCALAR(add, std::plus, false);
FOREACH_BINARY_OP_SCALAR(mul, std::multiplies, false);

// In the case of division, integer inputs will result in float.
// Currently multi tensor apply can only return result of the same type as input.
FOREACH_BINARY_OP_SCALAR(div, std::divides, true);

// In the case of subtraction, we dont allow scalar to be boolean following the torch.sub logic
void foreach_tensor_sub_scalar_kernel_cuda_(TensorList tensors, const Scalar& scalar) {
    check_foreach_api_restrictions(tensors);
    for (const auto & t : tensors) {
        at::native::sub_check(t, scalar);
    }
    if (can_use_fast_route(tensors)) {
        foreach_binary_op_<std::minus>(tensors, scalar);
        return;
    }

    OptionalDeviceGuard g;
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors)) {
        g.reset_device(at::Device(dev_dtypemtainputs.first));
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {
            auto & mta_input_tensors = dtype_mtainputs.second.mta_input_tensors;
            if (!can_use_fast_route(mta_input_tensors, scalar)) {
                at::native::foreach_tensor_sub_scalar_kernel_slow_(TensorList{mta_input_tensors[0]}, scalar);
            } else {
                foreach_binary_op_<std::minus>(mta_input_tensors, scalar);
            }
        }
    }
}

std::vector<Tensor> foreach_tensor_sub_scalar_kernel_cuda(TensorList tensors, const Scalar& scalar) {
    check_foreach_api_restrictions(tensors);
    for (const auto & t : tensors) {
        at::native::sub_check(t, scalar);
    }
    if (can_use_fast_route(tensors)) {
        return foreach_binary_op<std::minus>(tensors, scalar);
    }

    std::vector<Tensor> result;
    result.reserve(tensors.size());
    std::vector<size_t> index_map(tensors.size(), 0);

    size_t cur_index = 0;
    OptionalDeviceGuard g;
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors)) {
        g.reset_device(at::Device(dev_dtypemtainputs.first));
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {
            auto & mta_input_tensors = dtype_mtainputs.second.mta_input_tensors;
            if (!can_use_fast_route(mta_input_tensors, scalar)) {
                for (const auto & t : at::native::foreach_tensor_sub_scalar_kernel_slow(TensorList{mta_input_tensors[0]}, scalar)) {
                    result.emplace_back(std::move(t));
                }
            } else {
                for (const auto & t : foreach_binary_op<std::minus>(mta_input_tensors, scalar)) {
                    result.emplace_back(std::move(t));
                }
            }
            const auto & indices = dtype_mtainputs.second.indices;
            for (const auto & i : c10::irange(indices.size())) {
                index_map[indices[i]] = cur_index + i;
            }
            cur_index += indices.size();
        }
    }
    std::vector<Tensor> sorted_result;
    sorted_result.reserve(tensors.size());
    for (const auto & i : index_map) {
        sorted_result.emplace_back(std::move(result[i]));
    }
    return sorted_result;
}

}} // namespace at::native
