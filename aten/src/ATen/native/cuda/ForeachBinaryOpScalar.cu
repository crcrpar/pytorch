#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/BinaryOps.h>
namespace at { namespace native {

template<template<class> class Op>
std::vector<Tensor> foreach_binary_op(TensorList tensors, const Scalar& scalar) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors[0].scalar_type(), "foreach_binary_op_scalar_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<2>(tensor_lists,
                              BinaryOpScalarFunctor<scalar_t,
                                                    /* depth */ 2,
                                                    /* r_args_depth */ 1,
                                                    /* res_arg_index */ 1>(),
                              Op<opmath_t>(),
                              scalar.to<opmath_t>());
    });
    return tensor_lists[1];
}

template<template<class> class Op>
std::vector<Tensor> foreach_binary_op(std::vector<std::vector<Tensor>>& mta_input_tensors, const Scalar& scalar) {
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(mta_input_tensors[0].size());
    for (const auto& t: mta_input_tensors[0]) {
        vec_res.emplace_back(at::native::empty_like(t));
    }
    mta_input_tensors.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kBool, kBFloat16, kHalf,
        mta_input_tensors[0][0].scalar_type(), "foreach_binary_op_scalar_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<2>(mta_input_tensors,
                              BinaryOpScalarFunctor<scalar_t,
                                                    /* depth */ 2,
                                                    /* r_args_depth */ 1,
                                                    /* res_arg_index */ 1>(),
                              Op<opmath_t>(),
                              scalar.to<opmath_t>());
    });
    return std::move(mta_input_tensors[1]);
}

template<template<class> class Op>
void foreach_binary_op_(TensorList tensors, const Scalar& scalar) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors[0].scalar_type(), "foreach_binary_op_scalar_cuda_", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<1>(tensor_lists,
                              BinaryOpScalarFunctor<scalar_t,
                                                    /* depth */ 1,
                                                    /* r_args_depth */ 1,
                                                    /* res_arg_index */ 0>(),
                                                    Op<opmath_t>(),
                              scalar.to<opmath_t>());
    });
}

template<template<class> class Op>
void foreach_binary_op_(std::vector<std::vector<Tensor>>& mta_input_tensors, const Scalar& scalar) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kBool, kBFloat16, kHalf,
        mta_input_tensors[0][0].scalar_type(), "foreach_binary_op_scalar_cuda_", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<1>(mta_input_tensors,
                              BinaryOpScalarFunctor<scalar_t,
                                                    /* depth */ 1,
                                                    /* r_args_depth */ 1,
                                                    /* res_arg_index */ 0>(),
                                                    Op<opmath_t>(),
                              scalar.to<opmath_t>());
    });
}

#define FOREACH_BINARY_OP_SCALAR(NAME, OP, DIVISION_OP)                                             \
void foreach_tensor_##NAME##_scalar_kernel_cuda_(TensorList tensors, const Scalar& scalar) {               \
    check_foreach_api_restrictions(tensors);                                                        \
    OptionalDeviceGuard g;                                                                          \
    if (can_use_fast_route(tensors, scalar, DIVISION_OP)) {                                         \
        g.reset_device(tensors[0].device());                                                        \
        return foreach_binary_op_<OP>(tensors, scalar);                                             \
    }                                                                                               \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors)) {               \
        const bool is_cpu = (dev_dtypemtainputs.first == "cpu");                                    \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                       \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                  \
            auto & mtainput = dtype_mtainputs.second;                                               \
            if (is_cpu || !mtainput.is_mta_compatible(scalar, DIVISION_OP)) {                       \
                at::native::foreach_tensor_##NAME##_scalar_kernel_slow_(mtainput.get_tensorList(0), scalar);  \
            } else {                                                                                \
                foreach_binary_op_<OP>(mtainput.mta_input_tensors, scalar);                         \
            }                                                                                       \
        }                                                                                           \
    }                                                                                               \
}                                                                                                   \
                                                                                                    \
std::vector<Tensor> foreach_tensor_##NAME##_scalar_kernel_cuda(TensorList tensors, const Scalar& scalar) { \
    check_foreach_api_restrictions(tensors);                                                        \
    OptionalDeviceGuard g;                                                                          \
    if (can_use_fast_route(tensors, scalar, DIVISION_OP)) {                                         \
        g.reset_device(tensors[0].device());                                                        \
        return foreach_binary_op<OP>(tensors, scalar);                                              \
    }                                                                                               \
                                                                                                    \
    std::vector<at::Tensor> result;                                                                 \
    result.reserve(tensors.size());                                                                 \
    std::vector<size_t> index_map(tensors.size());                                                  \
                                                                                                    \
    size_t cur_index = 0;                                                                           \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors)) {               \
        const bool is_cpu = (dev_dtypemtainputs.first == "cpu");                                    \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                       \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                  \
            auto & mtainput = dtype_mtainputs.second;                                               \
            if (is_cpu || !mtainput.is_mta_compatible(scalar, DIVISION_OP)) {                       \
                for (const auto & t : at::native::foreach_tensor_##NAME##_scalar_kernel_slow(       \
                        mtainput.get_tensorList(0), scalar)) {                                      \
                    result.emplace_back(std::move(t));                                              \
                }                                                                                   \
            } else {                                                                                \
                for (const auto & t : foreach_binary_op<OP>(mtainput.mta_input_tensors, scalar)) {  \
                    result.emplace_back(std::move(t));                                              \
                }                                                                                   \
            }                                                                                       \
            const auto & indices = mtainput.indices;                                                \
            for (const auto & i : c10::irange(indices.size())) {                                    \
                index_map[indices[i]] = cur_index + i;                                              \
            }                                                                                       \
            cur_index += indices.size();                                                            \
        }                                                                                           \
    }                                                                                               \
    std::vector<at::Tensor> sorted_result;                                                          \
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
    OptionalDeviceGuard g;
    if (can_use_fast_route(tensors, scalar)) {
        g.reset_device(tensors[0].device());
        return foreach_binary_op_<std::minus>(tensors, scalar);
    }

    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors)) {
        const bool is_cpu = (dev_dtypemtainputs.first == "cpu");
        g.reset_device(at::Device(dev_dtypemtainputs.first));
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {
            auto & mtainput = dtype_mtainputs.second;
            if (is_cpu || !mtainput.is_mta_compatible(scalar)) {
                at::native::foreach_tensor_sub_scalar_kernel_slow_(mtainput.get_tensorList(0), scalar);
            } else {
                foreach_binary_op_<std::minus>(mtainput.mta_input_tensors, scalar);
            }
        }
    }
}

std::vector<Tensor> foreach_tensor_sub_scalar_kernel_cuda(TensorList tensors, const Scalar& scalar) {
    check_foreach_api_restrictions(tensors);
    for (const auto & t : tensors) {
        at::native::sub_check(t, scalar);
    }
    OptionalDeviceGuard g;
    if (can_use_fast_route(tensors, scalar)) {
        g.reset_device(tensors[0].device());
        return foreach_binary_op<std::minus>(tensors, scalar);
    }

    std::vector<at::Tensor> result;
    result.reserve(tensors.size());
    std::vector<size_t> index_map(tensors.size(), 0);

    size_t cur_index = 0;
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors)) {
        const bool is_cpu = (dev_dtypemtainputs.first == "cpu");
        g.reset_device(dev_dtypemtainputs.first);
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {
            auto & mtainput = dtype_mtainputs.second;
            if (is_cpu || !mtainput.is_mta_compatible(scalar, true)) {
                for (const auto & t : at::native::foreach_tensor_sub_scalar_kernel_slow(mtainput.get_tensorList(0), scalar)) {
                    result.emplace_back(std::move(t));
                }
            } else {
                for (const auto & t : foreach_binary_op<std::minus>(mtainput.mta_input_tensors, scalar)) {
                    result.emplace_back(std::move(t));
                }
            }
            const auto & indices = mtainput.indices;
            for (const auto & i : c10::irange(indices.size())){
                index_map[indices[i]] = cur_index + i;
            }
            cur_index += indices.size();
        }
    }
    std::vector<at::Tensor> sorted_result;
    sorted_result.reserve(tensors.size());
    for (const auto & i : index_map) {
        sorted_result.emplace_back(result[i]);
    }
    return sorted_result;
}

}} // namespace at::native
