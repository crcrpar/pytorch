#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/BinaryOps.h>

namespace at { namespace native {

template<template<class> class Op>
std::vector<Tensor> foreach_binary_op(TensorList tensors, at::ArrayRef<Scalar> scalars) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto& t: tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors.vec());
    tensor_lists.emplace_back(vec_res);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBFloat16, kHalf, kBool, tensors[0].scalar_type(), "foreach_binary_op_scalarlist_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<2, opmath_t>(tensor_lists,
                                        scalars,
                                        BinaryOpScalarListFunctor<scalar_t,
                                                                  /* depth */ 2,
                                                                  /* r_args_depth */ 1,
                                                                  /* res_arg_index */ 1>(),

                                        Op<opmath_t>());
    });
    return tensor_lists[1];
}

template<template<class> class Op>
std::vector<Tensor> foreach_binary_op(std::vector<std::vector<Tensor>>& mta_input_tensors, at::ArrayRef<Scalar> scalars) {
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(mta_input_tensors[0].size());
    for (const auto& t: mta_input_tensors[0]) {
        vec_res.emplace_back(at::native::empty_like(t));
    }
    mta_input_tensors.emplace_back(vec_res);

    AT_DISPATCH_ALL_TYPES_AND3(
        kBFloat16, kHalf, kBool,
        mta_input_tensors[0][0].scalar_type(), "foreach_binary_op_scalarlist_cuda",
        [&]() {
            using opmath_t = get_opmath_t<scalar_t>::opmath_t;
            multi_tensor_apply<2, opmath_t>(mta_input_tensors,
                                            scalars,
                                            BinaryOpScalarListFunctor<scalar_t,
                                                                    /* depth */ 2,
                                                                    /* r_args_depth */ 1,
                                                                    /* res_arg_index */ 1>(),

                                            Op<opmath_t>());
    });
    return std::move(mta_input_tensors[1]);
}

template<template<class> class Op>
void foreach_binary_op_(TensorList tensors, at::ArrayRef<Scalar> scalars) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors.vec());

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBFloat16, kHalf, kBool, tensors[0].scalar_type(), "foreach_binary_op_scalarlist_cuda_", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<1, opmath_t>(tensor_lists,
                                        scalars,
                                        BinaryOpScalarListFunctor<scalar_t,
                                                                  /* depth */ 1,
                                                                  /* r_args_depth */ 1,
                                                                  /* res_arg_index */ 0>(),
                                        Op<opmath_t>());
    });
}

template<template<class> class Op>
void foreach_binary_op_(std::vector<std::vector<Tensor>>& mta_input_tensors, at::ArrayRef<Scalar> scalars) {
    AT_DISPATCH_ALL_TYPES_AND3(
        kBFloat16, kHalf, kBool,
        mta_input_tensors[0][0].scalar_type(), "foreach_binary_op_scalarlist_cuda_", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<1, opmath_t>(mta_input_tensors,
                                        scalars,
                                        BinaryOpScalarListFunctor<scalar_t,
                                                                  /* depth */ 1,
                                                                  /* r_args_depth */ 1,
                                                                  /* res_arg_index */ 0>(),
                                        Op<opmath_t>());
    });
}

#define FOREACH_BINARY_OP_SCALARLIST(NAME, OP, DIV_OP)                                                                   \
void foreach_tensor_##NAME##_scalarlist_kernel_cuda_(TensorList tensors, at::ArrayRef<Scalar> scalars) {                 \
    check_foreach_api_restrictions(tensors, scalars);                                                                    \
    if (can_use_fast_route(tensors, scalars, DIV_OP)) {                                                                  \
        return foreach_binary_op_<OP>(tensors, scalars);                                                                 \
    }                                                                                                                    \
    OptionalDeviceGuard g;                                                                                               \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors, scalars)) {                           \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                                            \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                                       \
            auto & mta_input_tensors = dtype_mtainputs.second.mta_input_tensors;                                         \
            if (!dtype_mtainputs.second.is_mta_compatible(DIV_OP)) {                                                     \
                at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow_(                                             \
                        TensorList(mta_input_tensors[0]), dtype_mtainputs.second.get_scalarList());                      \
            } else {                                                                                                     \
                foreach_binary_op_<OP>(mta_input_tensors, dtype_mtainputs.second.get_scalarList());                      \
            }                                                                                                            \
        }                                                                                                                \
    }                                                                                                                    \
}                                                                                                                        \
                                                                                                                         \
std::vector<Tensor> foreach_tensor_##NAME##_scalarlist_kernel_cuda(TensorList tensors, at::ArrayRef<Scalar> scalars) {   \
    check_foreach_api_restrictions(tensors, scalars);                                                                    \
    if (can_use_fast_route(tensors, scalars, DIV_OP)) {                                                                  \
        return foreach_binary_op<OP>(tensors, scalars);                                                                  \
    }                                                                                                                    \
    std::vector<Tensor> result;                                                                                          \
    result.reserve(tensors.size());                                                                                      \
    std::vector<size_t> index_map(tensors.size(), 0);                                                                    \
                                                                                                                         \
    size_t cur_index = 0;                                                                                                \
    OptionalDeviceGuard g;                                                                                               \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors, scalars)) {                           \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                                            \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                                       \
            auto & mta_input_tensors = dtype_mtainputs.second.mta_input_tensors;                                         \
            if (!dtype_mtainputs.second.is_mta_compatible(DIV_OP)) {                                                     \
                for (const auto & t : at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow(                        \
                        TensorList(mta_input_tensors[0]), dtype_mtainputs.second.get_scalarList())) {                    \
                    result.emplace_back(std::move(t));                                                                   \
                }                                                                                                        \
            } else {                                                                                                     \
                for (const auto & t : foreach_binary_op<OP>(                                                             \
                        mta_input_tensors, dtype_mtainputs.second.get_scalarList())) {                                   \
                    result.emplace_back(std::move(t));                                                                   \
                }                                                                                                        \
            }                                                                                                            \
            const auto & indices = dtype_mtainputs.second.indices;                                                       \
            for (const auto & i : c10::irange(indices.size())) {                                                         \
                index_map[indices[i]] = cur_index + i;                                                                   \
            }                                                                                                            \
            cur_index += indices.size();                                                                                 \
        }                                                                                                                \
    }                                                                                                                    \
    std::vector<Tensor> sorted_result;                                                                                   \
    sorted_result.reserve(tensors.size());                                                                               \
    for (const auto & i : index_map) {                                                                                   \
        sorted_result.emplace_back(std::move(result[i]));                                                                \
    }                                                                                                                    \
    return sorted_result;                                                                                                \
}

FOREACH_BINARY_OP_SCALARLIST(add, std::plus, /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(mul, std::multiplies, /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(div, std::divides, /*div_op*/ true);

// This does not use FOREACH_BINARY_OP_SCALARLIST because
// In the case of subtraction, we dont allow scalar to be boolean following the torch.sub logic
void foreach_tensor_sub_scalarlist_kernel_cuda_(TensorList tensors, at::ArrayRef<Scalar> scalars) {
    check_foreach_api_restrictions(tensors, scalars);
    for (const auto & i : c10::irange(tensors.size())) {
        sub_check(tensors[i], scalars[i]);
    }
    if (can_use_fast_route({tensors}, scalars)) {
        return foreach_binary_op_<std::minus>(tensors, scalars);
    }
    OptionalDeviceGuard g;
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors, scalars)) {
        g.reset_device(at::Device(dev_dtypemtainputs.first));
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {
            auto & mta_input_tensors = dtype_mtainputs.second.mta_input_tensors;
            if (!dtype_mtainputs.second.is_mta_compatible()) {
                at::native::foreach_tensor_sub_scalarlist_kernel_slow_(TensorList{mta_input_tensors[0]}, dtype_mtainputs.second.get_scalarList());
            } else {
                foreach_binary_op_<std::minus>(mta_input_tensors, dtype_mtainputs.second.get_scalarList());
            }
        }
    }
}

std::vector<Tensor> foreach_tensor_sub_scalarlist_kernel_cuda(TensorList tensors, at::ArrayRef<Scalar> scalars) {
    check_foreach_api_restrictions(tensors, scalars);
    for (const auto & i : c10::irange(tensors.size())) {
        sub_check(tensors[i], scalars[i]);
    }
    if (can_use_fast_route({tensors}, scalars)) {
        return foreach_binary_op<std::minus>(tensors, scalars);
    }

    std::vector<Tensor> result;
    result.reserve(tensors.size());
    std::vector<size_t> index_map(tensors.size(), 0);

    size_t cur_index = 0;
    OptionalDeviceGuard g;
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors, scalars)) {
        g.reset_device(at::Device(dev_dtypemtainputs.first));
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {
            auto & mta_input_tensors = dtype_mtainputs.second.mta_input_tensors;
            if(!dtype_mtainputs.second.is_mta_compatible()) {
                for (const auto & t : at::native::foreach_tensor_sub_scalarlist_kernel_slow(
                        TensorList(mta_input_tensors[0]), dtype_mtainputs.second.get_scalarList())) {
                    result.emplace_back(std::move(t));
                }
            } else {
                for (const auto & t : foreach_binary_op<std::minus>(mta_input_tensors, dtype_mtainputs.second.get_scalarList())) {
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
