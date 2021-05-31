#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

namespace at { namespace native {

template<template<class> class Op>
std::vector<Tensor> foreach_tensor_list_op(TensorList tensors1, TensorList tensors2, const Scalar& alpha = 1) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors1.size());
    for (const auto& t: tensors1) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(tensors1.vec());
    tensor_lists.emplace_back(tensors2.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors1[0].scalar_type(), "foreach_binary_op_list_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<3>(tensor_lists,
                              BinaryOpListAlphaFunctor<scalar_t,
                                                       /* depth */ 3,
                                                       /* r_args_depth */ 2,
                                                       /* res_arg_index */ 2>(),
                              Op<opmath_t>(),
                              alpha.to<opmath_t>());
    });

    return tensor_lists[2];
}

template<template<class> class Op>
std::vector<Tensor> foreach_tensor_list_op(std::vector<std::vector<Tensor>>& mta_input_tensors, const Scalar& alpha = 1) {
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(mta_input_tensors[0].size());
    for (const auto & t : mta_input_tensors[0]) {
        vec_res.emplace_back(at::native::empty_like(t));
    }
    mta_input_tensors.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kBool, kBFloat16, kHalf,
        mta_input_tensors[0][0].scalar_type(),
        "foreach_binary_op_list_cuda",
        [&]() {
            using opmath_t = get_opmath_t<scalar_t>::opmath_t;
            multi_tensor_apply<3>(
                mta_input_tensors,
                BinaryOpListAlphaFunctor<scalar_t, /* depth */ 3, /* r_args_depth */ 2, /* res_arg_index */ 2>(),
                Op<opmath_t>(),
                alpha.to<opmath_t>());
        }
    );

    return std::move(mta_input_tensors[2]);
}

template<template<class> class Op>
void foreach_tensor_list_op_(TensorList tensors1, TensorList tensors2, const Scalar& alpha = 1) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(tensors1.vec());
    tensor_lists.emplace_back(tensors2.vec());

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, tensors1[0].scalar_type(), "foreach_binary_op_list_cuda_", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<2>(tensor_lists,
                              BinaryOpListAlphaFunctor<scalar_t,
                                                       /* depth */ 2,
                                                       /* r_args_depth */ 2,
                                                       /* res_arg_index */ 0>(),
                              Op<opmath_t>(),
                              alpha.to<opmath_t>());
    });
}

template<template<class> class Op>
void foreach_tensor_list_op_(std::vector<std::vector<at::Tensor>> & mta_input_tensors, const Scalar& alpha = 1) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kBool, kBFloat16, kHalf,
        mta_input_tensors[0][0].scalar_type(),
        "foreach_binary_op_list_cuda_",
        [&]() {
            using opmath_t = get_opmath_t<scalar_t>::opmath_t;
            multi_tensor_apply<2>(
                mta_input_tensors,
                BinaryOpListAlphaFunctor<scalar_t, /* depth */ 2, /* r_args_depth */ 2, /* res_arg_index */ 0>(),
                Op<opmath_t>(),
                alpha.to<opmath_t>());
        }
    );
}

#define FOREACH_BINARY_OP_LIST(NAME, OP, DIVISION_OP)                                                       \
void foreach_tensor_##NAME##_list_kernel_cuda_(TensorList tensors1, TensorList tensors2) {                  \
    check_foreach_api_restrictions(tensors1, tensors2);                                                     \
    if (can_use_fast_route(tensors1, tensors2, DIVISION_OP)) {                                              \
        return foreach_tensor_list_op_<OP>(tensors1, tensors2);                                             \
    }                                                                                                       \
    OptionalDeviceGuard g;                                                                                  \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors1, tensors2)) {            \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                               \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                          \
            auto & mta_input_tensors = dtype_mtainputs.second.mta_input_tensors;                            \
            if (!dtype_mtainputs.second.is_mta_compatible(DIVISION_OP)) {                                   \
                at::native::foreach_tensor_##NAME##_list_kernel_slow(                                       \
                    TensorList{mta_input_tensors[0]}, TensorList{mta_input_tensors[1]});                    \
            } else {                                                                                        \
                foreach_tensor_list_op_<OP>(mta_input_tensors);                                             \
            }                                                                                               \
        }                                                                                                   \
    }                                                                                                       \
}                                                                                                           \
                                                                                                            \
std::vector<Tensor> foreach_tensor_##NAME##_list_kernel_cuda(TensorList tensors1, TensorList tensors2) {    \
    check_foreach_api_restrictions(tensors1, tensors2);                                                     \
    if (can_use_fast_route(tensors1, tensors2, DIVISION_OP)) {                                              \
        return foreach_tensor_list_op<OP>(tensors1, tensors2);                                              \
    }                                                                                                       \
    std::vector<at::Tensor> result;                                                                         \
    result.reserve(tensors1.size());                                                                        \
    std::vector<size_t> index_map(tensors1.size());                                                         \
                                                                                                            \
    size_t cur_index = 0;                                                                                   \
    OptionalDeviceGuard g;                                                                                  \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors1, tensors2)) {            \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                               \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                          \
            auto & mta_input_tensors = dtype_mtainputs.second.mta_input_tensors;                            \
            if (!dtype_mtainputs.second.is_mta_compatible()) {                                              \
                for (const auto & t : at::native::foreach_tensor_##NAME##_list_kernel_slow(                 \
                        TensorList{mta_input_tensors[0]}, TensorList{mta_input_tensors[1]})) {              \
                    result.emplace_back(std::move(t));                                                      \
                }                                                                                           \
            } else {                                                                                        \
                for (const auto & t : foreach_tensor_list_op<OP>(mta_input_tensors)) {                      \
                    result.emplace_back(std::move(t));                                                      \
                }                                                                                           \
            }                                                                                               \
            const auto & indices = dtype_mtainputs.second.indices;                                          \
            for (const auto & i : c10::irange(indices.size())) {                                            \
                index_map[indices[i]] = cur_index + i;                                                      \
            }                                                                                               \
            cur_index += indices.size();                                                                    \
        }                                                                                                   \
    }                                                                                                       \
    std::vector<at::Tensor> sorted_result;                                                                  \
    sorted_result.reserve(tensors1.size());                                                                 \
    for (const auto & i : index_map) {                                                                      \
        sorted_result.emplace_back(std::move(result[i]));                                                   \
    }                                                                                                       \
    return sorted_result;                                                                                   \
}

#define FOREACH_BINARY_OP_LIST_ALPHA(NAME, OP)                                                                          \
void foreach_tensor_##NAME##_list_kernel_cuda_(TensorList tensors1, TensorList tensors2, const Scalar& alpha) {                \
    check_foreach_api_restrictions(tensors1, tensors2);                                                                 \
    if (can_use_fast_route({tensors1, tensors2}, alpha)) {                                                              \
        return foreach_tensor_list_op_<OP>(tensors1, tensors2, alpha);                                                  \
    }                                                                                                                   \
    OptionalDeviceGuard g;                                                                                              \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors1, tensors2)) {                        \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                                           \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                                      \
            auto & mta_input_tensors = dtype_mtainputs.second.mta_input_tensors;                                        \
            if (!dtype_mtainputs.second.is_mta_compatible(alpha)) {                                                     \
                at::native::foreach_tensor_##NAME##_list_kernel_slow_(                                                  \
                    TensorList{mta_input_tensors[0]}, TensorList{mta_input_tensors[1]}, alpha);                         \
            } else {                                                                                                    \
                foreach_tensor_list_op_<OP>(mta_input_tensors, alpha);                                                  \
            }                                                                                                           \
        }                                                                                                               \
    }                                                                                                                   \
}                                                                                                                       \
                                                                                                                        \
std::vector<Tensor> foreach_tensor_##NAME##_list_kernel_cuda(TensorList tensors1, TensorList tensors2, const Scalar& alpha) {  \
    check_foreach_api_restrictions(tensors1, tensors2);                                                                 \
    if (can_use_fast_route({tensors1, tensors2}, alpha)) {                                                              \
        return foreach_tensor_list_op<OP>(tensors1, tensors2, alpha);                                                   \
    }                                                                                                                   \
                                                                                                                        \
    std::vector<at::Tensor> result;                                                                                     \
    result.reserve(tensors1.size());                                                                                    \
    std::vector<size_t> index_map(tensors1.size(), 0);                                                                  \
                                                                                                                        \
    size_t cur_index = 0;                                                                                               \
    OptionalDeviceGuard g;                                                                                              \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors1, tensors2)) {                        \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                                           \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                                      \
            auto & mta_input_tensors = dtype_mtainputs.second.mta_input_tensors;                                        \
            if (!dtype_mtainputs.second.is_mta_compatible(alpha)) {                                                     \
                for (const auto & t : at::native::foreach_tensor_##NAME##_list_kernel_slow(                             \
                        TensorList{mta_input_tensors[0]}, TensorList{mta_input_tensors[1]}, alpha)) {                   \
                    result.emplace_back(std::move(t));                                                                  \
                }                                                                                                       \
            } else {                                                                                                    \
                for (const auto & t : foreach_tensor_list_op<OP>(mta_input_tensors, alpha)) {                           \
                    result.emplace_back(std::move(t));                                                                  \
                }                                                                                                       \
            }                                                                                                           \
            const auto & indices = dtype_mtainputs.second.indices;                                                      \
            for (const auto & i : c10::irange(indices.size())) {                                                        \
                index_map[indices[i]] = cur_index + i;                                                                  \
            }                                                                                                           \
            cur_index += indices.size();                                                                                \
        }                                                                                                               \
    }                                                                                                                   \
    std::vector<at::Tensor> sorted_result;                                                                              \
    sorted_result.reserve(tensors1.size());                                                                             \
    for (const auto & i : index_map) {                                                                                  \
        sorted_result.emplace_back(result[i]);                                                                          \
    }                                                                                                                   \
    return sorted_result;                                                                                               \
}

FOREACH_BINARY_OP_LIST_ALPHA(add, std::plus);
FOREACH_BINARY_OP_LIST_ALPHA(sub, std::minus);
FOREACH_BINARY_OP_LIST(mul, std::multiplies, /*division_op*/ false);
FOREACH_BINARY_OP_LIST(div, std::divides, /*division_op*/ true);

}} // namespace at::native
