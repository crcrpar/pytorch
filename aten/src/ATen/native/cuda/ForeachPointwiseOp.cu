#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/NumericUtils.h>

namespace at { namespace native {

template<template<class> class Op>
std::vector<Tensor> foreach_pointwise_op(TensorList input, TensorList tensors1, TensorList tensors2, const Scalar& scalar) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(input.size());
    for (const auto& t: input) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(input.vec());
    tensor_lists.emplace_back(tensors1.vec());
    tensor_lists.emplace_back(tensors2.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, input[0].scalar_type(), "foreach_pointwise_op_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<4>(tensor_lists,
                              PointwiseOpScalarFunctor<scalar_t,
                                                       /* depth */ 4,
                                                       /* r_args_depth */ 3,
                                                       /* res_arg_index */ 3>(),
                              Op<opmath_t>(),
                              scalar.to<opmath_t>());
    });

    return tensor_lists[3];
}

template<template<class> class Op>
std::vector<Tensor> foreach_pointwise_op(std::vector<std::vector<at::Tensor>>& mta_input_tensors, const Scalar& scalar) {
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(mta_input_tensors[0].size());
    for (const auto& t: mta_input_tensors[0]) {
        vec_res.emplace_back(at::native::empty_like(t));
    }
    mta_input_tensors.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kHalf, mta_input_tensors[0][0].scalar_type(), "foreach_pointwise_op_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<4>(mta_input_tensors,
                              PointwiseOpScalarFunctor<scalar_t,
                                                       /* depth */ 4,
                                                       /* r_args_depth */ 3,
                                                       /* res_arg_index */ 3>(),
                              Op<opmath_t>(),
                              scalar.to<opmath_t>());
    });

    return std::move(mta_input_tensors[3]);
}

template<template<class> class Op>
void foreach_pointwise_op_(TensorList input, TensorList tensors1, TensorList tensors2, const Scalar& scalar) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.emplace_back(input.vec());
    tensor_lists.emplace_back(tensors1.vec());
    tensor_lists.emplace_back(tensors2.vec());

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, input[0].scalar_type(), "foreach_pointwise_op__cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<3>(tensor_lists,
                              PointwiseOpScalarFunctor<scalar_t,
                                                       /* depth */ 3,
                                                       /* r_args_depth */ 3,
                                                       /* res_arg_index */ 0>(),
                              Op<opmath_t>(),
                              scalar.to<opmath_t>());
    });
}

template<template<class> class Op>
void foreach_pointwise_op_(std::vector<std::vector<at::Tensor>>& mta_input_tensors, const Scalar& scalar) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kHalf, mta_input_tensors[0][0].scalar_type(), "foreach_pointwise_op__cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<3>(mta_input_tensors,
                              PointwiseOpScalarFunctor<scalar_t,
                                                       /* depth */ 3,
                                                       /* r_args_depth */ 3,
                                                       /* res_arg_index */ 0>(),
                              Op<opmath_t>(),
                              scalar.to<opmath_t>());
    });
}

template<template<class> class Op>
void foreach_pointwise_op_(TensorList input, TensorList tensors1, TensorList tensors2, at::ArrayRef<Scalar> scalars) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.reserve(3);
    tensor_lists.emplace_back(input.vec());
    tensor_lists.emplace_back(tensors1.vec());
    tensor_lists.emplace_back(tensors2.vec());

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, input[0].scalar_type(), "foreach_pointwise_op__cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<3, opmath_t>(tensor_lists,
                                        scalars,
                                        PointwiseOpScalarListFunctor<scalar_t,
                                                                     /* depth */ 3,
                                                                     /* r_args_depth */ 3,
                                                                     /* res_arg_index */ 0>(),
                                        Op<opmath_t>());
    });
}

template<template<class> class Op>
void foreach_pointwise_op_(std::vector<std::vector<at::Tensor>>& mta_input_tensors, at::ArrayRef<Scalar> scalars) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kHalf, mta_input_tensors[0][0].scalar_type(), "foreach_pointwise_op__cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<3, opmath_t>(mta_input_tensors,
                                        scalars,
                                        PointwiseOpScalarListFunctor<scalar_t,
                                                                     /* depth */ 3,
                                                                     /* r_args_depth */ 3,
                                                                     /* res_arg_index */ 0>(),
                                        Op<opmath_t>());
    });
}

template<template<class> class Op>
std::vector<Tensor> foreach_pointwise_op(TensorList input, TensorList tensors1, TensorList tensors2, at::ArrayRef<Scalar> scalars) {
    std::vector<std::vector<at::Tensor>> tensor_lists;
    tensor_lists.reserve(4);
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(input.size());
    for (const auto& t: input) {
        vec_res.emplace_back(at::native::empty_like(t));
    }

    tensor_lists.emplace_back(input.vec());
    tensor_lists.emplace_back(tensors1.vec());
    tensor_lists.emplace_back(tensors2.vec());
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, input[0].scalar_type(), "foreach_pointwise_op_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<4, opmath_t>(tensor_lists,
                                        scalars,
                                        PointwiseOpScalarListFunctor<scalar_t,
                                                                     /* depth */ 4,
                                                                     /* r_args_depth */ 3,
                                                                     /* res_arg_index */ 3>(),
                                        Op<opmath_t>());
    });

    return tensor_lists[3];
}

template<template<class> class Op>
std::vector<Tensor> foreach_pointwise_op(std::vector<std::vector<at::Tensor>>& mta_input_tensors, at::ArrayRef<Scalar> scalars) {
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(mta_input_tensors[0].size());
    for (const auto& t: mta_input_tensors[0]) {
        vec_res.emplace_back(at::native::empty_like(t));
    }
    mta_input_tensors.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kHalf, mta_input_tensors[0][0].scalar_type(), "foreach_pointwise_op_cuda", [&]() {
        using opmath_t = get_opmath_t<scalar_t>::opmath_t;
        multi_tensor_apply<4, opmath_t>(mta_input_tensors,
                                        scalars,
                                        PointwiseOpScalarListFunctor<scalar_t,
                                                                     /* depth */ 4,
                                                                     /* r_args_depth */ 3,
                                                                     /* res_arg_index */ 3>(),
                                        Op<opmath_t>());
    });

    return std::move(mta_input_tensors[3]);
}

#define FOREACH_POINTWISE_OP_SCALAR(NAME, OP)                                                                                         \
std::vector<Tensor> foreach_tensor_##NAME##_scalar_cuda(TensorList input, TensorList tensors1, TensorList tensors2, const Scalar& scalar) {  \
    check_foreach_api_restrictions(input, tensors1, tensors2);                                                                        \
    OptionalDeviceGuard g;                                                                                                            \
    if (can_use_fast_route({input, tensors1, tensors2}, scalar) && !has_integral_tensor(input, /* includeBool */ true)) {             \
        g.reset_device(input[0].device());                                                                                            \
        return foreach_pointwise_op<OP>(input, tensors1, tensors2, scalar);                                                           \
    }                                                                                                                                 \
    std::vector<Tensor> result;                                                                                                       \
    result.reserve(input.size());                                                                                                     \
    std::vector<size_t> index_map(input.size(), 0);                                                                                   \
                                                                                                                                      \
    size_t cur_index = 0;                                                                                                             \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(input, tensors1, tensors2)) {                               \
        const bool is_cpu = (dev_dtypemtainputs.first == "cpu");                                                                      \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                                                         \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                                                    \
            auto & mtainput = dtype_mtainputs.second;                                                                                 \
            if (is_cpu || !mtainput.is_mta_compatible(scalar) || has_integral_tensor(mtainput.get_tensorList(0), /* includeBool */ true)) {     \
                for (const auto & t : at::native::foreach_tensor_##NAME##_scalar_slow(                                                \
                        mtainput.get_tensorList(0), mtainput.get_tensorList(1), mtainput.get_tensorList(2), scalar)) {                \
                    result.emplace_back(std::move(t));                                                                                \
                }                                                                                                                     \
            } else {                                                                                                                  \
                for (const auto & t : foreach_pointwise_op<OP>(mtainput.mta_input_tensors, scalar)) {                                 \
                    result.emplace_back(std::move(t));                                                                                \
                }                                                                                                                     \
            }                                                                                                                         \
            const auto & indices = mtainput.indices;                                                                                  \
            for (const auto & i : c10::irange(indices.size())) {                                                                      \
                index_map[indices[i]] = cur_index + i;                                                                                \
            }                                                                                                                         \
            cur_index += indices.size();                                                                                              \
        }                                                                                                                             \
    }                                                                                                                                 \
    std::vector<Tensor> sorted_result;                                                                                                \
    sorted_result.reserve(input.size());                                                                                              \
    for (const auto & i : index_map) {                                                                                                \
        sorted_result.emplace_back(result[i]);                                                                                        \
    }                                                                                                                                 \
    return sorted_result;                                                                                                             \
}                                                                                                                                     \
                                                                                                                                      \
void foreach_tensor_##NAME##_scalar_cuda_(TensorList input, TensorList tensors1, TensorList tensors2, const Scalar& scalar) {                \
    check_foreach_api_restrictions(input, tensors1, tensors2);                                                                        \
    OptionalDeviceGuard g;                                                                                                            \
    if (can_use_fast_route({input, tensors1, tensors2}, scalar) && !has_integral_tensor(input, /* includeBool */ true)) {             \
        g.reset_device(input[0].device());                                                                                            \
        return foreach_pointwise_op_<OP>(input, tensors1, tensors2, scalar);                                                          \
    }                                                                                                                                 \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(input, tensors1, tensors2)) {                               \
        const bool is_cpu = (dev_dtypemtainputs.first == "cpu");                                                                      \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                                                         \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                                                    \
            auto & mtainput = dtype_mtainputs.second;                                                                                 \
            if (is_cpu || !mtainput.is_mta_compatible(scalar) || has_integral_tensor(mtainput.get_tensorList(0), /* includeBool */ true)) {     \
                at::native::foreach_tensor_##NAME##_scalar_slow_(                                                                     \
                    mtainput.get_tensorList(0), mtainput.get_tensorList(1), mtainput.get_tensorList(2), scalar);                      \
            } else {                                                                                                                  \
                foreach_pointwise_op_<OP>(mtainput.mta_input_tensors, scalar);                                                        \
            }                                                                                                                         \
        }                                                                                                                             \
    }                                                                                                                                 \
}


#define FOREACH_POINTWISE_OP_SCALARLIST(NAME, OP)                                                                                                        \
std::vector<Tensor> foreach_tensor_##NAME##_scalarlist_cuda(TensorList input, TensorList tensors1, TensorList tensors2, at::ArrayRef<Scalar> scalars) {  \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);                                                                                  \
    OptionalDeviceGuard g;                                                                                                                               \
    if (can_use_fast_route({input, tensors1, tensors2}, scalars) && !has_integral_tensor(input, /* includeBool */ true)) {                               \
        g.reset_device(input[0].device());                                                                                                               \
        return foreach_pointwise_op<OP>(input, tensors1, tensors2, scalars);                                                                             \
    }                                                                                                                                                    \
                                                                                                                                                         \
    std::vector<Tensor> result;                                                                                                                          \
    result.reserve(input.size());                                                                                                                        \
    std::vector<size_t> index_map(input.size(), 0);                                                                                                      \
                                                                                                                                                         \
    size_t cur_index = 0;                                                                                                                                \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(input, tensors1, tensors2, scalars)) {                                         \
        const bool is_cpu = (dev_dtypemtainputs.first == "cpu");                                                                                         \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                                                                            \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                                                                       \
            auto & mtainput = dtype_mtainputs.second;                                                                                                    \
            if (is_cpu || !mtainput.is_mta_compatible() || has_integral_tensor(mtainput.get_tensorList(0), /* includeBool */ true)) {                    \
                for (const auto & t : at::native::foreach_tensor_##NAME##_scalarlist_slow(                                                               \
                        mtainput.get_tensorList(0), mtainput.get_tensorList(1), mtainput.get_tensorList(2), mtainput.get_scalarList())) {                \
                    result.emplace_back(std::move(t));                                                                                                   \
                }                                                                                                                                        \
            } else {                                                                                                                                     \
                for (const auto & t : foreach_pointwise_op<OP>(mtainput.mta_input_tensors, mtainput.get_scalarList())) {                                 \
                    result.emplace_back(std::move(t));                                                                                                   \
                }                                                                                                                                        \
            }                                                                                                                                            \
            const auto & indices = mtainput.indices;                                                                                                     \
            for (const auto & i : c10::irange(indices.size())) {                                                                                         \
                index_map[indices[i]] = cur_index + i;                                                                                                   \
            }                                                                                                                                            \
            cur_index += indices.size();                                                                                                                 \
        }                                                                                                                                                \
    }                                                                                                                                                    \
    std::vector<Tensor> sorted_result;                                                                                                                   \
    sorted_result.reserve(input.size());                                                                                                                 \
    for (const auto & i : index_map) {                                                                                                                   \
        sorted_result.emplace_back(result[i]);                                                                                                           \
    }                                                                                                                                                    \
    return sorted_result;                                                                                                                                \
}                                                                                                                                                        \
                                                                                                                                                         \
void foreach_tensor_##NAME##_scalarlist_cuda_(TensorList input, TensorList tensors1, TensorList tensors2, at::ArrayRef<Scalar> scalars) {                \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);                                                                                  \
    OptionalDeviceGuard g;                                                                                                                               \
    if (can_use_fast_route({input, tensors1, tensors2}, scalars) && !has_integral_tensor(input, /* includeBool */ true)) {                               \
        g.reset_device(input[0].device());                                                                                                               \
        return foreach_pointwise_op_<OP>(input, tensors1, tensors2, scalars);                                                                            \
    }                                                                                                                                                    \
                                                                                                                                                         \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(input, tensors1, tensors2, scalars)) {                                         \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                                                                            \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                                                                       \
            auto & mtainput = dtype_mtainputs.second;                                                                                                    \
            if (!mtainput.is_mta_compatible() || has_integral_tensor(mtainput.get_tensorList(0), /* includeBool */ true)) {                              \
                at::native::foreach_tensor_##NAME##_scalarlist_slow_(                                                                                    \
                    mtainput.get_tensorList(0), mtainput.get_tensorList(1), mtainput.get_tensorList(2), mtainput.get_scalarList());                      \
            } else {                                                                                                                                     \
                foreach_pointwise_op_<OP>(mtainput.mta_input_tensors, mtainput.get_scalarList());                                                        \
            }                                                                                                                                            \
        }                                                                                                                                                \
    }                                                                                                                                                    \
}

FOREACH_POINTWISE_OP_SCALAR(addcmul, std::multiplies);
FOREACH_POINTWISE_OP_SCALAR(addcdiv, std::divides);
FOREACH_POINTWISE_OP_SCALARLIST(addcmul, std::multiplies);
FOREACH_POINTWISE_OP_SCALARLIST(addcdiv, std::divides);

// FIXME(mkozuki): Float16 tests failing
template<template<class> class Op>
std::vector<Tensor> foreach_maximum_minimum_op(TensorList tensors1, TensorList tensors2) {
    std::vector<std::vector<Tensor>> tensor_lists{tensors1.vec(), tensors2.vec()};
    std::vector<Tensor> vec_res;
    vec_res.reserve(tensors1.size());
    for (const auto & t : tensors1) {
        vec_res.emplace_back(at::native::empty_like(t));
    }
    tensor_lists.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND(
        kHalf, tensors1[0].scalar_type(), "foreach_maximum_minimum_op_cuda",
        [&]() {
            using opmath_t = get_opmath_t<scalar_t>::opmath_t;
            multi_tensor_apply<3>(
                tensor_lists,
                PointwiseOpListFunctor<opmath_t, 3>(),
                Op<opmath_t>()
            );
        }
    );
    return tensor_lists[2];
}

template<template<class> class Op>
std::vector<Tensor> foreach_maximum_minimum_op(std::vector<std::vector<Tensor>>& mta_input_tensors) {
    std::vector<Tensor> vec_res;
    vec_res.reserve(mta_input_tensors[0].size());
    for (const auto & t : mta_input_tensors[0]) {
        vec_res.emplace_back(at::native::empty_like(t));
    }
    mta_input_tensors.emplace_back(std::move(vec_res));

    AT_DISPATCH_ALL_TYPES_AND(
        kHalf, mta_input_tensors[0][0].scalar_type(), "foreach_maximum_minimum_op_cuda",
        [&]() {
            using opmath_t = get_opmath_t<scalar_t>::opmath_t;
            multi_tensor_apply<3>(
                mta_input_tensors,
                PointwiseOpListFunctor<opmath_t, 3>(),
                Op<opmath_t>()
            );
        }
    );
    return std::move(mta_input_tensors[2]);
}

#include <limits>
#define FUNCTOR(op_name, functor_name)             \
template<typename T>                               \
struct functor_name {                              \
    __device__ T operator()(T t1, T t2) const {    \
        if (t1 != t1 || t2 != t2) {                \
            return std::numeric_limits<T>::quiet_NaN(); \
        } else {                                   \
            return ::op_name(t1, t2);              \
        }                                          \
    }                                              \
};                                                 \

// Why bool tensors are pushed to slowpath?
// Because `AT_DISPATCH_ALL_TYPES_AND` is used below.
// TODO(mkozuki): Check whether it's possible to handle bool tensors in fastpath.
#define FOREACH_MAXIMUM_MINIMUM_OP(NAME, OP_NAME, FUNCTOR_NAME)                                 \
FUNCTOR(OP_NAME, FUNCTOR_NAME);                                                                 \
std::vector<Tensor> foreach_tensor_##NAME##_cuda(TensorList tensors1, TensorList tensors2) {    \
    check_foreach_api_restrictions(tensors1, tensors2);                                         \
    OptionalDeviceGuard g;                                                                      \
    if (can_use_fast_route({tensors1, tensors2}) && !has_bool_tensor(tensors1)) {               \
        g.reset_device(tensors1[0].device());                                                   \
        return foreach_maximum_minimum_op<FUNCTOR_NAME>(tensors1, tensors2);                    \
    }                                                                                           \
    std::vector<Tensor> result;                                                                 \
    result.reserve(tensors1.size());                                                            \
    std::vector<size_t> index_map(tensors1.size(), 0);                                          \
                                                                                                \
    size_t cur_index = 0;                                                                       \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors1, tensors2)) {                                            \
        const bool is_cpu = (dev_dtypemtainputs.first == "cpu");                                                                            \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                                                               \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                                                          \
            auto & mtainput = dtype_mtainputs.second;                                                                                       \
            if (is_cpu || !mtainput.is_mta_compatible() || has_bool_tensor(mtainput.get_tensorList(0))) {                                   \
                for (const auto & t : at::native::foreach_tensor_##NAME##_slow(mtainput.get_tensorList(0), mtainput.get_tensorList(1))) {   \
                    result.emplace_back(std::move(t));                                                                                      \
                }                                                                                                                           \
            } else {                                                                                                                        \
                for (const auto & t : foreach_maximum_minimum_op<FUNCTOR_NAME>(mtainput.mta_input_tensors)) {                               \
                    result.emplace_back(std::move(t));                                          \
                }                                                                               \
            }                                                                                   \
            const auto & indices = mtainput.indices;                                            \
            for (const auto & i : c10::irange(indices.size())) {                                \
                index_map[indices[i]] = cur_index + i;                                          \
            }                                                                                   \
            cur_index += indices.size();                                                        \
        }                                                                                       \
    }                                                                                           \
    std::vector<Tensor> sorted_result;                                                          \
    sorted_result.reserve(tensors1.size());                                                     \
    for (const auto & i : index_map) {                                                          \
        sorted_result.emplace_back(std::move(result[i]));                                       \
    }                                                                                           \
    return sorted_result;                                                                       \
}                                                                                               \

FOREACH_MAXIMUM_MINIMUM_OP(maximum, max, MaximumFunctor);
FOREACH_MAXIMUM_MINIMUM_OP(minimum, min, MinimumFunctor);

}} // namespace at::native
