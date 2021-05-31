#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

namespace at { namespace native {

template <typename scalar_t, template<class> class Op>
std::vector<at::Tensor> foreach_unary_op(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists{tensors.vec()};
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(tensors.size());
    for (const auto & t : tensors) {
        vec_res.emplace_back(at::native::empty_like(t));
    }
    tensor_lists.emplace_back(std::move(vec_res));

    using opmath_t = typename get_opmath_t<scalar_t>::opmath_t;
    multi_tensor_apply<2>(
        tensor_lists,
        UnaryOpFunctor<scalar_t, /* depth */ 2, /* r_args_depth */ 1, /* res_arg_index */ 1>(),
        Op<opmath_t>()
    );
    return tensor_lists[1];
}

template <typename scalar_t, template<class> class Op>
std::vector<at::Tensor> foreach_unary_op(std::vector<std::vector<at::Tensor>>& mta_input_tensors) {
    std::vector<at::Tensor> vec_res;
    vec_res.reserve(mta_input_tensors[0].size());
    for (const auto & t : mta_input_tensors[0]) {
        vec_res.emplace_back(at::native::empty_like(t));
    }
    mta_input_tensors.emplace_back(std::move(vec_res));

    using opmath_t = typename get_opmath_t<scalar_t>::opmath_t;
    multi_tensor_apply<2>(
        mta_input_tensors,
        UnaryOpFunctor<scalar_t, /* depth */ 2, /* r_args_depth */ 1, /* res_arg_index */ 1>(),
        Op<opmath_t>()
    );
    return mta_input_tensors[1];
}

template <typename scalar_t, template<class> class Op>
void foreach_unary_op_(TensorList tensors) {
    std::vector<std::vector<at::Tensor>> tensor_lists{tensors.vec()};
    using opmath_t = typename get_opmath_t<scalar_t>::opmath_t;
    multi_tensor_apply<1>(
        tensor_lists,
        UnaryOpFunctor<scalar_t, /* depth */ 1, /* r_args_depth */ 1, /* res_arg_index */ 0>(),
        Op<opmath_t>()
    );
}

template <typename scalar_t, template<class> class Op>
void foreach_unary_op_(std::vector<std::vector<at::Tensor>>& mta_input_tensors) {
    using opmath_t = typename get_opmath_t<scalar_t>::opmath_t;
    multi_tensor_apply<1>(
        mta_input_tensors,
        UnaryOpFunctor<scalar_t, /* depth */ 1, /* r_args_depth */ 1, /* res_arg_index */ 0>(),
        Op<opmath_t>()
    );
}

template <template<class> class Op>
std::vector<Tensor> floating_complex_half(TensorList tensors) {
    return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half,  tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
std::vector<Tensor> floating_complex_half(std::vector<std::vector<Tensor>>& mta_input_tensors) {
    return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half,  mta_input_tensors[0][0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(mta_input_tensors);
    });
}

template <template<class> class Op>
void floating_complex_half_(TensorList tensors) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
        ScalarType::Half, tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        foreach_unary_op_<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
void floating_complex_half_(std::vector<std::vector<at::Tensor>>& mta_input_tensors) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
        ScalarType::Half, mta_input_tensors[0][0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        foreach_unary_op_<scalar_t, Op>(mta_input_tensors);
    });
}

template <template<class> class Op>
std::vector<Tensor> all_types_complex_bfloat16_half_bool(TensorList tensors) {
    return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
std::vector<Tensor> all_types_complex_bfloat16_half_bool(std::vector<std::vector<Tensor>>& mta_input_tensors) {
    return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, mta_input_tensors[0][0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(mta_input_tensors);
    });
}

template <template<class> class Op>
void all_types_complex_bfloat16_half_bool_(TensorList tensors) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool,
        tensors[0].scalar_type(), "foreach_unary_op_cuda",
        [&]() {
            foreach_unary_op_<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
void all_types_complex_bfloat16_half_bool_(std::vector<std::vector<at::Tensor>>& mta_input_tensors) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool,
        mta_input_tensors[0][0].scalar_type(), "foreach_unary_op_cuda",
        [&]() {
            foreach_unary_op_<scalar_t, Op>(mta_input_tensors);
    });
}

template <template<class> class Op>
std::vector<Tensor> floating_complex_half_bfloat16(TensorList tensors) {
    return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
std::vector<Tensor> floating_complex_half_bfloat16(std::vector<std::vector<Tensor>>& mta_input_tensors) {
    return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, mta_input_tensors[0][0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(mta_input_tensors);
    });
}

template <template<class> class Op>
void floating_complex_half_bfloat16_(TensorList tensors) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        foreach_unary_op_<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
void floating_complex_half_bfloat16_(std::vector<std::vector<at::Tensor>>& mta_input_tensors) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, mta_input_tensors[0][0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        foreach_unary_op_<scalar_t, Op>(mta_input_tensors);
    });
}

template <template<class> class Op>
std::vector<Tensor> all_types_half_complex_bfloat16(TensorList tensors) {
    return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(ScalarType::Half, at::ScalarType::BFloat16, tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
std::vector<Tensor> all_types_half_complex_bfloat16(std::vector<std::vector<Tensor>>& mta_input_tensors) {
    return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(ScalarType::Half, at::ScalarType::BFloat16, mta_input_tensors[0][0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(mta_input_tensors);
    });
}

template <template<class> class Op>
void all_types_half_complex_bfloat16_(TensorList tensors) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        ScalarType::Half, at::ScalarType::BFloat16,
        tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
            foreach_unary_op_<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
void all_types_half_complex_bfloat16_(std::vector<std::vector<at::Tensor>>& mta_input_tensors) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        ScalarType::Half, at::ScalarType::BFloat16,
        mta_input_tensors[0][0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
            foreach_unary_op_<scalar_t, Op>(mta_input_tensors);
    });
}

template <template<class> class Op>
std::vector<Tensor> floating_half(TensorList tensors) {
    return AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Half,  tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
std::vector<Tensor> floating_half(std::vector<std::vector<Tensor>>& mta_input_tensors) {
    return AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Half,  mta_input_tensors[0][0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(mta_input_tensors);
    });
}

template <template<class> class Op>
void floating_half_(TensorList tensors) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensors[0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        foreach_unary_op_<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
void floating_half_(std::vector<std::vector<at::Tensor>>& mta_input_tensors) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(mta_input_tensors[0][0].scalar_type(), "foreach_unary_op_cuda_", [&]() {
        foreach_unary_op_<scalar_t, Op>(mta_input_tensors);
    });
}

template <template<class> class Op>
std::vector<Tensor> floating_half_bfloat16(TensorList tensors) {
    return AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16,  tensors[0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
std::vector<Tensor> floating_half_bfloat16(std::vector<std::vector<Tensor>>& mta_input_tensors) {
    return AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16,  mta_input_tensors[0][0].scalar_type(), "foreach_unary_op_cuda", [&]() {
        return foreach_unary_op<scalar_t, Op>(mta_input_tensors);
    });
}

template <template<class> class Op>
void floating_half_bfloat16_(TensorList tensors) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half, ScalarType::BFloat16,
        tensors[0].scalar_type(), "foreach_unary_op_cuda_",
        [&]() {
            foreach_unary_op_<scalar_t, Op>(tensors);
    });
}

template <template<class> class Op>
void floating_half_bfloat16_(std::vector<std::vector<at::Tensor>>& mta_input_tensors) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half, ScalarType::BFloat16,
        mta_input_tensors[0][0].scalar_type(), "foreach_unary_op_cuda_",
        [&]() {
            foreach_unary_op_<scalar_t, Op>(mta_input_tensors);
    });
}

// makes the functor
#define STD_FUNCTOR(op_name, functor_name)                           \
template<typename T>                                                 \
struct functor_name {                                                \
    __device__ T operator()(T t) const { return std::op_name(t); }   \
};                                                                   \

// TODO(mkozuki): DRY the sorting of output tensors
// TODO(mkozuki): Make sorting inplace?
// given a functor and a "dispatch function", creates the outplace and inplace operations
#define OP_CUSTOM_FUNCTOR(function, op_name, functor_name)                \
std::vector<Tensor> foreach_tensor_##op_name##_cuda(TensorList tensors) { \
    check_foreach_api_restrictions(tensors);                              \
    if (can_use_fast_route(tensors) && !has_integral_tensor(tensors, /* includeBool */ true)) {  \
        return function<functor_name>(tensors);                           \
    }                                                                     \
                                                                          \
    std::vector<Tensor> result;                                           \
    result.reserve(tensors.size());                                       \
    std::vector<size_t> index_map(tensors.size(), 0);                     \
                                                                          \
    size_t cur_index = 0;                                                 \
    OptionalDeviceGuard g;                                                \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors)) {                                          \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                                                  \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                                             \
            auto & mta_input_tensors = dtype_mtainputs.second.mta_input_tensors;                                               \
            if (!dtype_mtainputs.second.is_mta_compatible() || has_integral_tensor(mta_input_tensors[0], /* includeBool */ true)) { \
                for (const auto & t : at::native::foreach_tensor_##op_name##_slow(TensorList{mta_input_tensors[0]})) {         \
                    result.emplace_back(std::move(t));                                                                         \
                }                                                                                                              \
            } else {                                                                                                           \
                for (const auto & t : function<functor_name>(mta_input_tensors)) {                                             \
                    result.emplace_back(std::move(t));                                                                         \
                }                                                                                                              \
            }                                                                                                                  \
            const auto & indices = dtype_mtainputs.second.indices;        \
            for (const auto & i : c10::irange(indices.size())) {          \
                index_map[indices[i]] = cur_index + i;                    \
            }                                                             \
            cur_index += indices.size();                                  \
        }                                                                 \
    }                                                                     \
    std::vector<Tensor> sorted_result;                                    \
    sorted_result.reserve(tensors.size());                                \
    for (const auto & i : index_map) {                                    \
        sorted_result.emplace_back(std::move(result[i]));                 \
    }                                                                     \
    return sorted_result;                                                 \
}                                                                         \
void foreach_tensor_##op_name##_cuda_(TensorList tensors) {               \
    check_foreach_api_restrictions(tensors);                              \
    if (can_use_fast_route(tensors) && !has_integral_tensor(tensors, /* includeBool */ true)) {   \
        function##_<functor_name>(tensors);                                                       \
        return;                                                                                   \
    }                                                                                             \
    OptionalDeviceGuard g;                                                                        \
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors)) {             \
        g.reset_device(at::Device(dev_dtypemtainputs.first));                                     \
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {                                \
            auto & mta_input_tensors = dtype_mtainputs.second.mta_input_tensors;                  \
            if (!dtype_mtainputs.second.is_mta_compatible() || has_integral_tensor(mta_input_tensors[0], /* includeBool */ true)) {              \
                at::native::foreach_tensor_##op_name##_slow_(TensorList{mta_input_tensors[0]});   \
            } else {                                                                              \
                function##_<functor_name>(mta_input_tensors);                                     \
            }                                                                                     \
        }                                                                                         \
    }                                                                                             \
}

// creates a functor, outplace version, and inplace version.
#define OP(function, op_name, functor_name)         \
STD_FUNCTOR(op_name, functor_name);                 \
OP_CUSTOM_FUNCTOR(function, op_name, functor_name); \

OP(floating_half_bfloat16, erfc, Erfc);
OP(floating_half_bfloat16, expm1, Expm1);
OP(floating_half, lgamma, Lgamma);
OP(floating_half_bfloat16, trunc, Truncf);
OP(floating_half_bfloat16, floor, Floor);
OP(floating_half_bfloat16, ceil, Ceil);

OP(floating_complex_half_bfloat16, acos, Acos);
OP(floating_complex_half_bfloat16, asin, Asin);
OP(floating_complex_half_bfloat16, atan, Atan);
OP(floating_complex_half_bfloat16, cosh, Cosh);
OP(floating_complex_half_bfloat16, tan, Tan);
OP(floating_complex_half_bfloat16, sin, Sin);
OP(floating_complex_half_bfloat16, sinh, Sinh);

OP(floating_complex_half_bfloat16, exp, Exp);
OP(floating_complex_half_bfloat16, tanh, Tanh);
OP(floating_complex_half_bfloat16, log, Log);
OP(floating_complex_half_bfloat16, log10, Log10);
OP(floating_complex_half_bfloat16, log2, Log2);
OP(floating_complex_half_bfloat16, cos, Cos);
OP(floating_complex_half_bfloat16, sqrt, Sqrt);

OP(floating_half_bfloat16, log1p, Log1p);
OP(floating_half_bfloat16, erf, Erf);

//
// Special cases
// These functions must be special cased as they can't be written as std::functor_name in OP macro
//
template<typename T>
struct Sigmoid {
    T one = T(1);
    __device__ T operator()(T t) const { return (one / (one + std::exp(-t))); }
};

template<typename T>
struct Round {
    __device__ T operator()(T t) const { return std::nearbyint(t); }
};

template<typename T>
struct Trunc {
    __device__ T operator()(T t) const { return t - std::trunc(t); }
};

template<typename T>
struct Reciprocal {
    T one = T(1);
    __device__ T operator()(T t) const { return (one / t); }
};

OP_CUSTOM_FUNCTOR(floating_half_bfloat16, sigmoid, Sigmoid)
OP_CUSTOM_FUNCTOR(floating_half_bfloat16, round, Round)
OP_CUSTOM_FUNCTOR(floating_half_bfloat16, frac, Trunc)
OP_CUSTOM_FUNCTOR(floating_complex_half_bfloat16, reciprocal, Reciprocal)

// note(mkozuki): tensor dtype checks of `neg` kernels.
// Since `check_foreach_api_restrictions` don't require all the tensors to have the same dtype,
// I think it safer to check every single tensor's dtype inside negation kernels.
std::vector<Tensor> foreach_tensor_neg_cuda(TensorList tensors) {
    check_foreach_api_restrictions(tensors);
    for (const auto & t : tensors) {
        TORCH_CHECK(t.scalar_type() != kBool,
            "_foreach_neg: There is a bool tensor in the passed-in TensorList. "
            "Negation on a bool tensor is not supported. If you are trying to invert a mask, please use the `~`"
            "or `logical_not()` operator on the individual tensors instead.");
    }
    if (can_use_fast_route(tensors)) {
        return all_types_half_complex_bfloat16<std::negate>(tensors);
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
            const OptionalDeviceGuard device_guard(device_of(mta_input_tensors[0][0]));
            if (!dtype_mtainputs.second.is_mta_compatible()) {
                for (const auto & t : at::native::foreach_tensor_neg_slow(TensorList{mta_input_tensors[0]})) {
                    result.emplace_back(std::move(t));
                }
            } else {
                for (const auto & t : all_types_half_complex_bfloat16<std::negate>(mta_input_tensors)) {
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

void foreach_tensor_neg_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);
    for (const auto & t : tensors) {
        TORCH_CHECK(
            t.scalar_type() != kBool,
            "_foreach_neg: There is a bool tensor in the passed-in TensorList. "
            "Negation on a bool tensor is not supported. If you are trying to invert a mask, please use the `~`"
            "or `logical_not()` operator on the individual tensors instead.");
    }
    if (can_use_fast_route(tensors)) {
        all_types_half_complex_bfloat16_<std::negate>(tensors);
        return;
    }

    OptionalDeviceGuard g;
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors)) {
        g.reset_device(at::Device(dev_dtypemtainputs.first));
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {
            auto & mta_input_tensors = dtype_mtainputs.second.mta_input_tensors;
            if (!dtype_mtainputs.second.is_mta_compatible()) {
                at::native::foreach_tensor_neg_slow_(TensorList{mta_input_tensors[0]});
            } else {
                all_types_half_complex_bfloat16_<std::negate>(mta_input_tensors);
            }
        }
    }
}

// Abs have to go via slow path in case of a complex type.
// This is because foreach kernels can't return a different dtype than passed, while
// abs with complex inputs will produce float output.
template<typename T>
struct Abs {
    __device__ T operator()(T t) const { return std::abs(t); }
};

std::vector<Tensor> foreach_tensor_abs_cuda(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (can_use_fast_route(tensors)) {
        return all_types_complex_bfloat16_half_bool<Abs>(tensors);
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
            if (!dtype_mtainputs.second.is_mta_compatible() ||
                std::any_of(mta_input_tensors[0].begin(), mta_input_tensors[0].begin(),
                    [](const auto & t) { return at::isComplexType(t.scalar_type()); })
            ) {
                for (const auto & t : at::native::foreach_tensor_abs_slow(TensorList{mta_input_tensors[0]})) {
                    result.emplace_back(std::move(t));
                }
            } else {
                for (const auto & t : all_types_complex_bfloat16_half_bool<Abs>(mta_input_tensors)) {
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

void foreach_tensor_abs_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (can_use_fast_route(tensors)) {
        all_types_complex_bfloat16_half_bool_<Abs>(tensors);
        return;
    }

    OptionalDeviceGuard g;
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors)) {
        g.reset_device(at::Device(dev_dtypemtainputs.first));
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {
            auto & mta_input_tensors = dtype_mtainputs.second.mta_input_tensors;
            const bool has_complex = std::any_of(
                mta_input_tensors[0].begin(), mta_input_tensors[0].end(),
                [](const auto & t) { return at::isComplexType(t.scalar_type()); });
            if (!dtype_mtainputs.second.is_mta_compatible() || has_complex) {
                at::native::foreach_tensor_abs_slow_(TensorList{mta_input_tensors[0]});
            } else {
                all_types_complex_bfloat16_half_bool_<Abs>(mta_input_tensors);
            }
        }
    }
}

void foreach_tensor_zero_cuda_(TensorList tensors) {
    check_foreach_api_restrictions(tensors);

    if (can_use_fast_route(tensors)) {
        std::vector<std::vector<Tensor>> tensor_lists{tensors.vec()};
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
            ScalarType::Half,
            tensors[0].scalar_type(),
            "foreach_zero_cuda_",
            [&]() {
                multi_tensor_apply<1>(
                    tensor_lists,
                    ZeroFunctor<scalar_t, /* depth */ 1, /* r_args_depth */ 1, /* res_arg_index */ 0>()
                );
            }
        );
    }

    OptionalDeviceGuard g;
    for (auto & dev_dtypemtainputs : separate_tensors_by_device_and_dtype(tensors)) {
        g.reset_device(at::Device(dev_dtypemtainputs.first));
        for (auto & dtype_mtainputs : dev_dtypemtainputs.second) {
            auto & mta_input_tensors = dtype_mtainputs.second.mta_input_tensors;
            if (!dtype_mtainputs.second.is_mta_compatible()) {
                at::native::foreach_tensor_zero_slow_(TensorList{mta_input_tensors[0]});
            } else {
                AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
                    ScalarType::Half,
                    mta_input_tensors[0][0].scalar_type(),
                    "foreach_zero_cuda_",
                    [&]() {
                        multi_tensor_apply<1>(
                            mta_input_tensors,
                            ZeroFunctor<scalar_t, /* depth */ 1, /* r_args_depth */ 1, /* res_arg_index */ 0>()
                        );
                    }
                );
            }
        }
    }
}

}} // namespace at::native
