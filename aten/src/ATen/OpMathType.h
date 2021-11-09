#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

namespace at {

// For FP16 or BFloat16 inputs, ops should perform internal math in FP32.
template<typename scalar_t> struct OpMathType { using type = scalar_t; };
template<> struct OpMathType<at::Half> { using type = float; };
template<> struct OpMathType<at::BFloat16> { using type = float; };

template<typename T>
using opmath_type = typename OpMathType<T>::type;

ScalarType toOpmathType(const ScalarType type) {
  if (type == at::ScalarType::BFloat16 || type == at::ScalarType::Half) {
    return at::ScalarType::Float;
  } else {
    return type;
  }
}

} // namespace at
