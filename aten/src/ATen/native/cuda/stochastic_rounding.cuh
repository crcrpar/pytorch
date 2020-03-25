#pragma once
#include <ATen/ATen.h>
#include <ATen/CUDAGenerator.h>

#include <cuda.h>
#include <cuda_fp16.h>

#include <math.h>


// 2^-10 is the step for normal FP16 numbers.
// 2^-24 is the unit in the last place (ULP)/precision limitation.
// 24 is **NOT** related to the number of mantissa bits of single precision format.
// ref:
//   - https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_decimal_values_in_[0,_1]
//   - https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_decimal_values_in_[1,_2048]
__device__ const float TWO_10 = 0.0009765625;
__device__ const float TWO_24 = 0.000000059604644775390625;


namespace at {
namespace native {

template <typename T>
__device__ __forceinline__ T _maybe__upcast(__half x) {
  return T(__half2float(x));
}

template <>
__device__ __forceinline__ __half _maybe__upcast(__half x) {
  return x;
}

__device__ __forceinline__ float get_delta_fp16(float x) {
  int exponent;
  frexpf(x, &exponent);
  exponent -= 1;
  if (exponent >= -14) {
    return TWO_10 * powf(2.0f, exponent);
  } else {
    return TWO_24;
  }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t round_stochastically(float x, float random_value) {
  if (x == 0.0f) {
    return scalar_t(0.0);
  }

  float delta = get_delta_fp16(x);
  float value;

  if (x < 0.0f) {
    value = x - random_value * delta;
  } else {
    value = x + random_value * delta;
  }

  return _maybe__upcast<scalar_t>(__float2half_rz(value));
}

} // namespace native
} // namespace at
