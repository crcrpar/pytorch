#pragma once

#include <cmath>
#include <utility>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <ATen/Utils.h>
#include <ATen/Generator.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>


// 2^-10 is the step for normal FP16 numbers.
// 2^-24 is the unit in the last place (ULP)/precision limitation.
// 24 is **NOT** related to the number of mantissa bits of single precision format.
__device__ const float TWO_10 = 0.0009765625;
__device__ const float TWO_24 = 0.000000059604644775390625;


template<typename out_type, typename rounded_type>
__device__ __forceinline__ T maybe_upcast<>(__half x) {
  return T(static_cast<float>(x));
}

template<typename out_type>
__device__ __forceinline__ __half maybe_upcast<out_type, __half>(__half x) {
  return T(__half2float(x));
}

template<>
__device__ __forceinline__ __half maybe_upcast<float, __half>(__half x) {
  return __half2float(x);
}


template<typename out_type, typename in_type>
__device__ __forceinline__ out_type get_delta_fp16(in_type x) {
  int exponent;
  frexp(maybe_upcast<out_type>(x), &exponent);
  exponent -= 1;
  if (exponent >= -14) {
    return out_type(TWO_10 * std::pow(2, exponent));
  } else {
    return out_type(TWO_24);
  }
}


template<typename out_type, typename in_type>
struct rounder {
  static_assert(std::is_same<out_type, __half>::value && std::is_same<in_type, float>::value, "rounder only supports out_type=__half and in_type=float");
}

template <>
struct rounder<__half, float> {
  __device__ __forceinline__ __half operator() (float x) {
    return __float2half_rz(x);
  }
}


// Natalia Magic
template<typename out_type, typename in_type, typename round_to=at::Half>
struct round_stochastically {
  static_assert(std::is_same<round_to, at::Half>::value, "round_stochastically only supports round_to=at::Half");
};

template<typename out_type, typename in_type>
struct round_stochastically<out_type, in_type, at::Half> {
  __device__ __forceinline__ out_type operator() (in_type x, float random_value) {
    if (x == 0.0) {
      return out_type(0.0);
    }
    const rounder<__half, float> rounder;
    const float delta = get_delta_fp16(x);
    float val;
    if (x < 0.0) {
      val = x - random_value * delta;
    } else {
      val = x + random_value * delta;
    }
  }
  const __half rounded = rounder(val);
  return maybe_upcast<out_type>(rounded);
}
