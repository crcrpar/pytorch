#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <c10/util/Exception.h>
#include <THC/THCGeneral.h>
#include <THC/THCNumerics.cuh>

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace at {
namespace native {
namespace {

__device__ inline int start_index(int a, int b, int c) {
  return (int)std::floor((float)(a * c) / b);
}

__device__ inline int end_index(int a, int b, int c) {
  return (int)std::ceil((float)((a + 1) * c) / b);
}

// CUDA: grid stride looping
// #define CUDA_KERNEL_LOOP(i, n) \
//   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename Dtype, typename Acctype>
__global__ void avg_pool2d_out_frame_cuda(
    T *input, T *output,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    bool ceil_mode, bool count_include_end,
    int64_t sizeD, int64_t inputHeight, int64_t inputWidth,
    int64_t outputHeight, int64_t outputWidth,
    int64_t istrideD, int64_t istrideH, int64_t istrideW)
{
  int oh, ow;

  int o_plane = blockIdx.x;
  int i_plane = o_plane;

  output = output + o_plane*outputHeight*outputWidth;
  input = input + i_plane*istrideD;

  int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
  int oendH = outputHeight;
  int ostepH = blockDim.y * gridDim.y;

  int
}
}  // namespace
}  // at::native
}  // at
