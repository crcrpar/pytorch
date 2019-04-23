#include "ATen/ATen/h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/NativeFunctions.h"
#include "ATen/TensorUtils.h"
#include "ATen/Utils.h"
#include "c10/util/Exception.h"
#include <THC/THCGeneral.h>
#include "THC/THCNumerics.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>

#define START_IND(a,b,c) (int)std::floor((float)(a * c) / b)
#define END_IND(a,b,c) (int)std::ceil((float)((a + 1) * c) / b)
// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0

#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit


namespace at {
namespace native {

namespace {

  // 5d tensor B x D x T x H x W
  // All kernels view batch dim B and dim D as collapsed.

  /*
   * Description:
   *    this function adaptively average pools an input 5D tensor along dimensions 2, 3, and 4
   *
   *    gridDim.y blocks work together on a single 2D output plane specified by
   *    (blockIdx.x + offsetZ).
   */
   template <typename T>
  __global__ void adaptiveaveragepool3d(T *input, T *output,
                              int isizeT, int isizeH, int isizeW,
                              int osizeT, int osizeH, int osizeW,
                              int64_t istrideD, int64_t istrideH, int64_t istrideW,
                              int64_t offsetZ)
  {
    // iterates on output pixels
    int ot, oh, ow;

    // compute offsets based on thread/block ID
    int ostartH = blockIdx.y * blockDim.y + threadIdx.y;
    int oendH   = osizeH;
    int ostepH  = gridDim.y * blockDim.y;
    int ostartW = threadIdx.x;
    int oendW   = osizeW;
    int ostepW  = blockDim.x;

    // select output plane
    int64_t o_plane = blockIdx.x + offsetZ;
    ot = o_plane % osizeT;     // output frame/time
    int d = o_plane / osizeT;  // slice/feature

    // input frame/time ramge is fixed.
    int istartT = START_IND(ot, osizeT, isizeT);
    int iendT = END_IND(ot, osizeT, isizeT);
    int kT = iendT - istartT;

    // input offset by slice/feature and earliest relevant frame/time
    T *input_dt = input + d*istrideD + istartT*istrideT;
    // output offset by slice/feature and frame/time
    T *output_dt = output + o_plane*osizeH*osizeW;

    // For all output pixels...
    for(oh = ostartH; oh < oendH; oh += ostepH) {

      int istartH = START_IND(oh, osizeH, isizeH);
      int iendH   = END_IND(oh, osizeH, isizeH);
      int kH = iendH - istartH;

      for(ow = ostartW; ow < oendW; ow += ostepW) {

        int istartW = START_IND(ow, osizeW, isizeW);
        int iendW   = END_IND(ow, osizeW, isizeW);
        int kW = iendW - istartW;

        // Compute the average pooling from corresponding input pixels
        T *ptr_input = input_dt + istartH*istrideH + istartW*istrideW;
        T *ptr_output = output_dt + oh*osizeW + ow;
        T sum = ScalarConvert<int, T>::to(0);

        int it, ih, iw;
        for(it = 0; it < kT; ++it) {
          for(ih = 0; ih < kH; ++ih) {
            for(iw = 0; iw < kW; ++iw) {
              T val = ptr_input[ih*istrideH + iw*istrideW];
              sum += val;
            }
          }
          ptr_input += istrideT;   // next input frame
        }
        // Update output
        *ptr_output = sum / kT / kH / kW;
      }
    }
  }

  /*
   * Description:
   *    This function computes the gradInput from gradOutput.
   *
   *    gridDim.y blocks work together on a single 2D input plane specified by
   *    (blockIdx.x + offsetZ).
   */
   template <typename T>
  __global__ void adaptiveaveragegradinput3d(T *gradInput, T *gradOutput,
                              int isizeT, int isizeH, int isizeW,
                              int osizeT, int osizeH, int osizeW,
                              int64_t offsetZ)
  {
    // iterators on input pixels
    int it, ih, iw;

    // compute offsets based on thread/block ID
    int istartH = blockIdx.y * blockDim.y + threadIdx.y;
    int iendH   = isizeH;
    int istepH  = gridDim.y * blockDim.y;
    int istartW = threadIdx.x;
    int iendW   = isizeW;
    int istepW  = blockDim.x;

    // select input plane
    int64_t i_plane = blockIdx.x + offsetZ;
    it = i_plane % isizeT;        // output frame/time
    int d = i_plane / isizeT;     // slice/feature

    // output frame/time ramge is fixed.
    int ostartT = START_IND(it, isizeT, osizeT);
    int oendT   = END_IND(it, isizeT, osizeT);

    // gradInput offset by slice/feature and frame/time
    T *gradInput_dt = gradInput + i_plane*isizeH*isizeW;
    // gradOutput offset by slice/feature and earliest relevant frame/time
    T *gradOutput_dt = gradOutput + (d*osizeT + ostartT)*osizeH*osizeW;

    // For all input pixels...
    for(ih = istartH; ih < iendH; ih += istepH) {

      int ostartH = START_IND(ih, isizeH, osizeH);
      int oendH   = END_IND(ih, isizeH, osizeH);

      for(iw = istartW; iw < iendW; iw += istepW) {

        int ostartW = START_IND(iw, isizeW, osizeW);
        int oendW   = END_IND(iw, isizeW, osizeW);

        // Compute the gradients from corresponding output pixels
        T *ptr_gradInput = gradInput_dt + ih*isizeW + iw;
        T *ptr_gradOutput = gradOutput_dt;

        // for all relevant output pixels
        int ot, oh, ow;
        for(ot = ostartT; ot < oendT; ++ot) {
          int kT = END_IND(ot, osizeT, isizeT) - START_IND(ot, osizeT, isizeT);
          for(oh = ostartH; oh < oendH; ++oh) {
            int kH = END_IND(oh, osizeH, isizeH) - START_IND(oh, osizeH, isizeH);
            for(ow = ostartW; ow < oendW; ++ow) {
              int kW = END_IND(ow, osizeW, isizeW) - START_IND(ow, osizeW, isizeW);
              T grad_delta = ptr_gradOutput[oh*osizeW + ow] / kW / kH / kT;
              *ptr_gradInput += grad_delta;
            }
          }
          ptr_gradOutput += osizeH*osizeW;   // next output frame
        }
      }
    }
  }

  /*
   * Description:
   *    This function computes the gradInput from gradOutput without assuming
   *    dependencies between input pixels and output pixels.
   *
   *    gridDim.y blocks work together on a single 2D output plane specified by
   *    (blockIdx.x + offsetZ).
   *
   *    (uses atomic add)
   */
   template <typename T>
  __global__ void atomicadaptiveaveragegradinput3d(
    T *gradInput, T *gradOutput,
    int isizeT, int isizeH, int isizeW,
    int osizeT, int osizeH, int osizeW,
    int64_t offsetZ)
  {
    // iterators on output pixels
    int ot, oh, ow;

    // compute offsets based on thread/block ID
    int ostartH = blockIdx.y * blockDim.y + threadIdx.y;
    int oendH = osizeH;
    int ostepH = gridDim.y * blockDim.y;
    int ostartH = threadIdx.x;
    int oendW = osizeW;
    int ostepW = blockDim.x;

    // select output plane
    int64_t o_plane = blockIdx.x + offsetZ;
    ot = o_plane % osizeT;  // output frame/time
    int d = o_plane / osizeT;  // output slice/feature

    // input frame/time range is fixed.
    int istarT = START_IND(ot, osizeT, isizeT);
    int iendT = END_INT(ot, osizeT, isizeT);
    int kT = iendT - istarT;

    // gradInput offset by slice/feature and earliest relevant frame/time
    T *gradInput_nt = gradInput + (d * isizeT + istartT) * isizeH * isizeW;
    // gradOutput offset by slice/feature and frame/time
    T *gradOutput_nt = gradOutput + o_plane * osizeH * osizeW;

    // For all output pixels...
    for (oh = ostartH; oh < oendH; oh += ostepH) {
      int istartH = START_IND(oh, osizeH, isizeH);
      int iendH = END_INT(oh, osizeH, isizeH);
      int kH = iendH - istartH;

      for (ow = ostartW; ow < oendW; ow += ostepW) {
        int istartW = START_IND(ow, osizeW, isizeW);
        int iendW = END_INT(ow, osizeW, isizeW);
        int kW = iendW - istartW;


        // Compute the gradients from corresponding input pixels
        T *ptr_gradInput = gradInput_nt + istartH * isizeW + istartW;
        T *ptr_gradOutput = gradOutput_nt + oh * osizeW + ow;
        T grad_delta = *ptr_gradOutput / kT / kW / kW;

        int it, ih, iw;
        for (it = 0; it < kT; ++it) {
          for (ih = 0; ih < kH; ++ih) {
            for (iw = 0; iw < kW; ++iw) {
              atomicAdd(&(ptr_gradInput(ih * isizeW + iw)), grad_delta);
            }
          }
          ptr_gradInput += isizeH * isizeW;  // next input frame
        }
      }
    }
  }

  // 5D tensor B x D x T x H x W

  void adaptive_avg_pool3d_out_cuda_template(
      Tensor& output,
      const Tensor& input,
      IntArrayRef output_size)
  {
    TensorArg(input_args( input, "input", 1 ),
              output_arg( output, "output", 2 ));
    checkAllSameGPU("cudnn_adaptive_avg_pool3d", {input_arg, output_arg});

    for (int64_t i = 0; i < input.ndimensions(); i++) {
      AT_CHECK(input.size(i) > 0,
        "adaptive_avg_pool3d(): expected input to have non-empty spatial dimensions, "
        "but input has sizes ", input.sizes(), " with dimension ", i, " being empty");
    }

    AT_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");
    Tensor input_ = input;
    int64_t grid_x = input.size(-3);
    if (input.ndimension() == 5) {
      input_ = input.contiguous();
      grid_x *= input_.size(-4);
    }
    int64_t sizeD = input_.size(-4);
    int64_t isizeT = input_.size(-3);
    int64_t isizeH = input_.size(-2);
    int64_t isizeW = input_.size(-1);

    int64_t istrideD = input_.stride(-4);
    int64_t istrideT = input_.stride(-3);
    int64_t istrideH = input_.stride(-2);
    int64_t istrideW = input_.stride(-1);

    int64_t osizeT = output_size[0];
    int64_t osizeH = output_size[1];
    int64_t osizeW = output_size[2];

    if (input.ndmension() == 5) {
      output.resize_({input.size(-5), sizeD, osizeT, osizeH, osizeW});
    } else {
      output.resize_({sizeD, osizeT, osizeH, osizeW});
    }
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_.scalar_type(), "adaptive_avg_pool3d_cuda", [&] {
          scalar_t *input_data = input_.data<scalar_t>();
          scalar_t *output_data = output.data<scalar_t>();

          // cuda blocks & threads
          int blocksT = std::max<int64_t>(static_cast<int>(16L / sizeD), 1);
          dim3 blocks(grid_x, blocksT);
          dim3 threads(32, 8);

          // run average pool kernel
          adaptiveaveragepool3d <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>> (
            input_data, output_data,
            isizeT, isizeH, isizeW, osizeT, osizeH, osizeW,
            istrideD, istrideT, istrideH, istrideW);
        }
     );
    THCudaCheck(cudaGetLastError());
  }

  void adaptive_avg_pool3d_backward_out_cuda_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input)
  {
    TensorArg grad_input_arg{ gradInput, "gradInput", 1 },
              grad_output_arg{ gradOutput_, "gradOutput_", 2 },
              input_arg{ input, "input", 3 };
    checkAllSameGPU("cudnn_adaptive_avg_pool3d_out",
                    {gradInput_arg, grad_output_arg, input_arg});

    // bool atomic = (isizeW % osizeW != 0) || (isizeH != 0);
    bool atomic = true;  // suboptimal, but without atomic it doesn't pass the tests

    Tensor gradOutput = gradOutput_.contiguous();

    int64_t sizeD = input.size(-4);
    int64_t isizeT = input.size(-3);
    int64_t isizeH = input.size(-2);
    int64_t isizeW = input.size(-1);

    int64_t osizeT = gradOutput.size(-3);
    int64_t osizeH = gradOutput.size(-2);
    int64_t osizeW = gradOutput.size(-1);

    int64_t grid_x = sizeD;
    if (input.ndimension() == 5) {
      grid_x *= input.size(-5);
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "adaptive_avg_pool3d_backward_cuda", [&] {
        scalar_t *gradOutput_data = gradOutput.data<scalar_t>();
        scalar_t *gradInput_data = gradInut.data<scalar_t>();

        // cuda blocks & threads
        int blocksT = std::max(static_cast<int>(16L / sizeD), 1);
        dim3 blocks(grid_x, blocksT);
        dim3 threads(32, 8);

        if (atomic) {
          atomicadaptiveaveragegradinput3d <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>> (
            gradInput_data, gradOutput_data, isizeT, isizeH, isizeW, osizeT, osizeH, osizeW);
        } else {
          adaptiveaveragegradinput3d <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>> (
            gradInput_data, gradOutput_data, isizeT, isizeH, isizeW, osizeT, osizeH, osizeW);
        }
      }
    );
    THCudaCheck(cudaGetLastError());
  }

} // namespace

  Tensor& adaptive_avg_pool3d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size)
  {
    adaptive_avg_pool3d_out_cuda_template(output, input, output_size);
    return output;
  }

  Tensor adaptive_avg_pool3d_cuda(
      at::Tensor const& input, IntArrayRef output_size)
  {
    auto output = at::empty({0}, input.options());
    adaptive_avg_pool3d_out_cuda_template(output, input, output_size);
    return output;
  }

  Tensor& adaptive_avg_pool3d_backward_out_cuda(
      Tensor& gradInput, const Tensor& gradOutput, const Tensor& input)
  {
    gradInput.resize_as_(input);
    adaptive_avg_pool3d_backward_out_cuda_template(gradInput, gradOutput, input);
    return gradInput;
  }

  Tensor adaptive_avg_pool3d_backward_cuda(const Tensor& gradOutput, const Tensor input)
  {
    auto gradInput = at::zeros_like(input);
    adaptive_avg_pool3d_backward_out_cuda_template(gradInput, gradOutput, input);
    return gradInput;
  }

} // at::native
} // at

#undef CUDA_MAX_THREADS
#undef START_IND
#undef END_IND
