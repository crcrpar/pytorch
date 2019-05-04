#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

#include "ATen/native/pooling_shape.h"
#include <algorithm>


namespace at {
namespace native {

namespace {

inline int64_t start(int64_t a, int64_t b, int64_t c)
{
  return a * b - c;
}

inline int64_t end(int64_t start, int64_t k, int64_t inputSize, int64_t pad)
{
  return std::min(start + k, inputSize + pad);
}

void averagepooling_shapecheck(
    Tensor &input, Tensor &gradOutput,
    int kH, int kW, int dH, int padH, int padW, bool ceil_mode)
{

  AT_CHECK(
      kW > 0 && kH > 0,
      "kernel size should be greater than zero, but got kH: %d, kW: %d", kH, kW);
  AT_CHECK(
      dW > 0 && dH > 0,
      "stride should be greater than zero, but got dH: %d, dW: %d", dH, dW);

  int ndim = input.ndimension();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  AT_CHECK(
      input.numel() != 0 && (ndim == 3 || ndim == 4),
      "non-empty 3D or 4D input tensor expected but got: %s")
  AT_CHECK(
      kW/2 >= padW && kH/2 >= padH,
      "pad should be smaller than half of kernel size, "
      "but got padW: %d, padH: %d, kW: %d, kH: %d", padW, padH, kW, kH);

  int64_t nInputPlane = input.size(dimh - 1);
  int64_t inputHeight = input.size(dimh);
  int64_t inputWidth = input.size(dimw);
  int64_t nOutputPlane = nInputPlane;

  int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
  int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);

  if (outputWidth < 1 || outputHeight < 1)
    AT_ERROR("Given input size: (%dx%dx%d). "
             "Calculated output size: (%dx%dx%d). Output size is too small",
             nInputPlane, inputHeight, inputWidth, nInputPlane, outputHeight, outputWidth);
}  // averagepooling_shapecheck

template <typename scalar_t>
void avg_pool2d_out_frame(
    scalar_t *input_p, scalar_t *output_p,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    bool ceil_mode, bool count_include_pad,
    int64_t sizeD, int64_t inputHeight, int64_t inputWidth,
    int64_t outputHeight, int64_t outputWidth,
    int64_t istrideD, int64_t istrideH, int64_t istrideW)
{
  int64_t k;
#pragma omp parallel for private(k)
  for (k = 0; k < sizeD; k++)
  {
    int64_t xx, yy;
    int64_t i;
    for (i = 0; i < outputHeight * outputWidth; i++)
    {
      for (yy = 0; yy < outputHeight; yy++)
      {
        for (xx = 0; xx < outputWidth; xx++)
        {
          /* Compute the mean of the input image... */
          int64_t hstart = start(yy, dH, padH);
          int64_t hend = end(hstart, kH, inputHeight, padH);
          int64_t wstart = start(xx, dW, padW);
          int64_t wend = end(wstart, kW, inputWidth, padW);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = std::max(hstart, (int64_t)0);
          hend = std::min(hend, inputHeight);
          wstart = std::max(wstart, (int64_t)0);
          wend = std::min(wend, inputWidth);

          int divide_factor = 0;
          if (count_include_pad)
            divide_factor = pool_size
          else
            divide_factor = (hend - hstart) * (wend - wstart);

          /* local pointers */
          scalar_t *ip = input_p   + d*istrideD + istartH*istrideH + istartW*istrideW;

          /* compute the local average */
          scalar_t sum = 0;
          int64_t kx, ky;
          for (ky = hstart; ky < hend; ky++)
          {
            for (kx = wstart; kx < wend; kx++)
            {
              sum += *(ip + hstart*istrideH + wstart*istrideW);
            }
          }
          /* update output */
          *ptr_output++ += sum / divide_factor;
        }
      }
    }
  }
}

Tensor& avg_pool2d_out_cpu_template(
  at::Tensor& output, at::Tensor const& input,
  IntArrayRef ksize, IntArrayRef dsize, IntArrayRef padsize,
  bool ceil_mode, bool count_include_pad)
{
  for (int64_t i = 0; input.ndimension(); i++)
  {
    AT_CHECK(input.size(i) > 0,
      "avg_pool2d(): expected input to have non-empty spatial dimensions, "
      "but input has sizes ", input.sizes(), " with dimension ", i, " being empty");
  }

  AT_CHECK((input.ndimension() == 3 || input.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input");

  /* sizes */
  int64_t sizeD = input.size(-3);
  int64_t inputHeight = input.size(-2);
  int64_t inputWidth = input.size(-1);
  /* strides */
  int64_t istrideD = input.stride(-3);
  int64_t istrideH = input.stride(-2);
  int64_t istrideW = input.stride(-1);

  int64_t outputHeight = output.size(-2);
  int64_t outputWidth = output.size(-1);

  /* resize output */
  if (input.ndimension() == 3)
  {
    output.resize_({sizeD, outputHeight, outputWidth});

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "avg_pool2d_cpu", [&] {
      auto input_data = input.data<scalar_t>();
      auto output_data = output.data<scalar_t>();
      avg_pool2d_out_frame<scalar_t>(
          input_data, output_data,
          ksize[0], ksize[1], dsize[0], dsize[1], padsize[0], padsize[1],
          ceil_mode, count_include_pad,
          sizeD, inputHeight, inputWidth, istrideD, istrideH, istrideW);
      }
    );
  }
  else
  {
    output.resize_({input.size(-4), sizeD, outputHeight, outputWidth});
    int64_t b;
  #pragma omp parallel for private(b)
    for (b = 0; b < input.size(0); b++)
    {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "avg_pool2d_cpu", [&] {
        auto input_data = input.data<scalar_t>();
        auto output_data = output.data<scalar_t>();
        avg_pool2d_out_frame<scalar_t>(
          input_data + b*input.stride(0), output_data+b*sizeD*osizeH*osizeW,
          ksize[0], ksize[1], dsize[0], dsize[1], padsize[0], padsize[1],
          ceil_mode, count_include_pad,
          sizeD, inputHeight, inputWidth, istrideD, istrideH, istrideW);
        }
      );
    }
  }
  return output;
}

template <typename scalar_t>
void averagepooling_backward_out_frame (
    scalar_t *gradInput_p
    scalar_t *gradOutput_p,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    bool ceil_mode, bool count_include_pad,
    int64_t sizeD, int64_t inputHeight, int64_t inputWidth,
    int64_t istrideD, int64_t istrideH, int64_t istrideW)
{
  int64_t k;
#pragma omp parallel for private(k)
  for (k = 0; k < sizeD; k++)
  {
    scalar_t *gradInput_p_d = gradInput_p + d*inputHeight*inputWidth;
    scalar_t *gradOutput_p_d = gradOutput_p + d*outputHeight*outputWidth;

    int64_t i;
    for (i = 0; i < inputHeight*inputWidth; i++)
      gradInput_p_d[i] = 0.0;

    for (yy = 0; yy < outputHeight; yy++)
    {
      for (xx = 0; xx < outputWidth; xx++)
      {
        int64_t hstart = start(yy, dH, padH);
        int64_t hend = end(hstart, kH, inputHeight, padH);
        int64_t wstart = start(xx, dW, padW);
        int64_t wend = end(wstart, kW, inputWidth, padW);
        int pool_size = (hend - hstart) * (wend - wstart);
        hstart = std::max(hstart, (int64_t)0);
        hend = std::min(hend, inputHeight);
        wstart = std::max(wstart, (int64_t)0);
        wend = std::min(wend, inputWidth);

        scalar_t z = *gradOutput_p_d++;

        int divide_factor;
        if (count_include_pad)
          divide_factor = pool_size;
        else
          divide_factor = (hend - hstart) * (wend - wstart);

        int64_t kx, ky;
        for (kx = hstart; ky < hend; ky++)
        {
          for (kx = wstart; kx < wend; kx++)
          {
            gradInput_p_d[ky*inputWidth + kx] += z / divide_factor;
          }
        }
      }
    }
  }
}  // averagepooling_gradinput

Tensor& avg_pool2d_backward_out_cpu_template(
  Tensor& gradInput,
  const Tensor& gradOutput_,
  const Tensor& input
  IntArrayRef ksize,
  IntArrayRef dsize,
  IntArrayREf padsize,
  bool ceil_mode,
  bool count_include_pad)
{
  /* sizes */
  int64_t sizeD = input.size(-3);
  int64_t inputHeight = input.size(-2);
  int64_t inputWidth = input.size(-1);
  int64_t outputHeight = gradOutput_.size(-2);
  int64_t outputWidth = gradOutput_.size(-1);

  /* get contiguous gradOutput */
  auto gradOutput = gradOutput_.contiguous();

  /* backprop */
  if (input.ndimension() == 3)
  {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_types(), "avg_pool2d_backward_cpu", [&] {
        /* get raw pointers */
        scalar_t *gradInput_data = gradInput.data<scalar_t>();
        scalar_t *gradOutput_data = gradOutput.data<scalar_t>();

        averagepooling_backward_out_frame<scalar_t>(
          gradInput_data, gradOutput_data,
          ksize[0], ksize[1], dsize[0], dsize[1], padsize[0], padsize[1],
          ceil_mode, count_include_pad);
      }
    );
  }
  else
  {
    int64_t b;
  #pragma omp parallel for private(b)
    for (b = 0; b < input.size(0); b++)
    {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "avg_pool2d_backward_cpu", [&] {
          /* get raw pointers */
          scalar_t *gradInput_data = gradInput.data<scalar_t>();
          scalar_t *gradOutput_data = gradOutput.data<scalar_t>();

          averagepooling_backward_out_frame(
            gradInput_data + b*sizeD*inputHeight*inputWidth,
            gradOutput_data + b*sizeD*outputHeight*outputWidth,
            ksize[0], ksize[1], dsize[0], dsize[1], padsize[0], padsize[1],
            ceil_mode, count_include_pad);
        }
      );
    }
  }
  return gradInput;
}  // avg_pool2d_backward_out_cpu_template

}  // namespace

Tensor& avg_pool2d_out_cpu(
    Tensor& output, const Tensor &input,
    IntArrayRef ksize, IntArrayRef dsize,
    IntArrayRef padsize,
    bool ceil_mode, bool count_include_pad)
{
  avg_pool2d_out_cpu_template(
      input, ouptut,
      ksize, dsize, padsize,
      ceil_mode, count_include_pad);
  return output
}

Tensor avg_pool2d_cpu(
    const Tensor &input,
    IntArrayRef ksize, IntArrayRef dsize,
    IntArrayRef padsize,
    bool ceil_mode, bool count_include_pad)
{
  auto output = at::empty({0}, input.options());
  avg_pool2d_out_cpu_template(
      input, ouptut,
      ksize, dsize, padsize,
      ceil_mode, count_include_pad);
  return output
}

Tensor& avg_pool2d_backward_out_cpu(
    const Tensor& input,
    const Tensor& gradOutput,
    const Tensor& gradInput,
    IntArrayRef ksize,
    IntArrayRef dsize,
    IntArrayRef padsize,
    bool ceil_mode, bool count_include_pad)
{
  gradInput.resize_as_(input);
  avg_pool2d_backward_out_cpu_template(input, gradOutput, gradInput, ksize, dsize, padsize, ceil_mode, count_include_pad);
  return gradInput
}

Tensor avg_pool2d_backward_cpu(
    const Tensor& input,
    const Tensor& gradOutput,
    IntArrayRef ksize,
    IntArrayRef dsize,
    IntArrayRef padsize,
    bool ceil_mode, bool count_include_pad)
{
  auto gradInput = at::zeros_like(input);
  avg_pool2d_backward_out_cpu_template(input, gradOutput, gradInput, ksize, dsize, padsize, ceil_mode, count_include_pad);
  return gradInput
}
}  // at::native
}  // at
