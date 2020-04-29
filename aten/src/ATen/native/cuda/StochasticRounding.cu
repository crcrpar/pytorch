#include <ATen/ATen.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/cuda/stochastic_rounding.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

namespace at {
namespace native {

template <typename input_t, typename output_t, typename IndexType, int ADims>
C10_LAUNCH_BOUNDS_2(256, 8)
__global__ void stochastic_rounding_kernel(
    const at::cuda::detail::TensorInfo<input_t, IndexType> input,
    at::cuda::detail::TensorInfo<output_t, IndexType> output,
    const int64_t numel,
    std::pair<uint64_t, uint64_t> seed_and_offset) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed_and_offset.first, tid, seed_and_offset.second, &state);

  for (int64_t i = tid; i < numel; i += blockDim.x * gridDim.x) {
    float inp = static_cast<float>(input.data[i]);
    output.data[i] = round_stochastically<output_t>(inp, curand_uniform(&state));
  }
}

Tensor stochastic_rounding_cuda(const Tensor& input, c10::optional<Generator> gen_) {

  if (input.scalar_type() == kHalf) {
    return input;
  }

  Tensor output = at::empty_like(input, input.options().dtype(kHalf), input.suggest_memory_format());
  const int64_t numel = input.numel();
  if (numel == 0) {
    return output;
  }

  const int block = 256;
  const int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block;
  unsigned int grid = (numel + block - 1) / block;
  grid = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid);

  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs((numel + block * grid - 1) / (block * grid));
  }

  if (cuda::detail::canUse32BitIndexMath(input)) {
    AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "stochastic_rounding_cuda", [&] {
        auto input_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(input);
        auto output_info = cuda::detail::getTensorInfo<at::Half, unsigned int>(output);
        input_info.collapseDims();
        output_info.collapseDims();

        switch (input_info.dims) {
          case 1:
            stochastic_rounding_kernel<scalar_t, at::Half, unsigned int, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              input_info, output_info, numel, rng_engine_inputs);
            break;
          default:
            stochastic_rounding_kernel<scalar_t, at::Half, unsigned int, -1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              input_info, output_info, numel, rng_engine_inputs);
        }
      });
  } else {
    AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "stochastic_rounding_cuda", [&] {
        auto input_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(input);
        auto output_info = cuda::detail::getTensorInfo<at::Half, uint64_t>(output);
        input_info.collapseDims();
        output_info.collapseDims();

        switch (input_info.dims) {
          case 1:
            stochastic_rounding_kernel<scalar_t, at::Half, uint64_t, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              input_info, output_info, numel, rng_engine_inputs);
            break;
          default:
            stochastic_rounding_kernel<scalar_t, at::Half, uint64_t, -1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              input_info, output_info, numel, rng_engine_inputs);
        }
      });
  }

  return output;
}

} // namespace native
} // namespace at
