#include <ATen/ATen.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/cuda/stochastic_rounding.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

#define UNROLL 4

namespace at {
namespace native {

template <typename input_t, typename output_t, typename IndexType, int ADims, int VEC>
C10_LAUNCH_BOUNDS_2(256, 8)
__global__ void vectorized_stochastic_rounding_kernel(
    const at::cuda::detail::TensorInfo<input_t, IndexType> input,
    at::cuda::detail::TensorInfo<output_t, IndexType> output,
    const int64_t numel,
    std::pair<uint64_t, uint64_t> seed_and_offset) {
  static_assert(VEC <= 4, "Value of VEC must be in [2, 4].");
  using LoadT = memory::aligned_vector<input_t, VEC>;
  using WriteT = memory::aligned_vector<output_t, VEC>;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed_and_offset.first, tid, seed_and_offset.second, &state);

  for (IndexType i = tid * VEC; i < numel; i += gridDim.x * blockDim.x * VEC) {
    input_t src[VEC];
    LoadT *value = reinterpret_cast<LoadT*>(&src);
    *value = *reinterpret_cast<LoadT*>(&input.data[i]);

    float4 rand = curand_uniform4(&state);
    output_t ret[VEC];
    for (int ii = 0; ii < VEC; ii++) {
      ret[ii] = round_stochastically<output_t>(src[ii], (&rand.x)[ii]);
    }
    *(reinterpret_cast<WriteT*>(&output.data[i])) = *reinterpret_cast<WriteT*>(&ret[0]);
  }
}

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

  for (IndexType i = tid; i < numel; i += blockDim.x * gridDim.x) {
    float4 rand = curand_uniform4(&state);
    input_t src[UNROLL];

    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = i + blockDim.x * gridDim.x * ii;
      const IndexType offset = cuda::detail::IndexToOffset<input_t, IndexType, ADims>::get(li, input);
      src[ii] = input.data[offset];
    }
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = i + blockDim.x * gridDim.x * ii;
      const IndexType offset = cuda::detail::IndexToOffset<output_t, IndexType, ADims>::get(li, output);
      output.data[offset] = round_stochastically<output_t>(src[ii], (&rand.x)[ii]);
    }
    __syncthreads();
  }
}

template <typename scalar_t>
int get_vector_size(Tensor input, Tensor output) {
  int vec_size = 4;
  auto memory_format = input.suggest_memory_format();
  if (!input.is_contiguous(memory_format) || !output.is_contiguous(memory_format))
    vec_size = 1;
  else
    vec_size = memory::can_vectorize_up_to<scalar_t>((char*)input.data_ptr());
  bool can_vectorize = true;
  do {
    can_vectorize = input.numel() % vec_size == 0 && output.numel() % vec_size == 0;
    if (!can_vectorize)
      vec_size /= 2;
  } while (vec_size > 1 && !can_vectorize);
  return can_vectorize ? vec_size : 1;
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
    rng_engine_inputs = gen->philox_engine_inputs((numel + block * grid * UNROLL - 1) / (block * grid * UNROLL));
  }

  if (cuda::detail::canUse32BitIndexMath(input)) {
    AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "stochastic_rounding_cuda", [&] {
        auto input_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(input);
        auto output_info = cuda::detail::getTensorInfo<at::Half, unsigned int>(output);
        input_info.collapseDims();
        output_info.collapseDims();

        int vec_size = get_vector_size<scalar_t>(input, output);

        if (vec_size > 1) {
          switch (vec_size) {
            case 4:
              vectorized_stochastic_rounding_kernel<scalar_t, at::Half, unsigned int, 1, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                input_info, output_info, numel, rng_engine_inputs);
              break;
            case 2:
              vectorized_stochastic_rounding_kernel<scalar_t, at::Half, unsigned int, 1, 2><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                input_info, output_info, numel, rng_engine_inputs);
              break;
          }
        } else {
          switch (input_info.dims) {
            case 1:
              stochastic_rounding_kernel<scalar_t, at::Half, unsigned int, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                input_info, output_info, numel, rng_engine_inputs);
              break;
            default:
              stochastic_rounding_kernel<scalar_t, at::Half, unsigned int, -1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                input_info, output_info, numel, rng_engine_inputs);
          }
        }
      });
  } else {
    AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "stochastic_rounding_cuda", [&] {
        auto input_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(input);
        auto output_info = cuda::detail::getTensorInfo<at::Half, uint64_t>(output);
        input_info.collapseDims();
        output_info.collapseDims();

        int vec_size = get_vector_size<scalar_t>(input, output);

        if (vec_size > 1) {
          switch (vec_size) {
            case 4:
              vectorized_stochastic_rounding_kernel<scalar_t, at::Half, uint64_t, 1, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                input_info, output_info, numel, rng_engine_inputs);
              break;
            case 2:
              vectorized_stochastic_rounding_kernel<scalar_t, at::Half, uint64_t, 1, 2><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                input_info, output_info, numel, rng_engine_inputs);
              break;
          }
        } else {
          switch (input_info.dims) {
            case 1:
              stochastic_rounding_kernel<scalar_t, at::Half, uint64_t, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                input_info, output_info, numel, rng_engine_inputs);
              break;
            default:
              stochastic_rounding_kernel<scalar_t, at::Half, uint64_t, -1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                input_info, output_info, numel, rng_engine_inputs);
          }
        }
      });
  }

  return output;
}

} // namespace native
} // namespace at

#undef UNROLL
