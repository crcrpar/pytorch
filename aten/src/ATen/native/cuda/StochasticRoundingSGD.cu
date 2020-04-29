#include <ATen/ATen.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/cuda/stochastic_rounding.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

#define UNROLL 4

namespace at {
namespace native {

template <typename scalar_t, typename IndexType, int ADims, int VEC>
C10_LAUNCH_BOUNDS_2(256, 8)
__global__ void vectorized_stochastic_rounding_sgd_step_kernel(
    at::cuda::detail::TensorInfo<scalar_t, IndexType> weights,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> gradients,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> momentum_buffer,
    float* inv_scale, float* found_inf,
    float weight_decay, float momentum, float dampening, float lr,
    bool nesterov, bool first_run, int numel, std::pair<uint64_t, uint64_t> seeds) {
  static_assert(VEC <= 4, "Value of VEC must be in [2, 4].");

  if (*found_inf) return;

  using LoadT = memory::aligned_vector<scalar_t, VEC>;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, tid, seeds.second, &state);

  for (IndexType i = tid * VEC; i < numel; i += gridDim.x * blockDim.x * VEC) {
    scalar_t params[VEC], grads[VEC], mom_buf[VEC];
    LoadT *p_value = reinterpret_cast<LoadT*>(&params);
    *p_value = *reinterpret_cast<LoadT*>(&weights.data[i]);
    LoadT *g_value = reinterpret_cast<LoadT*>(&grads);
    *g_value = *reinterpret_cast<LoadT*>(&gradients.data[i]);
    LoadT *m_value = reinterpret_cast<LoadT*>(&mom_buf);
    *m_value = *reinterpret_cast<LoadT*>(&momentum_buffer.data[i]);

    for (int ii = 0; ii < VEC; ii++) {
      float weight = static_cast<float>(params[ii]);
      float gradient = static_cast<float>(grads[ii]) * (*inv_scale);
      float velocity = static_cast<float>(mom_buf[ii]);
      float4 random_values = curand_uniform4(&state);
      if (weight_decay != 0.0f)
        gradient += weight_decay * weight;

      if (momentum != 0.0f) {
        if (!first_run)
          velocity = velocity * momentum + (1.0f - dampening) * gradient;
        else
          velocity = gradient;

        if (nesterov)
          gradient += momentum * velocity;
        else
          gradient = velocity;
      }

      weight -= lr * gradient;
      params[ii] = round_stochastically<scalar_t>(weight, random_values.x);
      if (momentum != 0.0f)
        mom_buf[ii] = round_stochastically<scalar_t>(velocity, random_values.y);
    }
    *(reinterpret_cast<LoadT*>(&weights.data[i])) = *reinterpret_cast<LoadT*>(&params[0]);
    if (momentum != 0.0f)
      *(reinterpret_cast<LoadT*>(&momentum_buffer.data[i])) = *reinterpret_cast<LoadT*>(&mom_buf[0]);
    __syncthreads();
  }
}

// SGD update math with Stochastic Rounding
template <typename scalar_t, typename IndexType, int ADims>
C10_LAUNCH_BOUNDS_2(256, 8)
__global__ void stochastic_rounding_sgd_step_kernel(
    at::cuda::detail::TensorInfo<scalar_t, IndexType> weights,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> gradients,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> momentum_buffer,
    float* inv_scale, float* found_inf,
    float weight_decay, float momentum, float dampening, float lr,
    bool nesterov, bool first_run, int numel, std::pair<uint64_t, uint64_t> seeds) {

  if (*found_inf) return;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, tid, seeds.second, &state);

  const IndexType unit = blockDim.x * gridDim.x * UNROLL;
  const IndexType rounded_size = (numel + unit - 1) / unit * unit;
  for (IndexType i = tid; i < rounded_size; i += blockDim.x * gridDim.x * UNROLL) {

    float params[UNROLL], grads[UNROLL], mom_buf[UNROLL];

    // load
    for (int ii = 0; ii < UNROLL; ii++) {
      const IndexType li = i + blockDim.x * gridDim.x * ii;
      if (li < numel) {
        IndexType offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(li, weights);
        params[ii] = static_cast<float>(weights.data[offset]);
        offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(li, gradients);
        grads[ii] = static_cast<float>(gradients.data[offset]) * (*inv_scale);
        offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(li, momentum_buffer);
        mom_buf[ii] = static_cast<float>(momentum_buffer.data[offset]);
      }
    }

    // update & store
    for (int ii = 0; ii < UNROLL; ii++) {
      const IndexType li = i + blockDim.x * gridDim.x * ii;
      if (li < numel) {
        if (weight_decay != 0.0f)
          grads[ii] += weight_decay * params[ii];

        if (momentum != 0.0f) {
          if (!first_run)
            mom_buf[ii] = mom_buf[ii] * momentum + (1.0f - dampening) * grads[ii];
          else
            mom_buf[ii] = grads[ii];

          if (nesterov)
            grads[ii] += momentum * mom_buf[ii];
          else
            grads[ii] = mom_buf[ii];
        }

        params[ii] -= lr * grads[ii];

        float4 random_values = curand_uniform4(&state);
        IndexType offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(li, weights);
        weights.data[offset] = round_stochastically<scalar_t>(params[ii], random_values.x);
        if (momentum != 0.0f) {
          offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(li, momentum_buffer);
          momentum_buffer.data[offset] = round_stochastically<scalar_t>(mom_buf[ii], random_values.y);
        }
      }
    }
    __syncthreads();
  }
}

template <typename scalar_t>
int get_vector_size(const Tensor& param, const Tensor& grad, const Tensor& momentum_buffer) {
  int vec_size = 4;
  auto memory_format = param.suggest_memory_format();
  if (!param.is_contiguous(memory_format) || !grad.is_contiguous(memory_format) || !momentum_buffer.is_contiguous(memory_format))
    return 1;
  vec_size = memory::can_vectorize_up_to<scalar_t>((char*)param.data_ptr());
  bool can_vectorize = true;
  do {
    can_vectorize = param.numel() % vec_size == 0 && grad.numel() % vec_size == 0 && momentum_buffer.numel() % vec_size == 0;
    if (!can_vectorize)
      vec_size /= 2;
  } while (vec_size > 1 && !can_vectorize);
  return can_vectorize ? vec_size : 1;
}

Tensor stochastic_rounding_sgd_step_cuda(
    Tensor& param, const Tensor& grad, Tensor& momentum_buffer,
    const Tensor& inv_scale, const Tensor& found_inf,
    double lr, double momentum, double weight_decay, double dampening,
    bool nesterov, bool first_run, c10::optional<Generator> gen_) {

  if (param.numel() == 0) return param;

  const int64_t numel = param.numel();
  const int block_size = 256;
  const int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  dim3 dim_block(block_size);
  dim3 grid((numel + block_size - 1) / block_size);
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);

  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  uint64_t counter_offset = ((numel + dim_block.x * grid.x * UNROLL - 1) / (dim_block.x * grid.x * UNROLL)) * 4;
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(counter_offset);
  }

  if (cuda::detail::canUse32BitIndexMath(param)) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      param.scalar_type(), "stochastic_rounding_sgd_step_cuda", [&] {
        auto param_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(param);
        auto grad_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(grad);
        auto momentum_buffer_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(momentum_buffer);
        param_info.collapseDims();
        grad_info.collapseDims();
        momentum_buffer_info.collapseDims();

        int vec_size = get_vector_size<scalar_t>(param, grad, momentum_buffer);

        if (vec_size > 1) {
          switch (vec_size) {
            case 4:
              vectorized_stochastic_rounding_sgd_step_kernel<scalar_t, unsigned int, 1, 4><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                param_info, grad_info, momentum_buffer_info,
                inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                static_cast<float>(weight_decay), static_cast<float>(momentum), static_cast<float>(dampening), static_cast<float>(lr),
                nesterov, first_run, numel, rng_engine_inputs);
              break;
            case 2:
              vectorized_stochastic_rounding_sgd_step_kernel<scalar_t, unsigned int, 1, 2><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                param_info, grad_info, momentum_buffer_info,
                inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                static_cast<float>(weight_decay), static_cast<float>(momentum), static_cast<float>(dampening), static_cast<float>(lr),
                nesterov, first_run, numel, rng_engine_inputs);
              break;
          }
        } else {
          switch (param_info.dims) {
            case 1:
              stochastic_rounding_sgd_step_kernel<scalar_t, unsigned int, 1><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                param_info, grad_info, momentum_buffer_info,
                inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                static_cast<float>(weight_decay), static_cast<float>(momentum), static_cast<float>(dampening), static_cast<float>(lr),
                nesterov, first_run, numel, rng_engine_inputs);
              break;
            default:
              stochastic_rounding_sgd_step_kernel<scalar_t, unsigned int, -1><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                param_info, grad_info, momentum_buffer_info,
                inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                static_cast<float>(weight_decay), static_cast<float>(momentum), static_cast<float>(dampening), static_cast<float>(lr),
                nesterov, first_run, numel, rng_engine_inputs);
          }
        }
      });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      param.scalar_type(), "stochastic_rounding_sgd_step_cuda", [&] {
        auto param_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(param);
        auto grad_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(grad);
        auto momentum_buffer_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(momentum_buffer);
        param_info.collapseDims();
        grad_info.collapseDims();
        momentum_buffer_info.collapseDims();

        int vec_size = get_vector_size<scalar_t>(param, grad, momentum_buffer);

        if (vec_size > 1) {
          switch (vec_size) {
            case 4:
              vectorized_stochastic_rounding_sgd_step_kernel<scalar_t, uint64_t, 1, 4><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                param_info, grad_info, momentum_buffer_info,
                inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                static_cast<float>(weight_decay), static_cast<float>(momentum), static_cast<float>(dampening), static_cast<float>(lr),
                nesterov, first_run, numel, rng_engine_inputs);
              break;
            case 2:
              vectorized_stochastic_rounding_sgd_step_kernel<scalar_t, uint64_t, 1, 2><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                param_info, grad_info, momentum_buffer_info,
                inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                static_cast<float>(weight_decay), static_cast<float>(momentum), static_cast<float>(dampening), static_cast<float>(lr),
                nesterov, first_run, numel, rng_engine_inputs);
              break;
          }
        } else {
          switch (param_info.dims) {
            case 1:
              stochastic_rounding_sgd_step_kernel<scalar_t, uint64_t, 1><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                param_info, grad_info, momentum_buffer_info,
                inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                static_cast<float>(weight_decay), static_cast<float>(momentum), static_cast<float>(dampening), static_cast<float>(lr),
                nesterov, first_run, numel, rng_engine_inputs);
              break;
            default:
              stochastic_rounding_sgd_step_kernel<scalar_t, uint64_t, -1><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                param_info, grad_info, momentum_buffer_info,
                inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                static_cast<float>(weight_decay), static_cast<float>(momentum), static_cast<float>(dampening), static_cast<float>(lr),
                nesterov, first_run, numel, rng_engine_inputs);
          }
        }
      });
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return param;
}

} // namespace native
} // namespace at

#undef UNROLL
