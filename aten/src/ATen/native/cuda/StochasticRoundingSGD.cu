#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/CUDAGenerator.h>
#include <ATen/native/cuda/stochastic_rounding.cuh>

#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>

#include <curand_kernel.h>


#if 0
#define DISPATCH_FLOAT_AND_HALF(TYPE, NAME, ...)                 \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)      \
      default:                                                               \
        AT_ERROR(#NAME, " not impemented for '", toString(_st), "'");       \
    }                                                                        \
  }()
#endif

namespace at {
namespace native {

// SGD update math with Stochastic Rounding
template <typename scalar_t>
__global__ void stochastic_rounding_sgd_step_kernel(
    scalar_t *weights, scalar_t *gradients, scalar_t *momentum_buffer,
    float inv_scale, float found_inf,
    float weight_decay, float momentum, float dampening, float lr,
    bool nesterov, bool first_run, int numel, std::pair<uint64_t, uint64_t> seeds)
{

  // 1.0 indicates that any gradients contain inf or nan.
  // See below about `found_inf`:
  //  - https://github.com/mcarilli/pytorch/blob/382d02f01d104049179f4f056cc9258caad029af/aten/src/ATen/native/cuda/AmpKernels.cu#L40-L41
  //  - https://github.com/mcarilli/pytorch/blob/382d02f01d104049179f4f056cc9258caad029af/aten/src/ATen/native/cuda/AmpKernels.cu#L116-L117
  if (found_inf) return;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, tid, seeds.second, &state);

  for (int i = tid; i < numel; i += blockDim.x * gridDim.x) {
    float weight = static_cast<float>(weights[i]);
    float gradient = static_cast<float>(gradients[i]) * inv_scale;
    float velocity = static_cast<float>(momentum_buffer[i]);
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

    // Rounding.
    weights[i] = round_stochastically<scalar_t>(weight, random_values.x);
    if (momentum != 0.0f)
      momentum_buffer[i] = round_stochastically<scalar_t>(velocity, random_values.y);
  }
}

#define N_RV_PER_API_CALL 4
Tensor stochastic_rounding_sgd_step_cuda(
    Tensor& param, const Tensor& grad, Tensor& momentum_buffer,
    const Tensor& inv_scale, const Tensor& found_inf,
    float weight_decay, float momentum, float dampening, float lr,
    bool nesterov, bool first_run, Generator gen_) {

  if (param.numel() == 0) return param;

  TORCH_CHECK(param.is_contiguous());
  TORCH_CHECK(grad.is_contiguous());
  TORCH_CHECK(momentum_buffer.is_contiguous());

  const int64_t numel = param.numel();
  const int block_size = 256;
  const int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  dim3 dim_block(block_size);
  dim3 grid((numel + block_size - 1) / block_size);
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);

  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  uint64_t counter_offset = ((numel + dim_block.x * grid.x - 1) / (dim_block.x * grid.x)) * N_RV_PER_API_CALL;
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(counter_offset);
  }

  float inv_scale_value, found_inf_value;

  if (inv_scale.defined()) {
    inv_scale_value = *inv_scale.data_ptr<float>();
  } else {
    inv_scale_value = 1.0f;
  }

  if (found_inf.defined()) {
    found_inf_value = *found_inf.data_ptr<float>();
  } else {
    found_inf_value = 0.0f;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        param.scalar_type(), "_stochastic_rounding_sgd_step_cuda", [&] {
        stochastic_rounding_sgd_step_kernel<scalar_t><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
            param.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            momentum_buffer.data_ptr<scalar_t>(),
            inv_scale_value, found_inf_value,
            weight_decay, momentum, dampening, lr,
            nesterov, first_run, numel, rng_engine_inputs);
      });
  return param;
}
#undef N_RV_PER_API_CALL

} // namespace native
} // namespace at
