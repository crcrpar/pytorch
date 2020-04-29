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
__global__ void vectorized_stochastic_rounding_adam_step_kernel(
    at::cuda::detail::TensorInfo<scalar_t, IndexType> weights,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> gradients,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> exp_avg,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> exp_avg_sq,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> max_exp_avg_sq,
    float *inv_scale, float *found_inf,
    float lr, float beta1, float beta2,
    float weight_decay, float eps, int step,
    bool is_decoupled, bool is_amsgrad,
    int numel, std::pair<uint64_t, uint64_t> seeds) {
  static_assert(VEC <= 4, "Value of VEC must be in [2, 4].");

  if (*found_inf) return;

  using LoadT = memory::aligned_vector<scalar_t, VEC>;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, tid, seeds.second, &state);

  float m_correction = 1.0 - powf(beta1, step);
  float v_correction = 1.0 - powf(beta2, step);

  for (IndexType i = tid * VEC; i < numel; i += gridDim.x * blockDim.x * VEC) {
    scalar_t params[VEC], grads[VEC], exp_g[VEC], exp_g2[VEC], max_exp_g2[VEC];
    LoadT *p_value = reinterpret_cast<LoadT*>(&params);
    *p_value = *reinterpret_cast<LoadT*>(&weights.data[i]);
    LoadT *g_value = reinterpret_cast<LoadT*>(&grads);
    *g_value = *reinterpret_cast<LoadT*>(&gradients.data[i]);
    LoadT *exp_g_value = reinterpret_cast<LoadT*>(&exp_g);
    *exp_g_value = *reinterpret_cast<LoadT*>(&exp_avg.data[i]);
    LoadT *exp_g2_value = reinterpret_cast<LoadT*>(&exp_g2);
    *exp_g2_value = *reinterpret_cast<LoadT*>(&exp_avg_sq.data[i]);
    LoadT *max_exp_g2_value = reinterpret_cast<LoadT*>(&max_exp_g2);
    *max_exp_g2_value = *reinterpret_cast<LoadT*>(&max_exp_avg_sq.data[i]);

    for (int ii = 0; ii < VEC; ii++) {
      float weight = static_cast<float>(params[ii]);
      float gradient = static_cast<float>(grads[ii]) * (*inv_scale);
      float m = static_cast<float>(exp_g[ii]);
      float v = static_cast<float>(exp_g2[ii]);
      v = v * v;
      float4 random_values = curand_uniform4(&state);

      if (weight_decay != 0.0f) {
        if (is_decoupled)
          weight *= (1 - lr * weight_decay);
        else
          gradient += weight_decay * weight;
      }

      // Update m and v;
      m = beta1 * m + (1 - beta1) * gradient;
      v = beta2 * v + (1 - beta2) * (gradient * gradient);

      // Unbias v
      float max_v = v;
      if (is_amsgrad) {
        float prev_max_v = static_cast<float>(max_exp_g2[ii]);
        prev_max_v = prev_max_v * prev_max_v;
        max_v = fmaxf(prev_max_v, v);
      }

      weight -= (lr / m_correction) * m / (sqrtf(max_v / v_correction) + eps);

      params[ii] = round_stochastically<scalar_t>(weight, random_values.x);
      exp_g[ii] = round_stochastically<scalar_t>(m, random_values.y);
      exp_g2[ii] = round_stochastically<scalar_t>(sqrtf(v), random_values.z);
      if (is_amsgrad) {
        max_exp_g2[ii] = round_stochastically<scalar_t>(sqrtf(max_v), random_values.w);
      }
    }
    *(reinterpret_cast<LoadT*>(&weights.data[i])) = *reinterpret_cast<LoadT*>(&params[0]);
    *(reinterpret_cast<LoadT*>(&exp_avg.data[i])) = *reinterpret_cast<LoadT*>(&exp_g[0]);
    *(reinterpret_cast<LoadT*>(&exp_avg_sq.data[i])) = *reinterpret_cast<LoadT*>(&exp_g2[0]);
    if (is_amsgrad) {
      *(reinterpret_cast<LoadT*>(&max_exp_avg_sq.data[i])) = *reinterpret_cast<LoadT*>(&max_exp_g2[0]);
    }
    __syncthreads();
  }
}

template <typename scalar_t, typename IndexType, int ADims>
C10_LAUNCH_BOUNDS_2(256, 8)
__global__ void stochastic_rounding_adam_step_kernel(
    at::cuda::detail::TensorInfo<scalar_t, IndexType> weights,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> gradients,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> exp_avg,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> exp_avg_sq,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> max_exp_avg_sq,
    float *inv_scale, float *found_inf,
    float lr, float beta1, float beta2,
    float weight_decay, float eps, int step,
    bool is_decoupled, bool is_amsgrad,
    int numel, std::pair<uint64_t, uint64_t> seeds) {

  if (*found_inf) return;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, tid, seeds.second, &state);

  float m_correction = 1.0 - powf(beta1, step);
  float v_correction = 1.0 - powf(beta2, step);

  const IndexType unit = blockDim.x * gridDim.x * UNROLL;
  const IndexType rounded_size = (numel + unit - 1) / unit * unit;
  for (IndexType i = tid; i < rounded_size; i += blockDim.x * gridDim.x * UNROLL) {

    float params[UNROLL], grads[UNROLL], ms[UNROLL], vs[UNROLL], max_vs[UNROLL];

    // load values
    for (int ii = 0; ii < UNROLL; ii++) {
      const IndexType li = i + blockDim.x * gridDim.x * ii;
      if (li < numel) {
        IndexType offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(li, weights);
        params[ii] = static_cast<float>(weights.data[offset]);
        offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(li, gradients);
        grads[ii] = static_cast<float>(gradients.data[offset]) * (*inv_scale);
        offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(li, exp_avg);
        ms[ii] = static_cast<float>(exp_avg.data[offset]);
        offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(li, exp_avg_sq);
        float v = static_cast<float>(exp_avg_sq.data[offset]);
        vs[ii] = v * v;
        if (is_amsgrad) {
          offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(li, max_exp_avg_sq);
          float max_v = static_cast<float>(max_exp_avg_sq.data[offset]);
          max_vs[ii] = max_v * max_v;
        }
      }
    }

    // update & store
    for (int ii = 0; ii < UNROLL; ii++) {
      const IndexType li = i + blockDim.x * gridDim.x * ii;
      if (li < numel) {

        if (weight_decay != 0.0f) {
          if (is_decoupled)
            params[ii] *= (1 - lr * weight_decay);
          else
            grads[ii] += weight_decay * params[ii];
        }

        // Update m and v.
        ms[ii] = beta1 * ms[ii] + (1.0 - beta1) * grads[ii];
        vs[ii] = beta2 * vs[ii] + (1.0 - beta2) * (grads[ii] * grads[ii]);

        // Unbias v
        float max_v = vs[ii];
        if (is_amsgrad) {
          float prev_max_v = static_cast<float>(max_vs[ii]);
          prev_max_v = prev_max_v * prev_max_v;
          max_v = fmaxf(prev_max_v, vs[ii]);
        }

        params[ii] -= (lr / m_correction) * ms[ii] / (sqrtf(max_v / v_correction) + eps);

        float4 random_values = curand_uniform4(&state);
        IndexType offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(li, weights);
        weights.data[offset] = round_stochastically<scalar_t>(params[ii], random_values.x);
        offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(li, exp_avg);
        exp_avg.data[offset] = round_stochastically<scalar_t>(ms[ii], random_values.y);
        offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(li, exp_avg_sq);
        exp_avg_sq.data[offset] = round_stochastically<scalar_t>(sqrtf(vs[ii]), random_values.z);
        if (is_amsgrad) {
          offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(li, max_exp_avg_sq);
          max_exp_avg_sq.data[offset] = round_stochastically<scalar_t>(sqrtf(max_vs[ii]), random_values.w);
        }
      }
    }

    __syncthreads();
  }
}

template <typename scalar_t>
int get_vector_size(const Tensor& param, const Tensor& grad, const Tensor& exp_avg, const Tensor& exp_avg_sq, const Tensor& max_exp_avg_sq, bool is_amsgrad) {
  int vec_size = 4;
  auto memory_format = param.suggest_memory_format();
  if (!param.is_contiguous(memory_format) || !grad.is_contiguous(memory_format) || !exp_avg.is_contiguous(memory_format) || !exp_avg_sq.is_contiguous(memory_format) || !max_exp_avg_sq.is_contiguous(memory_format))
    return 1;
  vec_size = memory::can_vectorize_up_to<scalar_t>((char*)param.data_ptr());
  bool can_vectorize = true;

  do {
    can_vectorize = param.numel() % vec_size == 0;
    if (!can_vectorize)
      vec_size /= 2;
  } while (vec_size > 1 && !can_vectorize);
  return can_vectorize ? vec_size : 1;
}

Tensor stochastic_rounding_adam_step_cuda(
    Tensor& param,
    const Tensor& grad,
    Tensor& exp_avg,
    Tensor& exp_avg_sq,
    Tensor& max_exp_avg_sq,
    const Tensor& inv_scale,
    const Tensor& found_inf,
    double lr, double beta1, double beta2,
    double weight_decay, double eps, int64_t step,
    bool is_decoupled, bool is_amsgrad, c10::optional<Generator> gen_) {

  if (param.numel() == 0) return param;

  const int64_t numel = param.numel();
  const int block_size = 256;
  const int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  dim3 dim_block(block_size);
  dim3 grid((numel + block_size - 1) / block_size);
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);

  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());

  uint64_t counter_offset = ((numel + dim_block.x * grid.x * UNROLL - 1) / (block_size * grid.x * UNROLL)) * 4;
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(counter_offset);
  }

  if (cuda::detail::canUse32BitIndexMath(param)) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        param.scalar_type(), "stochastic_rounding_adam_step_cuda", [&] {
          auto param_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(param);
          auto grad_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(grad);
          auto exp_avg_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(exp_avg);
          auto exp_avg_sq_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(exp_avg_sq);
          auto max_exp_avg_sq_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(max_exp_avg_sq);
          param_info.collapseDims();
          grad_info.collapseDims();
          exp_avg_info.collapseDims();
          exp_avg_sq_info.collapseDims();
          max_exp_avg_sq_info.collapseDims();

          int vec_size = get_vector_size<scalar_t>(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, is_amsgrad);

          if (vec_size > 1) {
            switch (vec_size) {
              case 4:
                vectorized_stochastic_rounding_adam_step_kernel<scalar_t, unsigned int, 1, 4><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                  param_info, grad_info, exp_avg_info, exp_avg_sq_info, max_exp_avg_sq_info,
                  inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                  lr, beta1, beta2, weight_decay, eps, step,
                  is_decoupled, is_amsgrad, numel, rng_engine_inputs);
                break;
              case 2:
                vectorized_stochastic_rounding_adam_step_kernel<scalar_t, unsigned int, 1, 2><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                  param_info, grad_info, exp_avg_info, exp_avg_sq_info, max_exp_avg_sq_info,
                  inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                  lr, beta1, beta2, weight_decay, eps, step,
                  is_decoupled, is_amsgrad, numel, rng_engine_inputs);
                break;
            }
          } else {
            switch (param_info.dims) {
              case 1:
                stochastic_rounding_adam_step_kernel<scalar_t, unsigned int, 1><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                    param_info, grad_info, exp_avg_info, exp_avg_sq_info, max_exp_avg_sq_info,
                    inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                    lr, beta1, beta2, weight_decay, eps, step,
                    is_decoupled, is_amsgrad, numel, rng_engine_inputs);
                break;
              default:
                stochastic_rounding_adam_step_kernel<scalar_t, unsigned int, -1><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                    param_info, grad_info, exp_avg_info, exp_avg_sq_info, max_exp_avg_sq_info,
                    inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                    lr, beta1, beta2, weight_decay, eps, step,
                    is_decoupled, is_amsgrad, numel, rng_engine_inputs);
            }
          }
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        param.scalar_type(), "stochastic_rounding_adam_step_cuda", [&] {
          auto param_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(param);
          auto grad_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(grad);
          auto exp_avg_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(exp_avg);
          auto exp_avg_sq_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(exp_avg_sq);
          auto max_exp_avg_sq_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(max_exp_avg_sq);
          param_info.collapseDims();
          grad_info.collapseDims();
          exp_avg_info.collapseDims();
          exp_avg_sq_info.collapseDims();
          max_exp_avg_sq_info.collapseDims();

          int vec_size = get_vector_size<scalar_t>(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, is_amsgrad);

          if (vec_size > 1) {
            switch (vec_size) {
              case 4:
                vectorized_stochastic_rounding_adam_step_kernel<scalar_t, uint64_t, 1, 4><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                  param_info, grad_info, exp_avg_info, exp_avg_sq_info, max_exp_avg_sq_info,
                  inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                  lr, beta1, beta2, weight_decay, eps, step,
                  is_decoupled, is_amsgrad, numel, rng_engine_inputs);
                break;
              case 2:
                vectorized_stochastic_rounding_adam_step_kernel<scalar_t, uint64_t, 1, 2><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                  param_info, grad_info, exp_avg_info, exp_avg_sq_info, max_exp_avg_sq_info,
                  inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                  lr, beta1, beta2, weight_decay, eps, step,
                  is_decoupled, is_amsgrad, numel, rng_engine_inputs);
                break;
            }
          } else {
            switch (param_info.dims) {
              case 1:
                stochastic_rounding_adam_step_kernel<scalar_t, uint64_t, 1><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                    param_info, grad_info, exp_avg_info, exp_avg_sq_info, max_exp_avg_sq_info,
                    inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                    lr, beta1, beta2, weight_decay, eps, step,
                    is_decoupled, is_amsgrad, numel, rng_engine_inputs);
                break;
              default:
                stochastic_rounding_adam_step_kernel<scalar_t, uint64_t, -1><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                    param_info, grad_info, exp_avg_info, exp_avg_sq_info, max_exp_avg_sq_info,
                    inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                    lr, beta1, beta2, weight_decay, eps, step,
                    is_decoupled, is_amsgrad, numel, rng_engine_inputs);
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
