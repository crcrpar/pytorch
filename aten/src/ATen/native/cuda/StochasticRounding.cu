#include <ATen/ATen.h>
#include <ATen/native/cuda/stochastic_rounding.cuh>


namespace at {
namespace native {

Tensor stochastic_rounding_cuda(const Tensor& input, c10::optional<Generator> gen_) {
  TORCH_CHECK(input.is_contiguous());
}

template<typename out_t, typename scalar_t>
void stochastic_rounding_cuda_kernel(
    at::Tensor &ret,
    const at::Tensor &input,
    at::PhiloxCudaState philox_args) {
  at::TensorIterator iter = at::TensorIteratorConfig()
    .add_output(ret)
    .add_input(input)
    .build();

  at::native::distribution_unary_kernel(iter, philox_args,
      [philox_args] GPU_LAMBDA (curandPhilox4_32_10_t &state, scalar_t inp) {
        #if defined(__CUDA_ARCH__)
        auto uniform_lambda = curand_uniform_wrapper(state);
        const float rand = uniform_lambda();
        return round_stochastically<out_type, scalar_t>(inp, rand);
        #else
        return inp;
        #endif
      })
}

template<typename out_t>
void stochastic_rounding_cuda_kernel_stub(
    at::Tensor &ret,
    const at::Tensor &input,
    at::PhiloxCudaState philox_args) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "stochastic_rounding_cuda", [&] {
    stochastic_rounding_cuda_kernel<out_t, scalar_t>(ret, input, philox_args);
  });
  return ret;
}


void stochastic_rounding_cuda_(Tensor &output, const Tensor &input, c10::optional<Generator> gen_) {
  TORCH_CHECK(output.numel() == input.numel());
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(44);
  }

  const auto dtype = output.dtype();

  switch (dtype) {
    case at::Double:
      stochastic_rounding_cuda_kernel_stub<at::Double>(output, input, rng_engine_inputs);
      break;
    case at::Float:
      stochastic_rounding_cuda_kenrel_stub<at::Float>(ouptut, input, rng_engine_inputs);
      break;
    case at::Half:
      stochastic_rounding_cuda_kernel_stub<at::Half>(output, input, rng_engine_inputs);
      break;
    default:
      break;
  }
}

Tensor stochastic_rounding_cuda(const Tensor &input, c10::optional<Generator> gen_) {
  Tensor ret = at::empty_like(input);
  stochastic_rounding_cuda_(ret, input, rng_engine_inputs);
  return ret;
}

} // namespace native
} // namespace at
