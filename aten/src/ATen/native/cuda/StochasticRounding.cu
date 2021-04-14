#include <ATen/ATen.h>
#include <ATen/native/cuda/stochastic_rounding.cuh>


namespace at {
namespace native {

template<typename out_t>
void stochastic_rounding_cuda_kernel(
    Tensor &ret,
    const Tensor &input,
    PhiloxCudaState philox_args) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "stochastic_rounding_cuda", [&] {
    at::TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(input)
      .build();

    at::native::distribution_unary_kernel(iter, philox_args,
        [philox_args] GPU_LAMBDA (curandPhilox4_32_10_t &state, scalar_t inp) {
          #if defined(__CUDA_ARCH__)
          auto uniform_lambda = curand_uniform_wrapper(state);
          const float rand = uniform_lambda();
          return round_stochastically<out_t, scalar_t>(inp, rand);
          #else
          return inp;
          #endif
        }
    );
  });
  return ret;
}


void stochastic_rounding_cuda_(Tensor &output, const Tensor &input, c10::optional<Generator> gen_) {
  TORCH_CHECK(output.numel() == input.numel());
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  PhiloxCudaState philox_args;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    philox_args = gen->philox_cuda_state(44);
  }

  const auto dtype = output.dtype();

  if (dtype == at::ScalarType::Double) {
    stochastic_rounding_cuda_kernel_stub<at::ScalarType::Double>(output, input, philox_args);
  } else if (dtype == at::ScalarType::Float) {
    stochastic_rounding_cuda_kernel_stub<at::ScalarType::Float>(output, input, philox_args);
  } else if (dtype == at::ScalarType::Half) {
    stochastic_rounding_cuda_kernel_stub<at::ScalarType::Half>(output, input, philox_args);
  } else {
    TORCH_CHECK(false, "Dtype must be one of Double, Float, Half");
  }
}

Tensor stochastic_rounding_cuda(const Tensor &input, c10::optional<Generator> gen_) {
  Tensor ret = at::empty_like(input);
  stochastic_rounding_cuda_(ret, input, gen_);
  return ret;
}

} // namespace native
} // namespace at
