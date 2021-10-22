#include <iostream>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cublas_v2.h>

void a_cublas_function() {
    printf("hello world\n");

    cublasHandle_t handle;
    TORCH_CUDABLAS_CHECK(cublasCreate(&handle));

    TORCH_CUDABLAS_CHECK(cublasDestroy(handle));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("a_cublas_function", &a_cublas_function, "a cublas function");
}
