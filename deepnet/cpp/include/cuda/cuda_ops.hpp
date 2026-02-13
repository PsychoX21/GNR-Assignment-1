#pragma once

#ifdef USE_CUDA

namespace deepnet {
namespace cuda {

// CUDA operations (stubs for now - can be implemented with actual CUDA kernels)
void add_cuda(const float *a, const float *b, float *out, int size);
void mul_cuda(const float *a, const float *b, float *out, int size);
void matmul_cuda(const float *a, const float *b, float *out, int M, int N,
                 int K);
void relu_cuda(const float *input, float *output, int size);
void sigmoid_cuda(const float *input, float *output, int size);

// Memory operations
void *cuda_malloc(size_t size);
void cuda_free(void *ptr);
void cuda_memcpy_host_to_device(void *dst, const void *src, size_t size);
void cuda_memcpy_device_to_host(void *dst, const void *src, size_t size);

} // namespace cuda
} // namespace deepnet

#endif // USE_CUDA
