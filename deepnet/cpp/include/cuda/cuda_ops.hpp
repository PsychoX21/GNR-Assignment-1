#pragma once

#ifdef USE_CUDA

#include <vector>

namespace deepnet {
namespace cuda {

// Low-level CUDA kernels (operate on device pointers)
void add_cuda(const float *a, const float *b, float *out, int size);
void mul_cuda(const float *a, const float *b, float *out, int size);
void matmul_cuda(const float *a, const float *b, float *out, int M, int N,
                 int K);
void relu_cuda(const float *input, float *output, int size);
void sigmoid_cuda(const float *input, float *output, int size);
void tanh_cuda(const float *input, float *output, int size);

// Memory operations
void *cuda_malloc(size_t size);
void cuda_free(void *ptr);
void cuda_memcpy_host_to_device(void *dst, const void *src, size_t size);
void cuda_memcpy_device_to_host(void *dst, const void *src, size_t size);

// ============================================================
// High-level host wrappers: host vector in -> compute on GPU -> host vector out
// These handle all device memory allocation and transfers internally.
// ============================================================
void add_cuda_host(const std::vector<float> &a, const std::vector<float> &b,
                   std::vector<float> &out, int size);
void mul_cuda_host(const std::vector<float> &a, const std::vector<float> &b,
                   std::vector<float> &out, int size);
void matmul_cuda_host(const std::vector<float> &a, const std::vector<float> &b,
                      std::vector<float> &out, int M, int N, int K);
void relu_cuda_host(const std::vector<float> &input,
                    std::vector<float> &output, int size);
void sigmoid_cuda_host(const std::vector<float> &input,
                       std::vector<float> &output, int size);
void tanh_cuda_host(const std::vector<float> &input,
                    std::vector<float> &output, int size);

// Check if CUDA is available at runtime
bool is_cuda_available();

} // namespace cuda
} // namespace deepnet

#endif // USE_CUDA
