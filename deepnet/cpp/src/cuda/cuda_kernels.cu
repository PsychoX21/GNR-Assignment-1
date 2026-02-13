#ifdef USE_CUDA

#include "cuda/cuda_ops.hpp"
#include "cuda/cuda_utils.hpp"
#include <cuda_runtime.h>
#include <vector>


namespace deepnet {
namespace cuda {

// Kernel implementations
constexpr int BLOCK_SIZE = 256;
constexpr int TILE_SIZE = 16;

// ============================================================
// CUDA Kernels
// ============================================================

__global__ void add_kernel(const float *a, const float *b, float *out,
                           int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] + b[idx];
  }
}

__global__ void mul_kernel(const float *a, const float *b, float *out,
                           int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] * b[idx];
  }
}

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M,
                              int N, int K) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    if (row < M && t * TILE_SIZE + threadIdx.x < K)
      As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0f;

    if (t * TILE_SIZE + threadIdx.y < K && col < N)
      Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

__global__ void relu_kernel(const float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = fmaxf(0.0f, input[idx]);
  }
}

__global__ void sigmoid_kernel(const float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
  }
}

__global__ void tanh_kernel(const float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = tanhf(input[idx]);
  }
}

// ============================================================
// Low-level kernel launchers (device pointers)
// ============================================================

void add_cuda(const float *a, const float *b, float *out, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  add_kernel<<<blocks, BLOCK_SIZE>>>(a, b, out, size);
  CUDA_CHECK(cudaGetLastError());
}

void mul_cuda(const float *a, const float *b, float *out, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  mul_kernel<<<blocks, BLOCK_SIZE>>>(a, b, out, size);
  CUDA_CHECK(cudaGetLastError());
}

void matmul_cuda(const float *a, const float *b, float *out, int M, int N,
                 int K) {
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  matmul_kernel<<<grid, block>>>(a, b, out, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}

void relu_cuda(const float *input, float *output, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  relu_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
  CUDA_CHECK(cudaGetLastError());
}

void sigmoid_cuda(const float *input, float *output, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sigmoid_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
  CUDA_CHECK(cudaGetLastError());
}

void tanh_cuda(const float *input, float *output, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanh_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
  CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Memory operations
// ============================================================

void *cuda_malloc(size_t size) {
  void *ptr;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  return ptr;
}

void cuda_free(void *ptr) {
  if (ptr) {
    CUDA_CHECK(cudaFree(ptr));
  }
}

void cuda_memcpy_host_to_device(void *dst, const void *src, size_t size) {
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void cuda_memcpy_device_to_host(void *dst, const void *src, size_t size) {
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void cuda_memcpy_device_to_device(void *dst, const void *src, size_t size) {
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

// ============================================================
// High-level host wrappers: host vector -> GPU compute -> host vector
// Pattern: allocate device mem, copy in, run kernel, copy out, free
// ============================================================

void add_cuda_host(const std::vector<float> &a, const std::vector<float> &b,
                   std::vector<float> &out, int size) {
  size_t bytes = size * sizeof(float);
  float *d_a = (float *)cuda_malloc(bytes);
  float *d_b = (float *)cuda_malloc(bytes);
  float *d_out = (float *)cuda_malloc(bytes);

  cuda_memcpy_host_to_device(d_a, a.data(), bytes);
  cuda_memcpy_host_to_device(d_b, b.data(), bytes);

  add_cuda(d_a, d_b, d_out, size);
  cudaDeviceSynchronize();

  cuda_memcpy_device_to_host(out.data(), d_out, bytes);

  cuda_free(d_a);
  cuda_free(d_b);
  cuda_free(d_out);
}

void mul_cuda_host(const std::vector<float> &a, const std::vector<float> &b,
                   std::vector<float> &out, int size) {
  size_t bytes = size * sizeof(float);
  float *d_a = (float *)cuda_malloc(bytes);
  float *d_b = (float *)cuda_malloc(bytes);
  float *d_out = (float *)cuda_malloc(bytes);

  cuda_memcpy_host_to_device(d_a, a.data(), bytes);
  cuda_memcpy_host_to_device(d_b, b.data(), bytes);

  mul_cuda(d_a, d_b, d_out, size);
  cudaDeviceSynchronize();

  cuda_memcpy_device_to_host(out.data(), d_out, bytes);

  cuda_free(d_a);
  cuda_free(d_b);
  cuda_free(d_out);
}

void matmul_cuda_host(const std::vector<float> &a, const std::vector<float> &b,
                      std::vector<float> &out, int M, int N, int K) {
  size_t bytes_a = M * K * sizeof(float);
  size_t bytes_b = K * N * sizeof(float);
  size_t bytes_out = M * N * sizeof(float);

  float *d_a = (float *)cuda_malloc(bytes_a);
  float *d_b = (float *)cuda_malloc(bytes_b);
  float *d_out = (float *)cuda_malloc(bytes_out);

  cuda_memcpy_host_to_device(d_a, a.data(), bytes_a);
  cuda_memcpy_host_to_device(d_b, b.data(), bytes_b);

  matmul_cuda(d_a, d_b, d_out, M, N, K);
  cudaDeviceSynchronize();

  cuda_memcpy_device_to_host(out.data(), d_out, bytes_out);

  cuda_free(d_a);
  cuda_free(d_b);
  cuda_free(d_out);
}

void relu_cuda_host(const std::vector<float> &input,
                    std::vector<float> &output, int size) {
  size_t bytes = size * sizeof(float);
  float *d_in = (float *)cuda_malloc(bytes);
  float *d_out = (float *)cuda_malloc(bytes);

  cuda_memcpy_host_to_device(d_in, input.data(), bytes);

  relu_cuda(d_in, d_out, size);
  cudaDeviceSynchronize();

  cuda_memcpy_device_to_host(output.data(), d_out, bytes);

  cuda_free(d_in);
  cuda_free(d_out);
}

void sigmoid_cuda_host(const std::vector<float> &input,
                       std::vector<float> &output, int size) {
  size_t bytes = size * sizeof(float);
  float *d_in = (float *)cuda_malloc(bytes);
  float *d_out = (float *)cuda_malloc(bytes);

  cuda_memcpy_host_to_device(d_in, input.data(), bytes);

  sigmoid_cuda(d_in, d_out, size);
  cudaDeviceSynchronize();

  cuda_memcpy_device_to_host(output.data(), d_out, bytes);

  cuda_free(d_in);
  cuda_free(d_out);
}

void tanh_cuda_host(const std::vector<float> &input,
                    std::vector<float> &output, int size) {
  size_t bytes = size * sizeof(float);
  float *d_in = (float *)cuda_malloc(bytes);
  float *d_out = (float *)cuda_malloc(bytes);

  cuda_memcpy_host_to_device(d_in, input.data(), bytes);

  tanh_cuda(d_in, d_out, size);
  cudaDeviceSynchronize();

  cuda_memcpy_device_to_host(output.data(), d_out, bytes);

  cuda_free(d_in);
  cuda_free(d_out);
}

} // namespace cuda
} // namespace deepnet

#endif // USE_CUDA
