#ifdef USE_CUDA

#include "cuda/cuda_ops.hpp"
#include "cuda/cuda_utils.hpp"
#include <cuda_runtime.h>


namespace deepnet {
namespace cuda {

// Kernel implementations
// Thread block size
constexpr int BLOCK_SIZE = 256;
constexpr int TILE_SIZE = 16;

// Element-wise addition kernel
__global__ void add_kernel(const float *a, const float *b, float *out,
                           int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] + b[idx];
  }
}

void add_cuda(const float *a, const float *b, float *out, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  add_kernel<<<blocks, BLOCK_SIZE>>>(a, b, out, size);
  CUDA_CHECK(cudaGetLastError());
}

// Element-wise multiplication kernel
__global__ void mul_kernel(const float *a, const float *b, float *out,
                           int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] * b[idx];
  }
}

void mul_cuda(const float *a, const float *b, float *out, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  mul_kernel<<<blocks, BLOCK_SIZE>>>(a, b, out, size);
  CUDA_CHECK(cudaGetLastError());
}

// Matrix multiplication kernel (simple tiled version)
__global__ void matmul_kernel(const float *A, const float *B, float *C, int M,
                              int N, int K) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    // Load tiles into shared memory
    if (row < M && t * TILE_SIZE + threadIdx.x < K)
      As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0f;

    if (t * TILE_SIZE + threadIdx.y < K && col < N)
      Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // Compute partial dot product
    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

void matmul_cuda(const float *a, const float *b, float *out, int M, int N,
                 int K) {
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  matmul_kernel<<<grid, block>>>(a, b, out, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}

// ReLU activation kernel
__global__ void relu_kernel(const float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = fmaxf(0.0f, input[idx]);
  }
}

void relu_cuda(const float *input, float *output, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  relu_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
  CUDA_CHECK(cudaGetLastError());
}

// Sigmoid activation kernel
__global__ void sigmoid_kernel(const float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
  }
}

void sigmoid_cuda(const float *input, float *output, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sigmoid_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
  CUDA_CHECK(cudaGetLastError());
}

// Tanh activation kernel
__global__ void tanh_kernel(const float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = tanhf(input[idx]);
  }
}

void tanh_cuda(const float *input, float *output, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanh_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
  CUDA_CHECK(cudaGetLastError());
}

// Memory operations with error checking
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

} // namespace cuda
} // namespace deepnet

#endif // USE_CUDA
