#include "tensor.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

#ifdef USE_CUDA
#include "cuda/cuda_ops.hpp"
#endif
namespace deepnet {

// Autograd function implementations
struct AddBackward : public AutogradFunction {
  std::vector<TensorPtr> backward(const TensorPtr &grad_output) override {
    return {grad_output, grad_output};
  }
};

struct MulBackward : public AutogradFunction {
  std::vector<TensorPtr> backward(const TensorPtr &grad_output) override {
    // d/dx (x * y) = y, d/dy (x * y) = x
    return {grad_output->mul(inputs[1]), grad_output->mul(inputs[0])};
  }
};

struct MatMulBackward : public AutogradFunction {
  std::vector<TensorPtr> backward(const TensorPtr &grad_output) override {
    // For C = A @ B:
    // dL/dA = dL/dC @ B^T
    // dL/dB = A^T @ dL/dC
    auto grad_a = grad_output->matmul(inputs[1]->transpose(0, 1));
    auto grad_b = inputs[0]->transpose(0, 1)->matmul(grad_output);
    return {grad_a, grad_b};
  }
};

struct ReLUBackward : public AutogradFunction {
  std::vector<TensorPtr> backward(const TensorPtr &grad_output) override {
    auto grad_input =
        Tensor::zeros(inputs[0]->shape, false, grad_output->is_cuda);
    for (size_t i = 0; i < grad_input->data.size(); ++i) {
      grad_input->data[i] =
          inputs[0]->data[i] > 0 ? grad_output->data[i] : 0.0f;
    }
    return {grad_input};
  }
};

// Tensor constructors
Tensor::Tensor() : requires_grad(false), is_cuda(false) {}

Tensor::Tensor(const std::vector<int> &shape, bool requires_grad, bool cuda)
    : shape(shape), requires_grad(requires_grad), is_cuda(cuda) {
  int total_size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  data.resize(total_size, 0.0f);
  if (requires_grad) {
    grad.resize(total_size, 0.0f);
  }
  compute_strides();
}

Tensor::Tensor(const std::vector<float> &data, const std::vector<int> &shape,
               bool requires_grad, bool cuda)
    : data(data), shape(shape), requires_grad(requires_grad), is_cuda(cuda) {
  if (requires_grad) {
    grad.resize(data.size(), 0.0f);
  }
  compute_strides();
}

// Factory methods
TensorPtr Tensor::zeros(const std::vector<int> &shape, bool requires_grad,
                        bool cuda) {
  return std::make_shared<Tensor>(shape, requires_grad, cuda);
}

TensorPtr Tensor::ones(const std::vector<int> &shape, bool requires_grad,
                       bool cuda) {
  auto tensor = std::make_shared<Tensor>(shape, requires_grad, cuda);
  std::fill(tensor->data.begin(), tensor->data.end(), 1.0f);
  return tensor;
}

TensorPtr Tensor::randn(const std::vector<int> &shape, float mean, float std,
                        bool requires_grad, bool cuda) {
  auto tensor = std::make_shared<Tensor>(shape, requires_grad, cuda);
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::normal_distribution<float> dist(mean, std);
  for (auto &val : tensor->data) {
    val = dist(gen);
  }
  return tensor;
}

TensorPtr Tensor::from_data(const std::vector<float> &data,
                            const std::vector<int> &shape, bool requires_grad,
                            bool cuda) {
  return std::make_shared<Tensor>(data, shape, requires_grad, cuda);
}

// Shape operations
int Tensor::size() const { return shape.empty() ? 0 : shape[0]; }

int Tensor::size(int dim) const {
  if (dim < 0)
    dim += static_cast<int>(shape.size());
  return shape[dim];
}

int Tensor::numel() const { return static_cast<int>(data.size()); }

int Tensor::ndim() const { return static_cast<int>(shape.size()); }

void Tensor::compute_strides() {
  strides.resize(shape.size());
  int stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
}

TensorPtr Tensor::reshape(const std::vector<int> &new_shape) {
  auto output = Tensor::from_data(data, new_shape, requires_grad, is_cuda);
  if (requires_grad) {
    output->grad_fn = nullptr; // Reshape doesn't need special gradient handling
  }
  return output;
}

TensorPtr Tensor::view(const std::vector<int> &new_shape) {
  return reshape(new_shape);
}

TensorPtr Tensor::flatten(int start_dim, int end_dim) {
  if (end_dim == -1)
    end_dim = static_cast<int>(shape.size()) - 1;

  std::vector<int> new_shape;
  int flat_size = 1;

  for (int i = 0; i < start_dim; ++i) {
    new_shape.push_back(shape[i]);
  }

  for (int i = start_dim; i <= end_dim; ++i) {
    flat_size *= shape[i];
  }
  new_shape.push_back(flat_size);

  for (size_t i = end_dim + 1; i < shape.size(); ++i) {
    new_shape.push_back(shape[i]);
  }

  return reshape(new_shape);
}

// Element-wise operations
TensorPtr Tensor::add(const TensorPtr &other) {
  check_shape_compatible(other);
  auto output =
      Tensor::zeros(shape, requires_grad || other->requires_grad, is_cuda);

  // CUDA path
#ifdef USE_CUDA
  if (is_cuda) {
    cuda::add_cuda_host(data, other->data, output->data, (int)data.size());
  } else
#endif
  {
    // CPU path
    #pragma omp parallel for
    for (int i = 0; i < (int)data.size(); ++i) {
      output->data[i] = data[i] + other->data[i];
    }
  }

  if (output->requires_grad) {
    auto grad_fn = std::make_shared<AddBackward>();
    grad_fn->inputs = {shared_from_this(), other};
    output->grad_fn = grad_fn;
  }

  return output;
}

TensorPtr Tensor::mul(const TensorPtr &other) {
  check_shape_compatible(other);
  auto output =
      Tensor::zeros(shape, requires_grad || other->requires_grad, is_cuda);

  // CUDA path
#ifdef USE_CUDA
  if (is_cuda) {
    cuda::mul_cuda_host(data, other->data, output->data, (int)data.size());
  } else
#endif
  {
    // CPU path
    #pragma omp parallel for
    for (int i = 0; i < (int)data.size(); ++i) {
      output->data[i] = data[i] * other->data[i];
    }
  }

  if (output->requires_grad) {
    auto grad_fn = std::make_shared<MulBackward>();
    grad_fn->inputs = {shared_from_this(), other};
    output->grad_fn = grad_fn;
  }

  return output;
}

TensorPtr Tensor::sub(const TensorPtr &other) {
  auto neg_other = other->mul_scalar(-1.0f);
  return add(neg_other);
}

TensorPtr Tensor::div(const TensorPtr &other) {
  auto output =
      Tensor::zeros(shape, requires_grad || other->requires_grad, is_cuda);
  #pragma omp parallel for
  for (int i = 0; i < (int)data.size(); ++i) {
    output->data[i] = data[i] / (other->data[i] + 1e-8f);
  }
  return output;
}

TensorPtr Tensor::add_scalar(float scalar) {
  auto output = clone();
  for (auto &val : output->data) {
    val += scalar;
  }
  return output;
}

TensorPtr Tensor::mul_scalar(float scalar) {
  auto output = clone();
  output->requires_grad = requires_grad;
  for (auto &val : output->data) {
    val *= scalar;
  }
  if (requires_grad) {
    output->grad.resize(data.size(), 0.0f);
  }
  return output;
}

// Operators
TensorPtr Tensor::operator+(const TensorPtr &other) { return add(other); }
TensorPtr Tensor::operator-(const TensorPtr &other) { return sub(other); }
TensorPtr Tensor::operator*(const TensorPtr &other) { return mul(other); }
TensorPtr Tensor::operator/(const TensorPtr &other) { return div(other); }

// Matrix operations
TensorPtr Tensor::matmul(const TensorPtr &other) {
  if (shape.size() != 2 || other->shape.size() != 2) {
    throw std::runtime_error("matmul requires 2D tensors");
  }
  if (shape[1] != other->shape[0]) {
    throw std::runtime_error("matmul shape mismatch");
  }

  int M = shape[0];
  int K = shape[1];
  int N = other->shape[1];

  auto output =
      Tensor::zeros({M, N}, requires_grad || other->requires_grad, is_cuda);

  // CUDA path
#ifdef USE_CUDA
  if (is_cuda) {
    cuda::matmul_cuda_host(data, other->data, output->data, M, N, K);
  } else
#endif
  {
    // CPU path
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
          sum += data[i * K + k] * other->data[k * N + j];
        }
        output->data[i * N + j] = sum;
      }
    }
  }

  if (output->requires_grad) {
    auto grad_fn = std::make_shared<MatMulBackward>();
    grad_fn->inputs = {shared_from_this(), other};
    output->grad_fn = grad_fn;
  }

  return output;
}

TensorPtr Tensor::mm(const TensorPtr &other) { return matmul(other); }

TensorPtr Tensor::transpose(int dim0, int dim1) {
  std::vector<int> new_shape = shape;
  std::swap(new_shape[dim0], new_shape[dim1]);

  auto output = Tensor::zeros(new_shape, requires_grad, is_cuda);

  // For 2D transpose
  if (shape.size() == 2 && dim0 == 0 && dim1 == 1) {
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < shape[1]; ++j) {
        output->data[j * shape[0] + i] = data[i * shape[1] + j];
      }
    }
  }

  return output;
}

// Reduction operations
TensorPtr Tensor::sum(int dim, bool keepdim) {
  if (dim == -1) {
    // Sum all elements
    float total = std::accumulate(data.begin(), data.end(), 0.0f);
    auto output = Tensor::from_data({total}, {1}, requires_grad, is_cuda);
    return output;
  }

// Dimension-specific sum (simplified)
  std::vector<int> new_shape;
  int reduced_size = shape[dim];
  int outer = 1, inner = 1;
  for (int i = 0; i < dim; ++i) {
    outer *= shape[i];
    new_shape.push_back(shape[i]);
  }
  if (keepdim) new_shape.push_back(1);
  for (size_t i = dim + 1; i < shape.size(); ++i) {
    inner *= shape[i];
    new_shape.push_back(shape[i]);
  }

  auto output = Tensor::zeros(new_shape, requires_grad, is_cuda);
  for (int o = 0; o < outer; ++o) {
    for (int n = 0; n < inner; ++n) {
      float sum_val = 0.0f;
      for (int r = 0; r < reduced_size; ++r) {
        sum_val += data[(o * reduced_size + r) * inner + n];
      }
      output->data[o * inner + n] = sum_val;
    }
  }
  return output;
}

TensorPtr Tensor::mean(int dim, bool keepdim) {
  auto sum_tensor = sum(dim, keepdim);
  return sum_tensor->mul_scalar(1.0f / data.size());
}

// Activations
TensorPtr Tensor::relu() {
  auto output = Tensor::zeros(shape, requires_grad, is_cuda);
#ifdef USE_CUDA
  if (is_cuda) {
    cuda::relu_cuda_host(data, output->data, (int)data.size());
  } else
#endif
  {
    #pragma omp parallel for
    for (int i = 0; i < (int)data.size(); ++i) {
      output->data[i] = std::max(0.0f, data[i]);
    }
  }

  if (requires_grad) {
    auto grad_fn = std::make_shared<ReLUBackward>();
    grad_fn->inputs = {shared_from_this()};
    output->grad_fn = grad_fn;
  }

  return output;
}

TensorPtr Tensor::leaky_relu(float negative_slope) {
  auto output = Tensor::zeros(shape, requires_grad, is_cuda);
  #pragma omp parallel for
  for (int i = 0; i < (int)data.size(); ++i) {
    output->data[i] = data[i] > 0 ? data[i] : negative_slope * data[i];
  }
  return output;
}

TensorPtr Tensor::tanh_() {
  auto output = Tensor::zeros(shape, requires_grad, is_cuda);
#ifdef USE_CUDA
  if (is_cuda) {
    cuda::tanh_cuda_host(data, output->data, (int)data.size());
  } else
#endif
  {
    #pragma omp parallel for
    for (int i = 0; i < (int)data.size(); ++i) {
      output->data[i] = std::tanh(data[i]);
    }
  }
  return output;
}

TensorPtr Tensor::sigmoid() {
  auto output = Tensor::zeros(shape, requires_grad, is_cuda);
#ifdef USE_CUDA
  if (is_cuda) {
    cuda::sigmoid_cuda_host(data, output->data, (int)data.size());
  } else
#endif
  {
    #pragma omp parallel for
    for (int i = 0; i < (int)data.size(); ++i) {
      output->data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
  }
  return output;
}

// Math operations
TensorPtr Tensor::exp() {
  auto output = Tensor::zeros(shape, requires_grad, is_cuda);
  #pragma omp parallel for
  for (int i = 0; i < (int)data.size(); ++i) {
    output->data[i] = std::exp(data[i]);
  }
  return output;
}

TensorPtr Tensor::log() {
  auto output = Tensor::zeros(shape, requires_grad, is_cuda);
  #pragma omp parallel for
  for (int i = 0; i < (int)data.size(); ++i) {
    output->data[i] = std::log(data[i] + 1e-8f);
  }
  return output;
}

TensorPtr Tensor::pow(float exponent) {
  auto output = Tensor::zeros(shape, requires_grad, is_cuda);
  #pragma omp parallel for
  for (int i = 0; i < (int)data.size(); ++i) {
    output->data[i] = std::pow(data[i], exponent);
  }
  return output;
}

TensorPtr Tensor::sqrt() {
  auto output = Tensor::zeros(shape, requires_grad, is_cuda);
  #pragma omp parallel for
  for (int i = 0; i < (int)data.size(); ++i) {
    output->data[i] = std::sqrt(data[i]);
  }
  return output;
}

TensorPtr Tensor::max(int dim, bool keepdim) {
  if (dim == -1) {
    float max_val = *std::max_element(data.begin(), data.end());
    return Tensor::from_data({max_val}, {1}, false, is_cuda);
  }
  throw std::runtime_error("Dimension-specific max not implemented yet");
}

TensorPtr Tensor::min(int dim, bool keepdim) {
  if (dim == -1) {
    float min_val = *std::min_element(data.begin(), data.end());
    return Tensor::from_data({min_val}, {1}, false, is_cuda);
  }
  throw std::runtime_error("Dimension-specific min not implemented yet");
}

TensorPtr Tensor::permute(const std::vector<int> &dims) {
  // Only support 4D permute for now (most common: NCHW <-> NHWC)
  if (dims.size() != shape.size()) {
    throw std::runtime_error("permute: dims size must match tensor dimensions");
  }

  std::vector<int> new_shape(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    new_shape[i] = shape[dims[i]];
  }

  auto output = Tensor::zeros(new_shape, requires_grad, is_cuda);

  // Generic permute using stride computation
  std::vector<int> new_strides(dims.size());
  int stride = 1;
  for (int i = (int)dims.size() - 1; i >= 0; --i) {
    new_strides[i] = stride;
    stride *= new_shape[i];
  }

  int total = (int)data.size();
  for (int flat = 0; flat < total; ++flat) {
    // Convert flat index to multi-dimensional indices
    int remaining = flat;
    std::vector<int> old_idx(shape.size());
    for (int d = 0; d < (int)shape.size(); ++d) {
      old_idx[d] = remaining / strides[d];
      remaining %= strides[d];
    }

    // Compute new flat index
    int new_flat = 0;
    for (size_t d = 0; d < dims.size(); ++d) {
      new_flat += old_idx[dims[d]] * new_strides[d];
    }

    output->data[new_flat] = data[flat];
  }

  return output;
}

// Autograd
void Tensor::backward(const TensorPtr &gradient) {
  if (!requires_grad)
    return;

  // Ensure grad buffer is allocated
  if (grad.size() != data.size()) {
    grad.resize(data.size(), 0.0f);
  }

  if (gradient) {
    for (size_t i = 0; i < grad.size(); ++i) {
      grad[i] += gradient->data[i];
    }
  } else {
    std::fill(grad.begin(), grad.end(), 1.0f);
  }

  if (grad_fn) {
    // Create a gradient tensor from this tensor's accumulated grad
    auto grad_tensor = Tensor::from_data(grad, shape, false, is_cuda);
    auto input_grads = grad_fn->backward(grad_tensor);
    for (size_t i = 0; i < input_grads.size(); ++i) {
      if (i < grad_fn->inputs.size() && grad_fn->inputs[i]->requires_grad) {
        grad_fn->inputs[i]->backward(input_grads[i]);
      }
    }
  }
}

void Tensor::zero_grad() { std::fill(grad.begin(), grad.end(), 0.0f); }

TensorPtr Tensor::detach() {
  return Tensor::from_data(data, shape, false, is_cuda);
}

// CUDA operations (stubs for now)
void Tensor::cuda() { is_cuda = true; }

void Tensor::cpu() { is_cuda = false; }

TensorPtr Tensor::to(bool cuda) {
  auto output = clone();
  output->is_cuda = cuda;
  return output;
}

// Utility
void Tensor::fill_(float value) { std::fill(data.begin(), data.end(), value); }

void Tensor::uniform_(float min, float max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min, max);
  for (auto &val : data) {
    val = dist(gen);
  }
}

void Tensor::normal_(float mean, float std) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::normal_distribution<float> dist(mean, std);
  for (auto &val : data) {
    val = dist(gen);
  }
}

TensorPtr Tensor::clone() {
  return Tensor::from_data(data, shape, requires_grad, is_cuda);
}

std::string Tensor::shape_str() const {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    ss << shape[i];
    if (i < shape.size() - 1)
      ss << ", ";
  }
  ss << "]";
  return ss.str();
}

void Tensor::print(const std::string &name) const {
  std::cout << name << " shape: " << shape_str() << std::endl;
  std::cout << "Data (first 10): ";
  for (size_t i = 0; i < std::min(size_t(10), data.size()); ++i) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

void Tensor::check_shape_compatible(const TensorPtr &other) const {
  if (shape != other->shape) {
    throw std::runtime_error("Shape mismatch: " + shape_str() + " vs " +
                             other->shape_str());
  }
}

float &Tensor::at(const std::vector<int> &indices) {
  return data[compute_offset(indices)];
}

const float &Tensor::at(const std::vector<int> &indices) const {
  return data[compute_offset(indices)];
}

int Tensor::compute_offset(const std::vector<int> &indices) const {
  int offset = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    offset += indices[i] * strides[i];
  }
  return offset;
}

} // namespace deepnet
