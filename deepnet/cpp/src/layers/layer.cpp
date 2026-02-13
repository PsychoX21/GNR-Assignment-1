#include "layers/layer.hpp"
#include <cmath>
#include <stdexcept>

namespace deepnet {

// Conv2D Implementation
Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size, int stride,
               int padding, bool bias)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_size(kernel_size), stride(stride), padding(padding),
      use_bias(bias) {

  // Initialize weights with He initialization
  float std_val = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));

  weight = Tensor::randn({out_channels, in_channels, kernel_size, kernel_size},
                         0.0f, std_val, true, false);

  if (use_bias) {
    bias_ = Tensor::zeros({out_channels}, true, false);
  }
}

TensorPtr Conv2D::forward(const TensorPtr &input) {
  // Input shape: [batch, in_channels, height, width]
  if (input->shape.size() != 4) {
    throw std::runtime_error("Conv2D expects 4D input");
  }

  // Cache input for backward
  last_input = input;

  int batch = input->shape[0];
  int in_h = input->shape[2];
  int in_w = input->shape[3];

  // Calculate output dimensions
  int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
  int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

  // Initialize output
  auto output = Tensor::zeros({batch, out_channels, out_h, out_w},
                              true, input->is_cuda);

  // Perform convolution (naive implementation)
  #pragma omp parallel for
  for (int b = 0; b < batch; ++b) {
    for (int oc = 0; oc < out_channels; ++oc) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          float sum = 0.0f;

          for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
              for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;

                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                  int in_idx = ((b * in_channels + ic) * in_h + ih) * in_w + iw;
                  int w_idx = ((oc * in_channels + ic) * kernel_size + kh) *
                                  kernel_size +
                              kw;
                  sum += input->data[in_idx] * weight->data[w_idx];
                }
              }
            }
          }

          if (use_bias) {
            sum += bias_->data[oc];
          }

          int out_idx = ((b * out_channels + oc) * out_h + oh) * out_w + ow;
          output->data[out_idx] = sum;
        }
      }
    }
  }

  return output;
}

TensorPtr Conv2D::backward(const TensorPtr &grad_output) {
  // grad_output shape: [batch, out_channels, out_h, out_w]
  if (!last_input) {
    throw std::runtime_error("Conv2D::backward called without forward");
  }

  int batch = last_input->shape[0];
  int in_h = last_input->shape[2];
  int in_w = last_input->shape[3];
  int out_h = grad_output->shape[2];
  int out_w = grad_output->shape[3];

  // Ensure grad buffers are allocated
  if (weight->grad.size() != weight->data.size()) {
    weight->grad.resize(weight->data.size(), 0.0f);
  }
  if (use_bias && bias_->grad.size() != bias_->data.size()) {
    bias_->grad.resize(bias_->data.size(), 0.0f);
  }

  // Gradient w.r.t. input
  auto grad_input = Tensor::zeros(last_input->shape, false, last_input->is_cuda);

  // Compute gradients
  for (int b = 0; b < batch; ++b) {
    for (int oc = 0; oc < out_channels; ++oc) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          int out_idx = ((b * out_channels + oc) * out_h + oh) * out_w + ow;
          float grad_val = grad_output->data[out_idx];

          // Gradient w.r.t. bias
          if (use_bias) {
            bias_->grad[oc] += grad_val;
          }

          for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
              for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;

                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                  int in_idx = ((b * in_channels + ic) * in_h + ih) * in_w + iw;
                  int w_idx = ((oc * in_channels + ic) * kernel_size + kh) *
                                  kernel_size + kw;

                  // Gradient w.r.t. weight
                  weight->grad[w_idx] += grad_val * last_input->data[in_idx];

                  // Gradient w.r.t. input
                  grad_input->data[in_idx] += grad_val * weight->data[w_idx];
                }
              }
            }
          }
        }
      }
    }
  }

  return grad_input;
}

std::vector<TensorPtr> Conv2D::parameters() {
  if (use_bias) {
    return {weight, bias_};
  }
  return {weight};
}

// Linear Implementation
Linear::Linear(int in_features, int out_features, bool bias)
    : in_features(in_features), out_features(out_features), use_bias(bias) {

  // Xavier initialization
  float limit = std::sqrt(6.0f / (in_features + out_features));
  weight = Tensor::randn({out_features, in_features}, 0.0f, limit, true, false);

  if (use_bias) {
    bias_ = Tensor::zeros({out_features}, true, false);
  }
}

TensorPtr Linear::forward(const TensorPtr &input) {
  // Input can be 2D [batch, in_features] or 1D [in_features]
  TensorPtr x = input;
  bool is_1d = (input->shape.size() == 1);

  if (is_1d) {
    x = input->reshape({1, input->shape[0]});
  }

  if (x->shape.size() != 2 || x->shape[1] != in_features) {
    throw std::runtime_error("Linear layer input size mismatch");
  }

  // Cache input for backward (always as 2D)
  last_input = x;

  int batch = x->shape[0];

  // Manual matmul: output = x @ weight^T
  auto output = Tensor::zeros({batch, out_features}, true, x->is_cuda);
  #pragma omp parallel for
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < out_features; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < in_features; ++k) {
        sum += x->data[i * in_features + k] * weight->data[j * in_features + k];
      }
      output->data[i * out_features + j] = sum;
    }
  }

  if (use_bias) {
    // Add bias
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < out_features; ++j) {
        output->data[i * out_features + j] += bias_->data[j];
      }
    }
  }

  if (is_1d) {
    output = output->reshape({out_features});
  }

  return output;
}

TensorPtr Linear::backward(const TensorPtr &grad_output) {
  // grad_output shape: [batch, out_features]
  if (!last_input) {
    throw std::runtime_error("Linear::backward called without forward");
  }

  int batch = last_input->shape[0];

  // Handle 1D grad_output
  TensorPtr grad = grad_output;
  if (grad->shape.size() == 1) {
    grad = grad->reshape({1, (int)grad->data.size()});
  }

  // Ensure grad buffers are allocated
  if (weight->grad.size() != weight->data.size()) {
    weight->grad.resize(weight->data.size(), 0.0f);
  }
  if (use_bias && bias_->grad.size() != bias_->data.size()) {
    bias_->grad.resize(bias_->data.size(), 0.0f);
  }

  // Gradient w.r.t. weight: grad_weight += grad_output^T @ input
  // grad_output: [batch, out_features], input: [batch, in_features]
  // grad_weight: [out_features, in_features]
  #pragma omp parallel for
  for (int j = 0; j < out_features; ++j) {
    for (int k = 0; k < in_features; ++k) {
      float sum = 0.0f;
      for (int i = 0; i < batch; ++i) {
        sum += grad->data[i * out_features + j] * last_input->data[i * in_features + k];
      }
      weight->grad[j * in_features + k] += sum;
    }
  }

  // Gradient w.r.t. bias: grad_bias += sum over batch of grad_output
  if (use_bias) {
    for (int j = 0; j < out_features; ++j) {
      float sum = 0.0f;
      for (int i = 0; i < batch; ++i) {
        sum += grad->data[i * out_features + j];
      }
      bias_->grad[j] += sum;
    }
  }

  // Gradient w.r.t. input: grad_input = grad_output @ weight
  // grad_output: [batch, out_features], weight: [out_features, in_features]
  // grad_input: [batch, in_features]
  auto grad_input = Tensor::zeros({batch, in_features}, false, last_input->is_cuda);
  #pragma omp parallel for
  for (int i = 0; i < batch; ++i) {
    for (int k = 0; k < in_features; ++k) {
      float sum = 0.0f;
      for (int j = 0; j < out_features; ++j) {
        sum += grad->data[i * out_features + j] * weight->data[j * in_features + k];
      }
      grad_input->data[i * in_features + k] = sum;
    }
  }

  return grad_input;
}

std::vector<TensorPtr> Linear::parameters() {
  if (use_bias) {
    return {weight, bias_};
  }
  return {weight};
}

// ReLU Implementation
TensorPtr ReLU::forward(const TensorPtr &input) {
  last_input = input;
  return input->relu();
}

TensorPtr ReLU::backward(const TensorPtr &grad_output) {
  auto grad_input = Tensor::zeros(last_input->shape, false, grad_output->is_cuda);
  #pragma omp parallel for
  for (int i = 0; i < (int)grad_input->data.size(); ++i) {
    grad_input->data[i] = last_input->data[i] > 0 ? grad_output->data[i] : 0.0f;
  }
  return grad_input;
}

// LeakyReLU Implementation
TensorPtr LeakyReLU::forward(const TensorPtr &input) {
  last_input = input;
  return input->leaky_relu(negative_slope);
}

TensorPtr LeakyReLU::backward(const TensorPtr &grad_output) {
  auto grad_input = Tensor::zeros(last_input->shape, false, grad_output->is_cuda);
  #pragma omp parallel for
  for (int i = 0; i < (int)grad_input->data.size(); ++i) {
    grad_input->data[i] = last_input->data[i] > 0 ? grad_output->data[i]
                                                    : negative_slope * grad_output->data[i];
  }
  return grad_input;
}

// Tanh Implementation
TensorPtr Tanh::forward(const TensorPtr &input) {
  last_output = input->tanh_();
  return last_output;
}

TensorPtr Tanh::backward(const TensorPtr &grad_output) {
  // d/dx tanh(x) = 1 - tanh(x)^2
  auto grad_input = Tensor::zeros(last_output->shape, false, grad_output->is_cuda);
  #pragma omp parallel for
  for (int i = 0; i < (int)grad_input->data.size(); ++i) {
    float t = last_output->data[i];
    grad_input->data[i] = grad_output->data[i] * (1.0f - t * t);
  }
  return grad_input;
}

// Sigmoid Implementation
TensorPtr Sigmoid::forward(const TensorPtr &input) {
  last_output = input->sigmoid();
  return last_output;
}

TensorPtr Sigmoid::backward(const TensorPtr &grad_output) {
  // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
  auto grad_input = Tensor::zeros(last_output->shape, false, grad_output->is_cuda);
  #pragma omp parallel for
  for (int i = 0; i < (int)grad_input->data.size(); ++i) {
    float s = last_output->data[i];
    grad_input->data[i] = grad_output->data[i] * s * (1.0f - s);
  }
  return grad_input;
}

// Flatten Implementation
TensorPtr Flatten::forward(const TensorPtr &input) {
  original_shape = input->shape;
  return input->flatten(start_dim, end_dim);
}

TensorPtr Flatten::backward(const TensorPtr &grad_output) {
  // Reshape gradient back to original shape
  return grad_output->reshape(original_shape);
}

} // namespace deepnet
