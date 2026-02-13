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
  int weight_size = out_channels * in_channels * kernel_size * kernel_size;
  float std = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));

  weight = Tensor::randn({out_channels, in_channels, kernel_size, kernel_size},
                         true, false, 0.0f, std);

  if (use_bias) {
    bias_ = Tensor::zeros({out_channels}, true, false);
  }
}

TensorPtr Conv2D::forward(const TensorPtr &input) {
  // Input shape: [batch, in_channels, height, width]
  if (input->shape.size() != 4) {
    throw std::runtime_error("Conv2D expects 4D input");
  }

  int batch = input->shape[0];
  int in_h = input->shape[2];
  int in_w = input->shape[3];

  // Calculate output dimensions
  int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
  int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

  // Initialize output
  auto output = Tensor::zeros({batch, out_channels, out_h, out_w},
                              input->requires_grad, input->is_cuda);

  // Perform convolution (naive implementation for now)
  for (int b = 0; b < batch; ++b) {
    for (int oc = 0; oc < out_channels; ++oc) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          float sum = 0.0f;

          // Convolve with kernel
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
  weight = Tensor::randn({out_features, in_features}, true, false, 0.0f, limit);

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

  // output = input @ weight^T + bias
  auto output = x->matmul(weight->transpose(0, 1));

  if (use_bias) {
    // Broadcast bias
    for (int i = 0; i < output->shape[0]; ++i) {
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

std::vector<TensorPtr> Linear::parameters() {
  if (use_bias) {
    return {weight, bias_};
  }
  return {weight};
}

// ReLU Implementation
TensorPtr ReLU::forward(const TensorPtr &input) { return input->relu(); }

// LeakyReLU Implementation
TensorPtr LeakyReLU::forward(const TensorPtr &input) {
  return input->leaky_relu(negative_slope);
}

// Tanh Implementation
TensorPtr Tanh::forward(const TensorPtr &input) { return input->tanh_(); }

// Sigmoid Implementation
TensorPtr Sigmoid::forward(const TensorPtr &input) { return input->sigmoid(); }

// Flatten Implementation
TensorPtr Flatten::forward(const TensorPtr &input) {
  return input->flatten(start_dim, end_dim);
}

} // namespace deepnet
