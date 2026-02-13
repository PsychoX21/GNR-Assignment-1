#include "layers/pooling.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace deepnet {

// MaxPool2D Implementation
MaxPool2D::MaxPool2D(int kernel_size, int stride)
    : kernel_size(kernel_size), stride(stride == -1 ? kernel_size : stride) {}

TensorPtr MaxPool2D::forward(const TensorPtr &input) {
  if (input->shape.size() != 4) {
    throw std::runtime_error("MaxPool2D expects 4D input");
  }

  int batch = input->shape[0];
  int channels = input->shape[1];
  int in_h = input->shape[2];
  int in_w = input->shape[3];

  int out_h = (in_h - kernel_size) / stride + 1;
  int out_w = (in_w - kernel_size) / stride + 1;

  // Cache input shape for backward
  input_shape = input->shape;

  auto output = Tensor::zeros({batch, channels, out_h, out_w},
                              true, input->is_cuda);

  // Clear and resize max_indices
  max_indices.resize(batch * channels * out_h * out_w);

  #pragma omp parallel for
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          float max_val = -std::numeric_limits<float>::infinity();
          int max_idx = 0;

          for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
              int ih = oh * stride + kh;
              int iw = ow * stride + kw;
              int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;

              if (input->data[in_idx] > max_val) {
                max_val = input->data[in_idx];
                max_idx = in_idx;
              }
            }
          }

          int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
          output->data[out_idx] = max_val;
          max_indices[out_idx] = max_idx;
        }
      }
    }
  }

  return output;
}

TensorPtr MaxPool2D::backward(const TensorPtr &grad_output) {
  // Route gradients to the max elements
  auto grad_input = Tensor::zeros(input_shape, false, grad_output->is_cuda);

  for (size_t i = 0; i < max_indices.size(); ++i) {
    grad_input->data[max_indices[i]] += grad_output->data[i];
  }

  return grad_input;
}

// AvgPool2D Implementation
AvgPool2D::AvgPool2D(int kernel_size, int stride)
    : kernel_size(kernel_size), stride(stride == -1 ? kernel_size : stride) {}

TensorPtr AvgPool2D::forward(const TensorPtr &input) {
  if (input->shape.size() != 4) {
    throw std::runtime_error("AvgPool2D expects 4D input");
  }

  int batch = input->shape[0];
  int channels = input->shape[1];
  int in_h = input->shape[2];
  int in_w = input->shape[3];

  int out_h = (in_h - kernel_size) / stride + 1;
  int out_w = (in_w - kernel_size) / stride + 1;

  input_shape = input->shape;

  auto output = Tensor::zeros({batch, channels, out_h, out_w},
                              true, input->is_cuda);

  float pool_size = static_cast<float>(kernel_size * kernel_size);

  #pragma omp parallel for
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          float sum = 0.0f;

          for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
              int ih = oh * stride + kh;
              int iw = ow * stride + kw;
              int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
              sum += input->data[in_idx];
            }
          }

          int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
          output->data[out_idx] = sum / pool_size;
        }
      }
    }
  }

  return output;
}

TensorPtr AvgPool2D::backward(const TensorPtr &grad_output) {
  int batch = input_shape[0];
  int channels = input_shape[1];
  int in_h = input_shape[2];
  int in_w = input_shape[3];

  int out_h = grad_output->shape[2];
  int out_w = grad_output->shape[3];

  auto grad_input = Tensor::zeros(input_shape, false, grad_output->is_cuda);

  float pool_size = static_cast<float>(kernel_size * kernel_size);

  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
          float grad_val = grad_output->data[out_idx] / pool_size;

          for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
              int ih = oh * stride + kh;
              int iw = ow * stride + kw;
              int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
              grad_input->data[in_idx] += grad_val;
            }
          }
        }
      }
    }
  }

  return grad_input;
}

} // namespace deepnet
