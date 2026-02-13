#include "layers/pooling.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>


namespace deepnet {

// MaxPool2D Implementation
MaxPool2D::MaxPool2D(int kernel_size, int stride)
    : kernel_size(kernel_size), stride(stride == -1 ? kernel_size : stride) {}

TensorPtr MaxPool2D::forward(const TensorPtr &input) {
  // Input shape: [batch, channels, height, width]
  if (input->shape.size() != 4) {
    throw std::runtime_error("MaxPool2D expects 4D input");
  }

  int batch = input->shape[0];
  int channels = input->shape[1];
  int in_h = input->shape[2];
  int in_w = input->shape[3];

  int out_h = (in_h - kernel_size) / stride + 1;
  int out_w = (in_w - kernel_size) / stride + 1;

  auto output = Tensor::zeros({batch, channels, out_h, out_w},
                              input->requires_grad, input->is_cuda);

  // Perform max pooling
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          float max_val = -std::numeric_limits<float>::infinity();

          // Find max in kernel window
          for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
              int ih = oh * stride + kh;
              int iw = ow * stride + kw;

              if (ih < in_h && iw < in_w) {
                int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                max_val = std::max(max_val, input->data[in_idx]);
              }
            }
          }

          int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
          output->data[out_idx] = max_val;
        }
      }
    }
  }

  return output;
}

// AvgPool2D Implementation
AvgPool2D::AvgPool2D(int kernel_size, int stride)
    : kernel_size(kernel_size), stride(stride == -1 ? kernel_size : stride) {}

TensorPtr AvgPool2D::forward(const TensorPtr &input) {
  // Input shape: [batch, channels, height, width]
  if (input->shape.size() != 4) {
    throw std::runtime_error("AvgPool2D expects 4D input");
  }

  int batch = input->shape[0];
  int channels = input->shape[1];
  int in_h = input->shape[2];
  int in_w = input->shape[3];

  int out_h = (in_h - kernel_size) / stride + 1;
  int out_w = (in_w - kernel_size) / stride + 1;

  auto output = Tensor::zeros({batch, channels, out_h, out_w},
                              input->requires_grad, input->is_cuda);

  // Perform avg pooling
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          float sum = 0.0f;
          int count = 0;

          // Sum all values in kernel window
          for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
              int ih = oh * stride + kh;
              int iw = ow * stride + kw;

              if (ih < in_h && iw < in_w) {
                int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                sum += input->data[in_idx];
                count++;
              }
            }
          }

          int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
          output->data[out_idx] = sum / count;
        }
      }
    }
  }

  return output;
}

// AdaptiveAvgPool2D Implementation
TensorPtr AdaptiveAvgPool2D::forward(const TensorPtr &input) {
  // Simple implementation: pool to output_size x output_size
  int batch = input->shape[0];
  int channels = input->shape[1];
  int in_h = input->shape[2];
  int in_w = input->shape[3];

  auto output = Tensor::zeros({batch, channels, output_size, output_size},
                              input->requires_grad, input->is_cuda);

  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int oh = 0; oh < output_size; ++oh) {
        for (int ow = 0; ow < output_size; ++ow) {
          int start_h = (oh * in_h) / output_size;
          int end_h = ((oh + 1) * in_h) / output_size;
          int start_w = (ow * in_w) / output_size;
          int end_w = ((ow + 1) * in_w) / output_size;

          float sum = 0.0f;
          int count = 0;

          for (int ih = start_h; ih < end_h; ++ih) {
            for (int iw = start_w; iw < end_w; ++iw) {
              int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
              sum += input->data[in_idx];
              count++;
            }
          }

          int out_idx =
              ((b * channels + c) * output_size + oh) * output_size + ow;
          output->data[out_idx] = sum / count;
        }
      }
    }
  }

  return output;
}

} // namespace deepnet
