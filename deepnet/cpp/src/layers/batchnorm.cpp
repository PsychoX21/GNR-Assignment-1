#include "layers/batchnorm.hpp"
#include <cmath>
#include <random>
#include <stdexcept>


namespace deepnet {

// Dropout Implementation
TensorPtr Dropout::forward(const TensorPtr &input) {
  if (!training || p == 0.0f) {
    return input;
  }

  auto output = input->clone();
  mask = Tensor::zeros(input->shape, false, input->is_cuda);
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::bernoulli_distribution dist(1.0 - p);

  // Inverted dropout
  float scale = 1.0f / (1.0f - p);
  for (size_t i = 0; i < output->data.size(); ++i) {
    if (!dist(gen)) {
      output->data[i] = 0.0f;
      mask->data[i] = 0.0f;
    } else {
      output->data[i] *= scale;
      mask->data[i] = scale;
    }
  }

  return output;
}

TensorPtr Dropout::backward(const TensorPtr &grad_output) {
  if (!training || p == 0.0f || !mask) {
    return grad_output;
  }

  auto grad_input = Tensor::zeros(grad_output->shape, false, grad_output->is_cuda);
  for (size_t i = 0; i < grad_input->data.size(); ++i) {
    grad_input->data[i] = grad_output->data[i] * mask->data[i];
  }
  return grad_input;
}

// BatchNorm2D Implementation
BatchNorm2D::BatchNorm2D(int num_features, float eps, float momentum)
    : num_features(num_features), eps(eps), momentum(momentum) {

  gamma = Tensor::ones({num_features}, true, false);
  beta = Tensor::zeros({num_features}, true, false);
  running_mean = Tensor::zeros({num_features}, false, false);
  running_var = Tensor::ones({num_features}, false, false);
}

TensorPtr BatchNorm2D::forward(const TensorPtr &input) {
  // Input shape: [batch, channels, height, width]
  if (input->shape.size() != 4) {
    throw std::runtime_error("BatchNorm2D expects 4D input");
  }

  int batch = input->shape[0];
  int channels = input->shape[1];
  int height = input->shape[2];
  int width = input->shape[3];

  if (channels != num_features) {
    throw std::runtime_error("BatchNorm2D channel mismatch");
  }

  // Cache input for backward
  last_input = input;
  normalized = Tensor::zeros(input->shape, false, input->is_cuda);
  batch_std_inv.resize(channels);

  auto output =
      Tensor::zeros(input->shape, true, input->is_cuda);

  if (training) {
    // Compute mean and variance per channel
    std::vector<float> mean(channels, 0.0f);
    std::vector<float> var(channels, 0.0f);
    int spatial_size = height * width;
    int n = batch * spatial_size;

    // Compute mean
    #pragma omp parallel for
    for (int c = 0; c < channels; ++c) {
      float sum = 0.0f;
      for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            int idx = ((b * channels + c) * height + h) * width + w;
            sum += input->data[idx];
          }
        }
      }
      mean[c] = sum / n;
    }

    // Compute variance
    #pragma omp parallel for
    for (int c = 0; c < channels; ++c) {
      float sum_sq = 0.0f;
      for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            int idx = ((b * channels + c) * height + h) * width + w;
            float diff = input->data[idx] - mean[c];
            sum_sq += diff * diff;
          }
        }
      }
      var[c] = sum_sq / n;
    }

    // Update running statistics
    for (int c = 0; c < channels; ++c) {
      running_mean->data[c] =
          (1.0f - momentum) * running_mean->data[c] + momentum * mean[c];
      running_var->data[c] =
          (1.0f - momentum) * running_var->data[c] + momentum * var[c];
    }

    // Normalize and cache
    #pragma omp parallel for
    for (int c = 0; c < channels; ++c) {
      float std_inv = 1.0f / std::sqrt(var[c] + eps);
      batch_std_inv[c] = std_inv;
      for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            int idx = ((b * channels + c) * height + h) * width + w;
            float x_hat = (input->data[idx] - mean[c]) * std_inv;
            normalized->data[idx] = x_hat;
            output->data[idx] = gamma->data[c] * x_hat + beta->data[c];
          }
        }
      }
    }
  } else {
    // Use running statistics for inference
    for (int c = 0; c < channels; ++c) {
      float std_inv = 1.0f / std::sqrt(running_var->data[c] + eps);
      batch_std_inv[c] = std_inv;
      for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            int idx = ((b * channels + c) * height + h) * width + w;
            float x_hat = (input->data[idx] - running_mean->data[c]) * std_inv;
            normalized->data[idx] = x_hat;
            output->data[idx] = gamma->data[c] * x_hat + beta->data[c];
          }
        }
      }
    }
  }

  return output;
}

TensorPtr BatchNorm2D::backward(const TensorPtr &grad_output) {
  int batch = last_input->shape[0];
  int channels = last_input->shape[1];
  int height = last_input->shape[2];
  int width = last_input->shape[3];
  int spatial = height * width;
  int n = batch * spatial;

  // Ensure grad buffers
  if (gamma->grad.size() != gamma->data.size())
    gamma->grad.resize(gamma->data.size(), 0.0f);
  if (beta->grad.size() != beta->data.size())
    beta->grad.resize(beta->data.size(), 0.0f);

  auto grad_input = Tensor::zeros(last_input->shape, false, last_input->is_cuda);

  #pragma omp parallel for
  for (int c = 0; c < channels; ++c) {
    float g = gamma->data[c];
    float si = batch_std_inv[c];

    // Accumulate grad_gamma and grad_beta
    float dg = 0.0f, db = 0.0f;
    float sum_dy = 0.0f, sum_dy_xhat = 0.0f;
    for (int b = 0; b < batch; ++b) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int idx = ((b * channels + c) * height + h) * width + w;
          float dy = grad_output->data[idx];
          float xh = normalized->data[idx];
          dg += dy * xh;
          db += dy;
          sum_dy += dy;
          sum_dy_xhat += dy * xh;
        }
      }
    }
    gamma->grad[c] += dg;
    beta->grad[c] += db;

    // Compute grad_input
    for (int b = 0; b < batch; ++b) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int idx = ((b * channels + c) * height + h) * width + w;
          float dy = grad_output->data[idx];
          float xh = normalized->data[idx];
          grad_input->data[idx] = g * si / n *
              (n * dy - sum_dy - xh * sum_dy_xhat);
        }
      }
    }
  }

  return grad_input;
}

std::vector<TensorPtr> BatchNorm2D::parameters() { return {gamma, beta}; }

// BatchNorm1D Implementation
BatchNorm1D::BatchNorm1D(int num_features, float eps, float momentum)
    : num_features(num_features), eps(eps), momentum(momentum) {

  gamma = Tensor::ones({num_features}, true, false);
  beta = Tensor::zeros({num_features}, true, false);
  running_mean = Tensor::zeros({num_features}, false, false);
  running_var = Tensor::ones({num_features}, false, false);
}

TensorPtr BatchNorm1D::forward(const TensorPtr &input) {
  // Input shape: [batch, features]
  if (input->shape.size() != 2) {
    throw std::runtime_error("BatchNorm1D expects 2D input");
  }

  int batch = input->shape[0];
  int features = input->shape[1];

  if (features != num_features) {
    throw std::runtime_error("BatchNorm1D feature mismatch");
  }

  // Cache for backward
  last_input = input;
  normalized = Tensor::zeros(input->shape, false, input->is_cuda);
  batch_std_inv.resize(features);

  auto output =
      Tensor::zeros(input->shape, true, input->is_cuda);

  if (training) {
    std::vector<float> mean(features, 0.0f);
    std::vector<float> var(features, 0.0f);

    // Compute mean
    #pragma omp parallel for
    for (int f = 0; f < features; ++f) {
      float sum = 0.0f;
      for (int b = 0; b < batch; ++b) {
        sum += input->data[b * features + f];
      }
      mean[f] = sum / batch;
    }

    // Compute variance
    #pragma omp parallel for
    for (int f = 0; f < features; ++f) {
      float sum_sq = 0.0f;
      for (int b = 0; b < batch; ++b) {
        float diff = input->data[b * features + f] - mean[f];
        sum_sq += diff * diff;
      }
      var[f] = sum_sq / batch;
    }

    // Update running statistics
    for (int f = 0; f < features; ++f) {
      running_mean->data[f] =
          (1.0f - momentum) * running_mean->data[f] + momentum * mean[f];
      running_var->data[f] =
          (1.0f - momentum) * running_var->data[f] + momentum * var[f];
    }

    // Normalize and cache
    #pragma omp parallel for
    for (int f = 0; f < features; ++f) {
      float std_inv = 1.0f / std::sqrt(var[f] + eps);
      batch_std_inv[f] = std_inv;
      for (int b = 0; b < batch; ++b) {
        int idx = b * features + f;
        float x_hat = (input->data[idx] - mean[f]) * std_inv;
        normalized->data[idx] = x_hat;
        output->data[idx] = gamma->data[f] * x_hat + beta->data[f];
      }
    }
  } else {
    // Use running statistics
    for (int f = 0; f < features; ++f) {
      float std_inv = 1.0f / std::sqrt(running_var->data[f] + eps);
      batch_std_inv[f] = std_inv;
      for (int b = 0; b < batch; ++b) {
        int idx = b * features + f;
        float x_hat = (input->data[idx] - running_mean->data[f]) * std_inv;
        normalized->data[idx] = x_hat;
        output->data[idx] = gamma->data[f] * x_hat + beta->data[f];
      }
    }
  }

  return output;
}

TensorPtr BatchNorm1D::backward(const TensorPtr &grad_output) {
  int batch = last_input->shape[0];
  int features = last_input->shape[1];

  // Ensure grad buffers
  if (gamma->grad.size() != gamma->data.size())
    gamma->grad.resize(gamma->data.size(), 0.0f);
  if (beta->grad.size() != beta->data.size())
    beta->grad.resize(beta->data.size(), 0.0f);

  auto grad_input = Tensor::zeros(last_input->shape, false, last_input->is_cuda);

  #pragma omp parallel for
  for (int f = 0; f < features; ++f) {
    float g = gamma->data[f];
    float si = batch_std_inv[f];

    float dg = 0.0f, db = 0.0f;
    float sum_dy = 0.0f, sum_dy_xhat = 0.0f;
    for (int b = 0; b < batch; ++b) {
      int idx = b * features + f;
      float dy = grad_output->data[idx];
      float xh = normalized->data[idx];
      dg += dy * xh;
      db += dy;
      sum_dy += dy;
      sum_dy_xhat += dy * xh;
    }
    gamma->grad[f] += dg;
    beta->grad[f] += db;

    for (int b = 0; b < batch; ++b) {
      int idx = b * features + f;
      float dy = grad_output->data[idx];
      float xh = normalized->data[idx];
      grad_input->data[idx] = g * si / batch *
          (batch * dy - sum_dy - xh * sum_dy_xhat);
    }
  }

  return grad_input;
}

std::vector<TensorPtr> BatchNorm1D::parameters() { return {gamma, beta}; }

} // namespace deepnet
