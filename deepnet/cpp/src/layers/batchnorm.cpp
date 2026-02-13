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
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::bernoulli_distribution dist(1.0 - p);

  // Inverted dropout
  float scale = 1.0f / (1.0f - p);
  for (size_t i = 0; i < output->data.size(); ++i) {
    if (!dist(gen)) {
      output->data[i] = 0.0f;
    } else {
      output->data[i] *= scale;
    }
  }

  return output;
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

  auto output =
      Tensor::zeros(input->shape, input->requires_grad, input->is_cuda);

  if (training) {
    // Compute mean and variance per channel
    std::vector<float> mean(channels, 0.0f);
    std::vector<float> var(channels, 0.0f);
    int spatial_size = height * width;
    int n = batch * spatial_size;

    // Compute mean
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

    // Normalize
    for (int c = 0; c < channels; ++c) {
      float std_inv = 1.0f / std::sqrt(var[c] + eps);
      for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            int idx = ((b * channels + c) * height + h) * width + w;
            float normalized = (input->data[idx] - mean[c]) * std_inv;
            output->data[idx] = gamma->data[c] * normalized + beta->data[c];
          }
        }
      }
    }
  } else {
    // Use running statistics for inference
    for (int c = 0; c < channels; ++c) {
      float std_inv = 1.0f / std::sqrt(running_var->data[c] + eps);
      for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            int idx = ((b * channels + c) * height + h) * width + w;
            float normalized =
                (input->data[idx] - running_mean->data[c]) * std_inv;
            output->data[idx] = gamma->data[c] * normalized + beta->data[c];
          }
        }
      }
    }
  }

  return output;
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

  auto output =
      Tensor::zeros(input->shape, input->requires_grad, input->is_cuda);

  if (training) {
    // Compute mean and variance per feature
    std::vector<float> mean(features, 0.0f);
    std::vector<float> var(features, 0.0f);

    // Compute mean
    for (int f = 0; f < features; ++f) {
      float sum = 0.0f;
      for (int b = 0; b < batch; ++b) {
        sum += input->data[b * features + f];
      }
      mean[f] = sum / batch;
    }

    // Compute variance
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

    // Normalize
    for (int f = 0; f < features; ++f) {
      float std_inv = 1.0f / std::sqrt(var[f] + eps);
      for (int b = 0; b < batch; ++b) {
        int idx = b * features + f;
        float normalized = (input->data[idx] - mean[f]) * std_inv;
        output->data[idx] = gamma->data[f] * normalized + beta->data[f];
      }
    }
  } else {
    // Use running statistics
    for (int f = 0; f < features; ++f) {
      float std_inv = 1.0f / std::sqrt(running_var->data[f] + eps);
      for (int b = 0; b < batch; ++b) {
        int idx = b * features + f;
        float normalized = (input->data[idx] - running_mean->data[f]) * std_inv;
        output->data[idx] = gamma->data[f] * normalized + beta->data[f];
      }
    }
  }

  return output;
}

std::vector<TensorPtr> BatchNorm1D::parameters() { return {gamma, beta}; }

} // namespace deepnet
