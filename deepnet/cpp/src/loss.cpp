#include "loss.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>


namespace deepnet {

// Softmax helper
TensorPtr CrossEntropyLoss::softmax(const TensorPtr &input) {
  // Input shape: [batch, num_classes]
  int batch = input->shape[0];
  int num_classes = input->shape[1];

  auto output = Tensor::zeros(input->shape, false, input->is_cuda);

  for (int b = 0; b < batch; ++b) {
    // Find max for numerical stability
    float max_val = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < num_classes; ++c) {
      max_val = std::max(max_val, input->data[b * num_classes + c]);
    }

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
      float exp_val = std::exp(input->data[b * num_classes + c] - max_val);
      output->data[b * num_classes + c] = exp_val;
      sum_exp += exp_val;
    }

    // Normalize
    for (int c = 0; c < num_classes; ++c) {
      output->data[b * num_classes + c] /= sum_exp;
    }
  }

  return output;
}

// Log-Softmax helper (more numerically stable)
TensorPtr CrossEntropyLoss::log_softmax(const TensorPtr &input) {
  // Input shape: [batch, num_classes]
  int batch = input->shape[0];
  int num_classes = input->shape[1];

  auto output = Tensor::zeros(input->shape, false, input->is_cuda);

  for (int b = 0; b < batch; ++b) {
    // Find max for numerical stability
    float max_val = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < num_classes; ++c) {
      max_val = std::max(max_val, input->data[b * num_classes + c]);
    }

    // Compute log-sum-exp
    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
      sum_exp += std::exp(input->data[b * num_classes + c] - max_val);
    }
    float log_sum_exp = max_val + std::log(sum_exp);

    // Compute log probabilities
    for (int c = 0; c < num_classes; ++c) {
      output->data[b * num_classes + c] =
          input->data[b * num_classes + c] - log_sum_exp;
    }
  }

  return output;
}

// Cross Entropy Loss
TensorPtr CrossEntropyLoss::forward(const TensorPtr &input,
                                    const std::vector<int> &targets) {
  if (input->shape.size() != 2) {
    throw std::runtime_error(
        "CrossEntropyLoss expects 2D input [batch, num_classes]");
  }

  int batch = input->shape[0];
  int num_classes = input->shape[1];

  if (static_cast<int>(targets.size()) != batch) {
    throw std::runtime_error("Target size mismatch");
  }

  // Compute log-softmax
  auto log_probs = log_softmax(input);

  // Compute negative log likelihood
  float total_loss = 0.0f;
  for (int b = 0; b < batch; ++b) {
    int target_class = targets[b];
    if (target_class < 0 || target_class >= num_classes) {
      throw std::runtime_error("Target class out of range");
    }
    total_loss -= log_probs->data[b * num_classes + target_class];
  }

  // Return mean loss
  float mean_loss = total_loss / batch;
  return Tensor::from_data({mean_loss}, {1}, false, input->is_cuda);
}

// MSE Loss
TensorPtr MSELoss::forward(const TensorPtr &input, const TensorPtr &target) {
  if (input->shape != target->shape) {
    throw std::runtime_error("MSELoss: input and target shapes must match");
  }

  float sum_sq_error = 0.0f;
  for (size_t i = 0; i < input->data.size(); ++i) {
    float diff = input->data[i] - target->data[i];
    sum_sq_error += diff * diff;
  }

  float mean_loss = sum_sq_error / input->data.size();
  return Tensor::from_data({mean_loss}, {1}, false, input->is_cuda);
}

} // namespace deepnet
