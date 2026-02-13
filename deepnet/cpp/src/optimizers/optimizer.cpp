#include "optimizers/optimizer.hpp"
#include <cmath>

namespace deepnet {

// Base Optimizer
void Optimizer::zero_grad() {
  for (auto &param : parameters) {
    if (param->requires_grad) {
      param->zero_grad();
    }
  }
}

void Optimizer::add_parameters(const std::vector<TensorPtr> &params) {
  parameters.insert(parameters.end(), params.begin(), params.end());
}

// SGD Implementation
SGD::SGD(const std::vector<TensorPtr> &params, float lr, float momentum,
         float weight_decay, bool nesterov)
    : lr(lr), momentum(momentum), weight_decay(weight_decay),
      nesterov(nesterov) {
  parameters = params;

  if (momentum > 0.0f) {
    for (const auto &param : parameters) {
      velocity.push_back(Tensor::zeros(param->shape, false, param->is_cuda));
    }
  }
}

void SGD::step() {
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto &param = parameters[i];
    if (!param->requires_grad)
      continue;

    // Add weight decay
    if (weight_decay > 0.0f) {
      for (size_t j = 0; j < param->grad.size(); ++j) {
        param->grad[j] += weight_decay * param->data[j];
      }
    }

    if (momentum > 0.0f) {
      // Update velocity: v = momentum * v + grad
      for (size_t j = 0; j < param->data.size(); ++j) {
        velocity[i]->data[j] = momentum * velocity[i]->data[j] + param->grad[j];
      }

      if (nesterov) {
        // Nesterov: param -= lr * (momentum * v + grad)
        for (size_t j = 0; j < param->data.size(); ++j) {
          param->data[j] -=
              lr * (momentum * velocity[i]->data[j] + param->grad[j]);
        }
      } else {
        // Standard momentum: param -= lr * v
        for (size_t j = 0; j < param->data.size(); ++j) {
          param->data[j] -= lr * velocity[i]->data[j];
        }
      }
    } else {
      // Standard SGD: param -= lr * grad
      for (size_t j = 0; j < param->data.size(); ++j) {
        param->data[j] -= lr * param->grad[j];
      }
    }
  }
}

// Adam Implementation
Adam::Adam(const std::vector<TensorPtr> &params, float lr, float beta1,
           float beta2, float eps, float weight_decay)
    : lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay),
      t(0) {
  parameters = params;

  for (const auto &param : parameters) {
    m.push_back(Tensor::zeros(param->shape, false, param->is_cuda));
    v.push_back(Tensor::zeros(param->shape, false, param->is_cuda));
  }
}

void Adam::step() {
  t++;

  for (size_t i = 0; i < parameters.size(); ++i) {
    auto &param = parameters[i];
    if (!param->requires_grad)
      continue;

    // Add weight decay
    if (weight_decay > 0.0f) {
      for (size_t j = 0; j < param->grad.size(); ++j) {
        param->grad[j] += weight_decay * param->data[j];
      }
    }

    // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
    for (size_t j = 0; j < param->data.size(); ++j) {
      m[i]->data[j] = beta1 * m[i]->data[j] + (1.0f - beta1) * param->grad[j];
    }

    // Update biased second moment estimate: v = beta2 * v + (1 - beta2) *
    // grad^2
    for (size_t j = 0; j < param->data.size(); ++j) {
      v[i]->data[j] = beta2 * v[i]->data[j] +
                      (1.0f - beta2) * param->grad[j] * param->grad[j];
    }

    // Compute bias correction
    float m_hat_scale = 1.0f / (1.0f - static_cast<float>(std::pow(beta1, t)));
    float v_hat_scale = 1.0f / (1.0f - static_cast<float>(std::pow(beta2, t)));

    // Update parameters: param -= lr * m_hat / (sqrt(v_hat) + eps)
    for (size_t j = 0; j < param->data.size(); ++j) {
      float m_hat = m[i]->data[j] * m_hat_scale;
      float v_hat = v[i]->data[j] * v_hat_scale;
      param->data[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
  }
}

} // namespace deepnet
