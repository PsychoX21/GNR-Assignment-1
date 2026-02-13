#pragma once

#include "../tensor.hpp"
#include "layer.hpp"

namespace deepnet {

// Dropout Layer
class Dropout : public Layer {
public:
  Dropout(float p = 0.5f) : p(p) {}

  TensorPtr forward(const TensorPtr &input) override;
  TensorPtr backward(const TensorPtr &grad_output) override;

private:
  float p; // Dropout probability
  TensorPtr mask; // Cached dropout mask for backward
};

// BatchNorm2D Layer
class BatchNorm2D : public Layer {
public:
  BatchNorm2D(int num_features, float eps = 1e-5f, float momentum = 0.1f);

  TensorPtr forward(const TensorPtr &input) override;
  TensorPtr backward(const TensorPtr &grad_output) override;
  std::vector<TensorPtr> parameters() override;

private:
  int num_features;
  float eps, momentum;

  TensorPtr gamma; // Scale parameter
  TensorPtr beta;  // Shift parameter
  TensorPtr running_mean;
  TensorPtr running_var;
  TensorPtr last_input;   // Cached for backward
  TensorPtr normalized;   // Cached x_hat for backward
  std::vector<float> batch_std_inv; // Cached 1/sqrt(var+eps)
};

// BatchNorm1D Layer (for Linear layers)
class BatchNorm1D : public Layer {
public:
  BatchNorm1D(int num_features, float eps = 1e-5f, float momentum = 0.1f);

  TensorPtr forward(const TensorPtr &input) override;
  TensorPtr backward(const TensorPtr &grad_output) override;
  std::vector<TensorPtr> parameters() override;

private:
  int num_features;
  float eps, momentum;

  TensorPtr gamma;
  TensorPtr beta;
  TensorPtr running_mean;
  TensorPtr running_var;
  TensorPtr last_input;
  TensorPtr normalized;
  std::vector<float> batch_std_inv;
};

} // namespace deepnet
