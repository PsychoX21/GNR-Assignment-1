#pragma once

#include "../tensor.hpp"
#include "layer.hpp"

namespace deepnet {

// Dropout Layer
class Dropout : public Layer {
public:
  Dropout(float p = 0.5f) : p(p) {}

  TensorPtr forward(const TensorPtr &input) override;

private:
  float p; // Dropout probability
};

// BatchNorm2D Layer
class BatchNorm2D : public Layer {
public:
  BatchNorm2D(int num_features, float eps = 1e-5f, float momentum = 0.1f);

  TensorPtr forward(const TensorPtr &input) override;
  std::vector<TensorPtr> parameters() override;

private:
  int num_features;
  float eps, momentum;

  TensorPtr gamma; // Scale parameter
  TensorPtr beta;  // Shift parameter
  TensorPtr running_mean;
  TensorPtr running_var;
};

// BatchNorm1D Layer (for Linear layers)
class BatchNorm1D : public Layer {
public:
  BatchNorm1D(int num_features, float eps = 1e-5f, float momentum = 0.1f);

  TensorPtr forward(const TensorPtr &input) override;
  std::vector<TensorPtr> parameters() override;

private:
  int num_features;
  float eps, momentum;

  TensorPtr gamma;
  TensorPtr beta;
  TensorPtr running_mean;
  TensorPtr running_var;
};

} // namespace deepnet
