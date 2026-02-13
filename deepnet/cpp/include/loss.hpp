#pragma once

#include "tensor.hpp"

namespace deepnet {

// Cross Entropy Loss
class CrossEntropyLoss {
public:
  CrossEntropyLoss() = default;

  // Compute loss: input shape [batch, num_classes], target shape [batch]
  TensorPtr forward(const TensorPtr &input, const std::vector<int> &targets);

private:
  TensorPtr softmax(const TensorPtr &input);
  TensorPtr log_softmax(const TensorPtr &input);
};

// Mean Squared Error Loss
class MSELoss {
public:
  MSELoss() = default;

  TensorPtr forward(const TensorPtr &input, const TensorPtr &target);
};

} // namespace deepnet
