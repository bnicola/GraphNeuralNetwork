// Residual.h
#pragma once
#include "Layer.h"

class Residual : public Layer
{
public:
  int skipFromIdx_;

  Residual(int index, int numNeurons, Activation act, int skipFromIdx);

  void forward() override;
  void backward() override;
  void step(double lr, int t) override;
  void zero_grad() override;

  std::string typeName() const override { return "Residual"; }
  int skipFrom() const override { return skipFromIdx_; }
};