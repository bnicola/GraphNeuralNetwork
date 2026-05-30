#pragma once

#include "Layer.h"
#include "Filter.h"   
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>

class MaxPool2DLayer : public Layer
{
public:
  MaxPool2DLayer(int index,
    int inputH, int inputW, int numChannels,
    int poolH, int poolW,
    int strideH, int strideW);

  void forward() override;
  void backward() override;
  void step(double lr, int t) override;
  void zero_grad() override;

  std::string typeName() const override { return "MaxPool2D"; }

  void setPrevLayer(Layer* prev) { prevLayer_ = prev; }

  int inputH() const { return inputH_; }
  int inputW() const { return inputW_; }
  int numChans() const { return numChans_; }
  int poolH() const { return poolH_; }
  int poolW() const { return poolW_; }
  int strideH() const { return strideH_; }
  int strideW() const { return strideW_; }
  int outputH() const { return outputH_; }
  int outputW() const { return outputW_; }

private:
  int inputH_, inputW_, numChans_;
  int poolH_, poolW_;
  int strideH_, strideW_;
  int outputH_, outputW_;

  std::vector<Neuron*> winners_;
  Layer* prevLayer_ = nullptr;

  static int computeOutputDim(int input, int pool, int stride);
};