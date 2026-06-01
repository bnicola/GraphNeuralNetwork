#pragma once

#include "Layer.h"
#include "Filter.h"   
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>

class MaxPool1DLayer : public Layer
{
public:
  MaxPool1DLayer(int index, int inputSize, int numChannels, int poolSize, int stride);

  void forward() override;
  void backward() override;
  void step(double lr, int t) override;
  void zero_grad() override;

  std::string typeName() const override { return "MaxPool1D"; }

  void setPrevLayer(Layer* prev) { prevLayer_ = prev; }

  int inputSize() const { return inputSize_; }
  int numChans() const { return numChans_; }
  int poolH() const { return poolSize_; }
  int strideH() const { return stride_; }
  int outputH() const { return outputSize_; }

private:
  int inputSize_, numChans_;
  int poolSize_;
  int stride_;
  int outputSize_;

  std::vector<Neuron*> winners_;
  Layer* prevLayer_ = nullptr;

  static int computeOutputDim(int input, int pool, int stride);
};