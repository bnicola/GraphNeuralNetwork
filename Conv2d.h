#pragma once
#include "Layer.h"
#include "Filter.h"
#include <cassert>

class Conv2DLayer : public Layer
{
public:
  Conv2DLayer(int index, Layer* prev,
              int kernelHeight = 3, int kernelWidth = 3,
              int strideH = 1, int strideW = 1,
              int numFilters = 1,
              Activation act = Activation::RELU,
              double initStd = 0.3);

  void forward() override;
  void backward() override;
  void step(double lr, int t) override;
  void zero_grad() override;

  std::string typeName() const override { return "Conv2D"; }

  int inputHeight() const  { return inputH_; }
  int inputWidth() const   { return inputW_; }
  int kernelHeight() const { return kernelH_; }
  int kernelWidth() const  { return kernelW_; }
  int strideH() const      { return strideH_; }
  int strideW() const      { return strideW_; }
  int outputHeight() const { return ((inputH_ - kernelH_) / strideH_ + 1);}
  int outputWidth() const  { return ((inputW_ - kernelW_) / strideW_ + 1); }
  int numFilters() const   { return numFilters_; }

  std::vector<Filter*> filters_;   // One filter per output channel

private:
  int inputH_, inputW_;
  int kernelH_, kernelW_;
  int strideH_, strideW_;
  int numFilters_;
  int inputChannels_;

  static int outputHeight(int inH, int kH, int sH);
  static int outputWidth(int inW, int kW, int sW);
};