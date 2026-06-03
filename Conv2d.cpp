#include "Conv2d.h"

Conv2DLayer::Conv2DLayer(int index, Layer* prev,
                         int kernelHeight, 
                         int kernelWidth,
                         int strideH, int strideW,
                         int numFilters,
                         Activation act,
                         double initStd)
  : Layer(index, act)
  , inputH_(prev->height())
  , inputW_(prev->width())
  , kernelH_(kernelHeight)
  , kernelW_(kernelWidth)
  , strideH_(strideH)
  , strideW_(strideW)
  , numFilters_(numFilters)
{
  assert(strideH >= 1 && strideW >= 1 && "Stride must be at least 1");
  assert(numFilters >= 1 && "Must have at least 1 filter");

  inputChannels_ = prev->channels();
  for (int f = 0; f < numFilters; f++)
  {
    int filterParams = kernelH_ * kernelW_ * inputChannels_;
    filters_.push_back(new Filter(filterParams, initStd));
  }

  // The layer Shape (needed by next layer to get the input dimensions).
  height_   = outputHeight(inputH_, kernelH_, strideH_);
  width_    = outputWidth(inputW_, kernelW_, strideW_);
  channels_ = numFilters_;
  int totalNeurons = numFilters * height_ * width_;

  createNeurons(totalNeurons, "L" + std::to_string(index) + "-C2");
}

int Conv2DLayer::outputHeight(int inH, int kH, int sH)
{
  return (inH - kH) / sH + 1;
}

int Conv2DLayer::outputWidth(int inW, int kW, int sW)
{
  return (inW - kW) / sW + 1;
}

void Conv2DLayer::forward()
{
  for (auto* n : neurons_)
  {
    n->forward();
  }
}

void Conv2DLayer::backward()
{
  for (auto* n : neurons_)
  {
    n->backward();
  }
}

void Conv2DLayer::step(double lr, int t)
{
  for (auto* n : neurons_)
  {
    n->step(lr, t);
  }

  for (auto* f : filters_)
  {
    f->step(lr, t, size() / numFilters_);
  }
}

void Conv2DLayer::zero_grad()
{
  for (auto* n : neurons_)
  {
    n->zero_grad();
  }
  for (auto* f : filters_)
  {
    f->zero_grad();
  }
}