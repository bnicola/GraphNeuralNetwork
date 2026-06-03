#pragma once
#include "Layer.h"
#include "Filter.h"

class Conv1DLayer : public Layer
{
public:
  Conv1DLayer(int index, Layer* prevLayer,
    int kernelSize, int stride, int numFilters,
    Activation act, double initStd);

  ~Conv1DLayer();

  void forward() override;
  void backward() override;
  void step(double lr, int t) override;
  void zero_grad() override;

  std::string typeName() const override { return "Conv1D"; }

  int kernelSize() const { return kernelSize_; }
  int stride()     const { return stride_; }
  int numFilters() const { return numFilters_; }

  // Shape
  int channels() const override { return channels_; }
  int height()   const override { return 1; }
  int width()    const override { return outputSeqLen_; }

  std::vector<Filter*> filters_;

private:
  int kernelSize_;
  int stride_;
  int numFilters_;

  int inputChannels_ = 1;
  int inputSeqLen_ = 0;
  int outputSeqLen_ = 0;
};