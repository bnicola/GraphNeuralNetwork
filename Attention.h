#pragma once
#include "Layer.h"
#include "Filter.h"
#include <vector>
#include <string>

class AttentionLayer : public Layer
{
public:
  AttentionLayer(int index, Layer* prevLayer, int outputChannels, Activation act, double initStd);
  ~AttentionLayer();

  void forward() override;
  void backward() override;
  void step(double lr, int t) override;
  void zero_grad() override;

  std::string typeName() const override { return "Attention"; }

  // Dimensions for single head configurations
  int channels() const override { return outputChannels_; } // Projected channels count
  int height()   const override { return 1; }
  int width()    const override { return seqLength_; }      // Timeline width (X-axis)

  // Exposed components for internal parameter tracking inside Network::wireAttention
  std::vector<Filter*> qFilters_;
  std::vector<Filter*> kFilters_;
  std::vector<Filter*> vFilters_;

private:
  int seqLength_;
  int inputChannels_;
  int outputChannels_;
};