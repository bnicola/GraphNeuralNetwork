#include "Conv1d.h"
#include <cassert>

Conv1DLayer::Conv1DLayer(int index, Layer* prevLayer,      
                         int kernelSize, int stride, int numFilters,        
                         Activation act, double initStd)
  : Layer(index, act)
  , kernelSize_(kernelSize)
  , stride_(stride)
  , numFilters_(numFilters)
{
  assert(prevLayer != nullptr);
  assert(stride >= 1);
  assert(kernelSize >= 1);

  inputChannels_ = prevLayer->channels();
  inputSeqLen_ = prevLayer->width();

  outputSeqLen_ = (inputSeqLen_ - kernelSize) / stride + 1;
  assert(outputSeqLen_ > 0);

  channels_ = numFilters_;
  height_ = 1;
  width_ = outputSeqLen_;

  // Each output channel has its own filter of size (inputChannels * kernelSize)
  for (int f = 0; f < numFilters; f++)
  {
    int filterParams = kernelSize * inputChannels_;
    filters_.push_back(new Filter(filterParams, initStd));
  }

  int totalNeurons = numFilters * outputSeqLen_;
  createNeurons(totalNeurons, "L" + std::to_string(index) + "-C");
}

Conv1DLayer::~Conv1DLayer()
{
  for (auto* f : filters_)
    delete f;
}

// =============================================================
//  forward
//  Delegates to each neuron. Neuron::forward() already handles
//  conv connections via the filter pointer — nothing special here.
// =============================================================
void Conv1DLayer::forward()
{
  for (auto* n : neurons_)
    n->forward();
}

// =============================================================
//  backward
//  Delegates to each neuron. Neuron::backward() already routes
//  gradients to filter_->accumulateGrad() — nothing special here.
// =============================================================
void Conv1DLayer::backward()
{
  for (auto* n : neurons_)
    n->backward();
}

// =============================================================
//  step
//  Two things:
//    1. Each neuron updates its OWN bias via Adam (in neuron->step())
//       Neuron::step() skips conv connections (filter!=nullptr)
//       so only the bias gets updated per neuron.
//    2. The filter updates its K shared weights via filter_->step()
//       This is called ONCE — not per neuron — so each weight
//       gets exactly one Adam update using the fully accumulated gradient.
// =============================================================
void Conv1DLayer::step(double lr, int t)
{
  // Update biases for all output neurons
  for (auto* n : neurons_)
  {
    n->step(lr, t);   // conv weights are skipped in Neuron::step()
  }

  // Update shared filter weights (one update per filter)
  int contribPerFilter = (int)neurons_.size() / numFilters_;
  for (auto* f : filters_)
  {
    f->step(lr, t, contribPerFilter);
  }
}

// =============================================================
//  zero_grad
//  Two things:
//    1. Clear each neuron's error, biasGradient, inConn gradients
//    2. Clear the filter's K gradient accumulators
//
//  Note: the filter gradients are NOT cleared by Neuron::zero_grad()
//  because one filter is shared across all neurons — only the layer
//  knows about the filter, so only the layer can clear it.
// =============================================================
void Conv1DLayer::zero_grad()
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