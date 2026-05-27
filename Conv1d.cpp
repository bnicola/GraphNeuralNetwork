#include "Conv1d.h"
#include <cassert>

Conv1DLayer::Conv1DLayer(int index, int prevSize, int kernelSize,
  int stride, int numFilters, Activation act, double initStd)
  : Layer(index, act)
  , kernelSize_(kernelSize)
  , stride_(stride)
  , numFilters_(numFilters)
{
  assert(stride >= 1);
  // Create the number of filter (for different 
  // features to be captred).
  for (int f = 0; f < numFilters; f++)
  {
    filters_.push_back(new Filter(kernelSize, initStd));
  }
  int n = outputSize(prevSize, kernelSize, stride);
  assert(n > 0);
  int totalNeurons = numFilters * outputSize(prevSize, kernelSize, stride);
  createNeurons(totalNeurons, "L" + std::to_string(index) + "-C");
}

Conv1DLayer::~Conv1DLayer()
{
  //delete[] filter_;
}

int Conv1DLayer::outputSize(int prevSize, int kernelSize, int stride)
{
  return (prevSize - kernelSize) / stride + 1;
}

// =============================================================
//  forward
//  Delegates to each neuron. Neuron::forward() already handles
//  conv connections via the filter pointer — nothing special here.
// =============================================================
void Conv1DLayer::forward()
{
  for (auto* n : neurons_)
  {
    n->forward();
  }
}

// =============================================================
//  backward
//  Delegates to each neuron. Neuron::backward() already routes
//  gradients to filter_->accumulateGrad() — nothing special here.
// =============================================================
void Conv1DLayer::backward()
{
  for (auto* n : neurons_)
  {
    n->backward();
  }
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
  // Each output neuron contributes to the same filter weights.
  // We must average the gradients across all output neurons.
  int numContributions = size();   // number of output neurons

  for (auto* n : neurons_)
  {
    n->step(lr, t);     // updates bias only (conv conns skipped)
  }

  // Update shared filter weights with averaged gradient
  for (auto* f : filters_)
  {
    f->step(lr, t, (int)neurons_.size());
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
  for (int i = 0; i < numFilters_; i++)
  {
    filters_.at(i)->zero_grad();   // clear accumulated gradients in filter
  }
  
}