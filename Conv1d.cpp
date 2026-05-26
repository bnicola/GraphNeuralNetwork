#include "Conv1d.h"
#include <cassert>
#include <cmath>

Conv1DLayer::Conv1DLayer(int idx, int prevSize, int kernelSize, Activation a, double initStd)
  : Layer(idx, a),
  filter_(new Filter(kernelSize, initStd))
{
  assert(prevSize > kernelSize && "Conv1D: kernel larger than input");

  int n = outputSize(prevSize, kernelSize);
  createNeurons(n, "L"+std::to_string(idx)+"-C");
}

Conv1DLayer::~Conv1DLayer()
{
  delete filter_;
}

int Conv1DLayer::outputSize(int prevSize, int kernelSize)
{
  return prevSize - kernelSize + 1;
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
  for (auto* n : neurons_)
  {
    n->step(lr, t);     // updates bias only (conv conns skipped)
  }

  filter_->step(lr, t);   // updates K shared filter weights once
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
  filter_->zero_grad();   // clear accumulated gradients in filter
}
