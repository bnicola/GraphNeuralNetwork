#include "Filter.h"
#include <random>
#include <stdexcept>

Filter::Filter(int kernelSize, double initStd)
    : kernelSize_(kernelSize),
      weights_(kernelSize),
      gradients_(kernelSize, 0.0),
      adam_(kernelSize)
{
  // initialise weights with small random values
  // same Xavier-style init as our dense connections
  std::mt19937 rng(std::random_device{}());
  std::normal_distribution<double> dist(0.0, initStd);
  for (auto& w : weights_)
  {
    w = dist(rng);
  }
}

// =============================================================
//  step
//  Called ONCE per training step, after ALL output neurons
//  have run their backward() and accumulated gradients.
//
//  Applies one Adam update per weight slot.
//  This is the key point — one update, not N updates.
// =============================================================
void Filter::step(double lr, int t, int contributions)
{
  for (int k = 0; k < kernelSize_; k++)
  {
    double avgGrad = gradients_[k] / contributions;

    // === GRADIENT CLIPPING - Very Important for Conv ===
    avgGrad = std::max(-5.0, std::min(5.0, avgGrad));   // clip to [-5, 5]

    adam_[k].update(weights_[k], avgGrad, lr, t);
  }
}

// =============================================================
//  zero_grad
//  Clears all gradient accumulators.
//  Called at the start of each training step.
// =============================================================
void Filter::zero_grad()
{
  std::fill(gradients_.begin(), gradients_.end(), 0.0);
}

// =============================================================
//  accumulateGrad
//  Called by Neuron::backward() for each conv connection.
//  Routes the gradient to the correct slot.
//
//  This is what makes shared weights work:
//    Multiple connections with the same filterSlot all ADD
//    their gradient to the same accumulator.
//    The final update sees the sum — correct averaged gradient.
// =============================================================
void Filter::accumulateGrad(int slot, double grad)
{
  if (slot < 0 || slot >= kernelSize_)
  {
    throw std::out_of_range("Filter::accumulateGrad: slot out of range");
  }
  gradients_[slot] += grad;
}
