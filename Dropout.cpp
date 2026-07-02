#include "Dropout.h"
#include "Filter.h"

Dropout::Dropout(int idx, int n, double rate, Layer* prev, std::mt19937& rng)
  : Layer(idx, Activation::LINEAR),
  rate_(rate), mask_(n, 1.0), rng_(rng), prevLayer_(prev), isTraining_(false)
{
  createNeurons(n, "L"+std::to_string(idx)+"-D");

  height_ = 1;
  width_ = n;
  channels_ = 1;
}

void Dropout::forward()
{
  // collect previous layer outputs
  std::vector<double> prev;
  prev.reserve(prevLayer_->size());
  for (auto* n : prevLayer_->neurons_)
    prev.push_back(n->output);

  // run the real forward using stored training flag
  forwardWithMask(prev, isTraining_);
}

// =============================================================
//  forwardWithMask
//  Called by Network::forward() with the previous layer's outputs.
//  Generates a new random mask each call during training.
// =============================================================
void Dropout::forwardWithMask(const std::vector<double>& prev, bool isTraining)
{
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int i = 0; i < (int)neurons_.size(); i++)
  {
    if (isTraining)
    {
      // regenerate mask every forward pass
      mask_[i] = (dist(rng_) > rate_) ? 1.0 : 0.0;
      // scale kept neurons so expected value is unchanged

      neurons_[i]->output = prev[i] * mask_[i] / (1.0 - rate_);
    }
    else
    {
      // no dropout at inference — all neurons active
      neurons_[i]->output = prev[i];
    }
  }
}


// =============================================================
//  backward
//  Collect error from next layer.
//  Zero error for dropped neurons so previous layer
//  receives no gradient through them.
// =============================================================
void Dropout::backward()
{
  for (int i = 0; i < (int)neurons_.size(); i++) 
  {
    // collect error from next layer via outConns
    neurons_[i]->error = 0.0;
    for (auto* c : neurons_[i]->outConns)
    {
      double w = (c->filter != nullptr) ? c->filter->weight(c->filterSlot) : c->weight;
      neurons_[i]->error += w * c->to->error;
    }

    // dropped neuron: zero error — no gradient flows bac
    neurons_[i]->error *= mask_[i];
  }
}

void Dropout::step(double, int)
{
  // no weights to update
}

void Dropout::zero_grad()
{
  for (auto* n : neurons_)
  {
    n->zero_grad();
  }
}
