#include "Attention.h"
#include <cassert>
#include <cmath>

AttentionLayer::AttentionLayer(int index, Layer* prevLayer, int outputChannels, Activation act, double initStd)
  : Layer(index, act)
  , outputChannels_(outputChannels)
{
  assert(prevLayer != nullptr);

  seqLength_ = prevLayer->width();    // X-Axis (Sequence Length)
  inputChannels_ = prevLayer->channels(); // Z-Axis (Input Dimension/embeddings)

  height_ = 1;
  width_ = seqLength_;
  channels_ = outputChannels_;

  // Allocate 1x1 Weight filters for Q, K, and V projections
  for (int f = 0; f < outputChannels_; f++)
  {
    qFilters_.push_back(new Filter(inputChannels_, initStd));
    kFilters_.push_back(new Filter(inputChannels_, initStd));
    vFilters_.push_back(new Filter(inputChannels_, initStd));
  }

  // Allocate continuous output neurons sequentially within neurons_ vector
  int neuronsPerProjection = outputChannels_ * seqLength_;

  // 1. Instantiate Q Neurons
  for (int i = 0; i < neuronsPerProjection; i++)
    neurons_.push_back(new Neuron("L" + std::to_string(index) + "-Q" + std::to_string(i), act));

  // 2. Instantiate K Neurons
  for (int i = 0; i < neuronsPerProjection; i++)
    neurons_.push_back(new Neuron("L" + std::to_string(index) + "-K" + std::to_string(i), act));

  // 3. Instantiate V Neurons
  for (int i = 0; i < neuronsPerProjection; i++)
    neurons_.push_back(new Neuron("L" + std::to_string(index) + "-V" + std::to_string(i), act));
}

AttentionLayer::~AttentionLayer()
{
  for (auto* f : qFilters_) delete f;
  for (auto* f : kFilters_) delete f;
  for (auto* f : vFilters_) delete f;
}

void AttentionLayer::forward()
{
  for (auto* n : neurons_)
  {
    n->forward();
  }
}

void AttentionLayer::backward()
{
  for (auto* n : neurons_)
  {
    n->backward();
  }
}

void AttentionLayer::step(double lr, int t)
{
  for (auto* n : neurons_)
  {
    n->step(lr, t);
  }

  // Apply parameter updates to each shared projection bank
  int contributions = seqLength_;
  for (int f = 0; f < outputChannels_; f++)
  {
    qFilters_[f]->step(lr, t, contributions);
    kFilters_[f]->step(lr, t, contributions);
    vFilters_[f]->step(lr, t, contributions);
  }
}

void AttentionLayer::zero_grad()
{
  for (auto* n : neurons_)
  {
    n->zero_grad();
  }
  for (int f = 0; f < outputChannels_; f++)
  {
    qFilters_[f]->zero_grad();
    kFilters_[f]->zero_grad();
    vFilters_[f]->zero_grad();
  }
}