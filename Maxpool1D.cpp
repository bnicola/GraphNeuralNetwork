#include "Maxpool1D.h"

MaxPool1DLayer::MaxPool1DLayer(int idx,
  int inputSize, int numChannels,
  int poolSize,
  int stride)
  : Layer(idx, Activation::LINEAR),
  inputSize_(inputSize), numChans_(numChannels),
  poolSize_(poolSize),
  stride_(stride)
{
  assert(inputSize >= poolSize && "MaxPool1D: pool > input");
  assert(stride >= 1 && "MaxPool1D: stride must be >= 1");

  outputSize_ = computeOutputDim(inputSize, poolSize, stride);

  int total = numChans_ * outputSize_;
  createNeurons(total, "L" + std::to_string(idx) + "-MP");

  winners_.resize(total, nullptr);
}

int MaxPool1DLayer::computeOutputDim(int input, int pool, int stride)
{
  return (input - pool) / stride + 1;
}

void MaxPool1DLayer::forward()
{
  assert(prevLayer_ != nullptr && "MaxPool1D: prevLayer_ not set");

  for (int c = 0; c < numChans_; c++)
  {        
    for (int i = 0; i < outputSize_; i++)
    {
      int dstIdx = (c * outputSize_) + i;
      Neuron* dst = neurons_[dstIdx];
      
      double maxVal = -1e18;
      Neuron* maxNeuron = nullptr;	  
      for (int k = 0; k < poolSize_; k++)
      {
        int srcIndex = (c * inputSize_) + (i * stride_) + k;

        Neuron* src = prevLayer_->neurons_[srcIndex];

        if (src->output > maxVal)
        {
          maxVal = src->output;
          maxNeuron = src;
        }          
      }

      dst->output = maxVal;
      winners_[dstIdx] = maxNeuron;
    }   
  }
}

void MaxPool1DLayer::backward()
{
  for (int i = 0; i < (int)neurons_.size(); i++)
  {
    neurons_[i]->error = 0.0;

    for (auto* c : neurons_[i]->outConns)
    {
      double w = (c->filter != nullptr) ? c->filter->weight(c->filterSlot) : c->weight;
      neurons_[i]->error += w * c->to->error;
    }

    if (winners_[i] != nullptr)
    {
      winners_[i]->error += neurons_[i]->error;
    }
  }
}

void MaxPool1DLayer::step(double, int)
{
  // no-op
}

void MaxPool1DLayer::zero_grad()
{
  for (auto* n : neurons_)
  {
    n->zero_grad();
  }
  std::fill(winners_.begin(), winners_.end(), nullptr);
}