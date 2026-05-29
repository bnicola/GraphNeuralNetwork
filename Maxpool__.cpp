#include "Maxpool.h"
#include <cassert>
#include <limits>

MaxPool2DLayer::MaxPool2DLayer(int idx,
                               int inputH, int inputW, int numChannels,
                               int poolH, int poolW,
                               int strideH, int strideW)
  : Layer(idx, Activation::LINEAR),
  inputH_(inputH), inputW_(inputW), numChans_(numChannels),
  poolH_(poolH), poolW_(poolW),
  strideH_(strideH), strideW_(strideW)
{
  assert(inputH  >= poolH && "MaxPool2D: pool height > input height");
  assert(inputW  >= poolW && "MaxPool2D: pool width  > input width");
  assert(strideH >= 1     && "MaxPool2D: strideH must be >= 1");
  assert(strideW >= 1     && "MaxPool2D: strideW must be >= 1");

  outputH_ = computeOutputDim(inputH, poolH, strideH);
  outputW_ = computeOutputDim(inputW, poolW, strideW);

  // total output neurons = channels(depth) * outputH * outputW
  int total = numChans_ * outputH_ * outputW_;
  createNeurons(total, "L" + std::to_string(idx) + "-MP");

  // one winner slot per output neuron, initially null
  winners_.resize(total, nullptr);
}

int MaxPool2DLayer::computeOutputDim(int input, int pool, int stride)
{
  return (input - pool) / stride + 1;
}

// =============================================================
//  forward
//
//  For each channel c, output position (oh, ow):
//    1. find output neuron flat index
//    2. scan the (pH x pW) window in the input
//    3. find the input neuron with the maximum value
//    4. store it as the winner
//    5. set output neuron value = max value
//
//  Input neuron flat index:
//    c * (inH*inW) + srcH * inW + srcW
//    where srcH = oh*sH + kh
//          srcW = ow*sW + kw
// =============================================================
void MaxPool2DLayer::forward()
{
  assert(prevLayer_ != nullptr && "MaxPool2D: prevLayer_ not set");

  for (int c = 0; c < numChans_; c++)
  {
    for (int oh = 0; oh < outputH_; oh++)
    {
      for (int ow = 0; ow < outputW_; ow++)
      {
        // flat index of this output neuron
        int dstIdx = c * (outputH_ * outputW_) + oh * outputW_ + ow;
        Neuron* dst = neurons_[dstIdx];

        // scan the pool window — find max
        double  maxVal = -std::numeric_limits<double>::infinity();
        Neuron* maxNeuron = nullptr;

        for (int kh = 0; kh < poolH_; kh++)
        {
          for (int kw = 0; kw < poolW_; kw++)
          {
            int srcH = oh * strideH_ + kh;
            int srcW = ow * strideW_ + kw;

            // flat index of this input neuron (channels-first, row-major)
            int srcIdx = c * (inputH_ * inputW_) + srcH * inputW_ + srcW;
            Neuron* src = prevLayer_->neurons_[srcIdx];

            if (src->output > maxVal)
            {
              maxVal = src->output;
              maxNeuron = src;
            }
          }
        }

        // set output value and record the winner
        dst->output = maxVal;
        winners_[dstIdx] = maxNeuron;
      }
    }
  }
}

// =============================================================
//  backward
//
//  For each output neuron:
//    collect error from next layer via outConns (same as always)
//    pass that error ONLY to the winning input neuron
//    all other input neurons in the window get zero
//
//  Why only the winner?
//    Only the max neuron affected the output.
//    Changing any other neuron in the window does not change
//    the output (as long as they stay below the max).
//    So their gradient is zero.
// =============================================================
void MaxPool2DLayer::backward()
{
  for (int i = 0; i < (int)neurons_.size(); i++)
  {
    neurons_[i]->error = 0.0;
    for (auto* c : neurons_[i]->outConns)
    {
      // read weight correctly — from filter if conv, from connection if dense
      double w = (c->filter != nullptr) ? c->filter->weight(c->filterSlot) : c->weight;
      neurons_[i]->error += w * c->to->error;
    }

    if (winners_[i] != nullptr)
    {
      winners_[i]->error += neurons_[i]->error;
    }
  }
}

// MaxPool has no weights — these are no-ops
void MaxPool2DLayer::step(double, int)
{
}

void MaxPool2DLayer::zero_grad()
{
  for (auto* n : neurons_)
  {
    n->zero_grad();
  }
  // clear winners — will be reset in next forward
  std::fill(winners_.begin(), winners_.end(), nullptr);
}