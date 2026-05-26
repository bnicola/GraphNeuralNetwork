#pragma once
#include "Layer.h"
#include "Filter.h"

// =============================================================
//  Conv1DLayer  (PyTorch: nn.Conv1d)
//
//  A 1D convolutional layer with one shared Filter.
//
//  CONCEPT:
//    The previous layer has N neurons (the "sequence").
//    This layer slides a window of size K across it.
//    Each position of the window produces one output neuron.
//    Output count = N - K + 1
//
//    Example: N=5, K=3 → 3 output neurons
//      output[0] reads from input[0,1,2]   using filter[0,1,2]
//      output[1] reads from input[1,2,3]   using filter[0,1,2]
//      output[2] reads from input[2,3,4]   using filter[0,1,2]
//
//  KEY POINT — shared weights:
//    filter[0] is the SAME weight for all three connections above.
//    This is what makes it "convolutional" — the same detector
//    is applied at every position of the sequence.
//
//  HOW IT FITS IN OUR GRAPH:
//    Each output neuron i has K inConns:
//      inConns[k].from       = input neuron [i+k]
//      inConns[k].filter     = this layer's Filter
//      inConns[k].filterSlot = k
//    Neuron::forward() reads filter->weight(k) for each conn.
//    Neuron::backward() calls filter->accumulateGrad(k, grad).
//
//  STEP AND ZERO_GRAD:
//    step() calls filter_->step() for weight update.
//    Neuron::step() skips conv connections (filter != nullptr).
//    zero_grad() clears both neuron errors AND filter gradients.
//
//  BIAS:
//    Each output neuron has its own bias (from Neuron base).
//    This matches PyTorch's Conv1d default (bias=True).
// =============================================================

class Conv1DLayer : public Layer
{
public:
    // prevSize   = number of neurons in previous layer
    // kernelSize = K, the filter width
    // act        = activation applied after convolution
    // initStd    = weight init standard deviation
    Conv1DLayer(int index, int prevSize, int kernelSize,
                int stride, Activation act, double initStd);

    ~Conv1DLayer();

    void forward()              override;
    void backward()             override;
    void step(double lr, int t) override;
    void zero_grad()            override;

    std::string typeName() const override { return "Conv1D"; }
    int         kernelSize()    const     { return filter_->size(); }
    int         stride() const { return stride_; }
    // the shared filter — accessed by Network::wireConv1D()
    Filter* filter_;

private:
  // output count = (prevSize - kernelSize)/stride + 1
  static int outputSize(int prevSize, int kernelSize, int stride);

private:
    int stride_;
};
