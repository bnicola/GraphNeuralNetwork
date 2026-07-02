#pragma once
#include "Layer.h"
#include <random>

// =============================================================
//  Dropout  (PyTorch: nn.Dropout)
//
//  During training: each neuron is randomly dropped with
//  probability rate_. Kept neurons are scaled by 1/(1-rate_)
//  so the expected sum stays the same.
//
//  During inference (isTraining=false): all neurons pass through
//  unchanged. No mask applied.
//
//  The mask is regenerated every forward pass — different
//  neurons are dropped each time.
//
//  forward() is replaced by forwardWithMask() which takes
//  the previous layer outputs and the training flag.
//
//  backward():
//    error collected from next layer via outConns.
//    dropped neurons (mask=0) have their error zeroed so
//    the previous layer receives no gradient through them.
//
//  No learnable parameters — step() is a no-op.
// =============================================================

class Dropout : public Layer
{
public:
    Dropout(int index, int numNeurons, double rate, Layer* prev, std::mt19937& rng);

    // called by Network::forward() instead of forward()
    void forwardWithMask(const std::vector<double>& prevOutputs,
                         bool isTraining);

    void forward()              override;   // no-op
    void backward()             override;
    void step(double lr, int t) override;   // no-op — no weights
    void zero_grad()            override;

    std::string typeName() const override { return "Dropout"; }
    void setTraining(bool t) { isTraining_ = t; }
    double rate() const { return rate_; }

private:
    double              rate_;
    std::vector<double> mask_;
    std::mt19937&       rng_;

    Layer*              prevLayer_;
    bool                isTraining_;
};
