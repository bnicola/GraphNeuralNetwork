#pragma once
#include "Layer.h"

// =============================================================
//  Linear  (PyTorch: nn.Linear)
//  Fully connected layer.
//  Every neuron in this layer connects to every neuron
//  in the previous layer via dense weighted edges.
//
//  forward():   each neuron computes z = Σ w·x + bias, then act(z)
//  backward():  each neuron collects error and accumulates gradients
//  step():      each neuron updates weights and bias via Adam
// =============================================================

class Linear : public Layer
{
public:
    Linear(int index, int numNeurons, Activation act);
    Linear(int idx, int height, int width, int channels, Activation a);

    void forward()              override;
    void backward()             override;
    void step(double lr, int t) override;
    void zero_grad()            override;

    std::string typeName() const override { return "Linear"; }
};
