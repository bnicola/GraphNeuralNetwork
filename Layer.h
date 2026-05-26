#pragma once
#include "Neuron.h"
#include "Activation.h"
#include <vector>
#include <string>

// =============================================================
//  Layer  (abstract base class)
//
//  Every layer type (Linear, Dropout, Residual) inherits from
//  this class and implements forward(), backward(), step(),
//  zero_grad().
//
//  neurons  holds all neurons in this layer.
//  index    is the layer's position in the network (0-based).
//  act      is the activation function applied by this layer.
// =============================================================

class Layer
{
public:
    Layer(int index, Activation act);
    virtual ~Layer();

    virtual void forward()              = 0;
    virtual void backward()             = 0;
    virtual void step(double lr, int t) = 0;
    virtual void zero_grad()            = 0;

    virtual std::string typeName() const = 0;
    virtual int         skipFrom()  const { return -1; }

    int size() const { return (int)neurons_.size(); }

    int                  index;
    Activation           act;
    std::vector<Neuron*> neurons_;

protected:
    // helper used by subclasses to create neurons
    void createNeurons(int n, const std::string& prefix);
};
