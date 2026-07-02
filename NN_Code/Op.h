#pragma once
#include <vector>

class Neuron;
class Connection;

// =============================================================
//  Op  — base class for any custom neuron operation
//
//  Attach to a Neuron via neuron->setOp(new MyOp()).
//  The Neuron then delegates forward() and backward()
//  entirely to the Op instead of doing its standard
//  weighted sum and chain rule.
//
//  The neuron OWNS the Op and deletes it in its destructor.
//
//  HOW TO ADD A NEW OPERATION:
//    1. Inherit from Op  (or from NumericalGradOp for free backward)
//    2. Implement forward()  — write result to neuron->output
//    3. Implement backward() — read neuron->error, write gradients
//    4. neuron->setOp(new MyOp())
// =============================================================

class Op
{
public:
    virtual ~Op() {}

    virtual void forward (Neuron* neuron,
                          const std::vector<Connection*>& inConns) = 0;

    virtual void backward(Neuron* neuron,
                          const std::vector<Connection*>& inConns) = 0;
};
