#pragma once
#include "Op.h"
#include <vector>

class Neuron;
class Connection;

// =============================================================
//  StandardOp
//
//  Replicates the original Neuron behaviour as an explicit Op.
//  Written out so the pattern is visible and teachable.
//
//  forward:
//    z      = bias + sum(w * x)   for trainable connections
//    output = act(z)
//    output += sum(x)             for skip connections
//
//  backward:
//    collect error from outConns
//    delta = error * act'(z)
//    accumulate weight gradients
//    pass error back through skip connections
// =============================================================

class StandardOp : public Op
{
public:
    void forward (Neuron* n, const std::vector<Connection*>& inConns) override;
    void backward(Neuron* n, const std::vector<Connection*>& inConns) override;
};
