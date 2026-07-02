#pragma once
#include "Op.h"
#include <vector>

class Neuron;
class Connection;

// =============================================================
//  ReluOp
//
//  Forces ReLU activation regardless of neuron->act.
//  Useful when you want to override the activation for
//  specific neurons without changing the whole layer.
//
//  forward:
//    z      = bias + sum(w * x)
//    output = max(0, z)
//
//  backward:
//    delta  = error * (z > 0 ? 1 : 0)
//    accumulate weight gradients normally
// =============================================================

class ReluOp : public Op
{
public:
    void forward (Neuron* n, const std::vector<Connection*>& inConns) override;
    void backward(Neuron* n, const std::vector<Connection*>& inConns) override;
};
