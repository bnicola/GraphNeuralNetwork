#pragma once
#include "Op.h"
#include <vector>

class Neuron;
class Connection;

// =============================================================
//  NumericalGradOp
//
//  Base class for any Op that wants free backward() via nudging.
//  You only need to write customForward(). Backward is automatic.
//
//  HOW TO USE:
//    1. Inherit from NumericalGradOp
//    2. Override customForward() — write any maths you want
//    3. Attach to neuron: neuron->setOp(new MyOp())
//    4. backward() is computed by nudging — no maths needed
//
//  HOW BACKWARD WORKS:
//    For each weight w:
//      nudge up:   w = w + h  -> run customForward -> outUp
//      nudge down: w = w - h  -> run customForward -> outDown
//      restore:    w = w
//      grad = ((outUp - outDown) / (2*h)) * error
//
//    Two-sided central difference — error proportional to h^2.
//    With h = 1e-4 this gives accuracy to ~1e-8. Very precise.
//
//  SAME IS DONE FOR:
//    - each trainable weight
//    - bias
//    - each input (to propagate error to previous layer)
// =============================================================

class NumericalGradOp : public Op
{
public:
    double h = 1e-4;   // nudge size

    // Override with your custom operation.
    // Must write result to neuron->output.
    virtual void customForward(Neuron* n,
                               const std::vector<Connection*>& inConns) = 0;

    // Delegates to customForward
    void forward(Neuron* n, const std::vector<Connection*>& inConns) override;

    // Computed entirely by nudging — never needs to be overridden
    void backward(Neuron* n, const std::vector<Connection*>& inConns) override;
};
