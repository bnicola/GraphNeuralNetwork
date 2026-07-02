#pragma once
#include "NumericalGradOp.h"
#include <vector>

class Neuron;
class Connection;

// =============================================================
//  CustomSquareOp
//
//  Example Op: output = (weighted_sum)^2
//
//  Good teaching example:
//    The analytic gradient of z^2 is 2z.
//    NumericalGradOp computes the same by nudging.
//    Compare them — they match to within 1e-7.
//
//  Shows how to inherit NumericalGradOp and only write
//  customForward(). Backward is completely free.
// =============================================================

class CustomSquareOp : public NumericalGradOp
{
public:
    void customForward(Neuron* n,
                       const std::vector<Connection*>& inConns) override;
};
