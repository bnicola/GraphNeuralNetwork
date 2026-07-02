#include "CustomSquareOp.h"
#include "Neuron.h"
#include "Connection.h"
#include "Filter.h"

// =============================================================
//  CustomSquareOp::customForward
//
//  Computes the standard weighted sum then squares it:
//    z      = bias + sum(w * x)
//    output = z * z
//
//  The analytic gradient of z^2 with respect to w is:
//    dOutput/dw = 2 * z * x
//
//  NumericalGradOp computes the same by nudging.
//  They match to within 1e-7 — verified in main.cpp.
// =============================================================

void CustomSquareOp::customForward(Neuron* n,
                                   const std::vector<Connection*>& inConns)
{
    double z = n->bias;
    for (auto* c : inConns)
    {
        if (!c->trainable) continue;
        double w = (c->filter != nullptr)
                   ? c->filter->weight(c->filterSlot)
                   : c->weight;
        z += w * c->from->output;
    }
    n->z      = z;
    n->output = z * z;
}
