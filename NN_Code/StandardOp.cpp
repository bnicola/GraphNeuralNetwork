#include "StandardOp.h"
#include "Neuron.h"
#include "Connection.h"
#include "Filter.h"
#include "Activation.h"

void StandardOp::forward(Neuron* n, const std::vector<Connection*>& inConns)
{
    n->z = n->bias;
    for (auto* c : inConns)
    {
        if (!c->trainable) continue;
        double w = (c->filter != nullptr)
                   ? c->filter->weight(c->filterSlot)
                   : c->weight;
        n->z += w * c->from->output;
    }

    n->output = applyAct(n->act, n->z);

    // skip connections added after activation
    for (auto* c : inConns)
    {
        if (!c->trainable)
            n->output += c->weight * c->from->output;
    }
}

void StandardOp::backward(Neuron* n, const std::vector<Connection*>& inConns)
{
    // collect error from next layer
    if (!n->outConns.empty())
    {
        n->error = 0.0;
        for (auto* c : n->outConns)
        {
            double w = (c->filter != nullptr)
                       ? c->filter->weight(c->filterSlot)
                       : c->weight;
            n->error += w * c->to->error;
        }
    }

    double delta = n->error * applyActDeriv(n->act, n->z);

    for (auto* c : inConns)
    {
        if (!c->trainable) continue;
        double grad = delta * c->from->output;
        if (c->filter != nullptr)
            c->filter->accumulateGrad(c->filterSlot, grad);
        else
            c->gradient += grad;
    }
    n->biasGradient += delta;

    // skip connections
    for (auto* c : inConns)
    {
        if (!c->trainable)
            c->gradient += n->error * c->from->output;
    }
}
