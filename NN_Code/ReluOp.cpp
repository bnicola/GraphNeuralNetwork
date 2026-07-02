#include "ReluOp.h"
#include "Neuron.h"
#include "Connection.h"
#include "Filter.h"

void ReluOp::forward(Neuron* n, const std::vector<Connection*>& inConns)
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
    n->output = (n->z > 0.0) ? n->z : 0.0;
}

void ReluOp::backward(Neuron* n, const std::vector<Connection*>& inConns)
{
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

    double delta = n->error * ((n->z > 0.0) ? 1.0 : 0.0);

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
}
