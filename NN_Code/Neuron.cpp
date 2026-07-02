#include "Neuron.h"
#include "Op.h"

Neuron::Neuron(const std::string& name, Activation act)
    : id(name), act(act),
      z(0.0), output(0.0), error(0.0),
      bias(0.0), biasGradient(0.0),
      op_(nullptr)
{}

Neuron::~Neuron()
{
    delete op_;
}

void Neuron::setOp(Op* op)
{
    delete op_;
    op_ = op;
}

// =============================================================
//  forward
// =============================================================

void Neuron::forward()
{
    if (inConns.empty()) return;

    if (op_ != nullptr)
    {
        op_->forward(this, inConns);
        return;
    }

    standardForward();
}

// =============================================================
//  backward
// =============================================================

void Neuron::backward()
{
    if (inConns.empty()) return;

    if (op_ != nullptr)
    {
        op_->backward(this, inConns);
        return;
    }

    standardBackward();
}

// =============================================================
//  standardForward — original Neuron::forward() unchanged
// =============================================================

void Neuron::standardForward()
{
    z = bias;
    for (auto* c : inConns)
    {
        if (!c->trainable) continue;
        double w = (c->filter != nullptr)
                   ? c->filter->weight(c->filterSlot)
                   : c->weight;
        z += w * c->from->output;
    }

    output = applyAct(act, z);

    for (auto* c : inConns)
    {
        if (!c->trainable)
            output += c->weight * c->from->output;
    }
}

// =============================================================
//  standardBackward — original Neuron::backward() unchanged
// =============================================================

void Neuron::standardBackward()
{
    if (!outConns.empty())
    {
        error = 0.0;
        for (auto* c : outConns)
        {
            double w = (c->filter != nullptr)
                       ? c->filter->weight(c->filterSlot)
                       : c->weight;
            error += w * c->to->error;
        }
    }

    double delta = error * applyActDeriv(act, z);

    for (auto* c : inConns)
    {
        if (!c->trainable) continue;
        double grad = delta * c->from->output;
        if (c->filter != nullptr)
            c->filter->accumulateGrad(c->filterSlot, grad);
        else
            c->gradient += grad;
    }
    biasGradient += delta;

    for (auto* c : inConns)
    {
        if (!c->trainable)
            c->gradient += error * c->from->output;
    }
}

// =============================================================
//  step — unchanged
// =============================================================

void Neuron::step(double lr, int t)
{
    for (auto* c : inConns)
    {
        if (c->trainable && c->filter == nullptr)
            c->adam.update(c->weight, c->gradient, lr, t);
    }
    biasAdam.update(bias, biasGradient, lr, t);
}

// =============================================================
//  zero_grad — unchanged
// =============================================================

void Neuron::zero_grad()
{
    error        = 0.0;
    biasGradient = 0.0;
    for (auto* c : inConns)
        c->gradient = 0.0;
}
