#include "NumericalGradOp.h"
#include "Neuron.h"
#include "Connection.h"
#include "Filter.h"

void NumericalGradOp::forward(Neuron* n, const std::vector<Connection*>& inConns)
{
    customForward(n, inConns);
}

void NumericalGradOp::backward(Neuron* n, const std::vector<Connection*>& inConns)
{
    // ── Step 1: collect error arriving from next layer ────────
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

    // ── Step 2: numerical gradient for each weight ────────────
    // Central difference: grad = (f(w+h) - f(w-h)) / (2h)
    // Error proportional to h^2 — far more accurate than one-sided.

    for (auto* c : inConns)
    {
        if (!c->trainable) continue;

        double* param = (c->filter != nullptr)
                        ? &c->filter->weightRef(c->filterSlot)
                        : &c->weight;

        double original = *param;

        *param = original + h;
        customForward(n, inConns);
        double outUp = n->output;

        *param = original - h;
        customForward(n, inConns);
        double outDown = n->output;

        *param = original;

        double grad = ((outUp - outDown) / (2.0 * h)) * n->error;

        if (c->filter != nullptr)
            c->filter->accumulateGrad(c->filterSlot, grad);
        else
            c->gradient += grad;
    }

    // ── Step 3: numerical gradient for bias ───────────────────
    {
        double original = n->bias;

        n->bias = original + h;
        customForward(n, inConns);
        double outUp = n->output;

        n->bias = original - h;
        customForward(n, inConns);
        double outDown = n->output;

        n->bias = original;

        n->biasGradient += ((outUp - outDown) / (2.0 * h)) * n->error;
    }

    // ── Step 4: numerical gradient for inputs ─────────────────
    // Nudge each input to find how much error to send back.
    // This is what replaces the outConns walk in standard backward.

    for (auto* c : inConns)
    {
        if (!c->trainable) continue;

        double origInput = c->from->output;

        c->from->output = origInput + h;
        customForward(n, inConns);
        double outUp = n->output;

        c->from->output = origInput - h;
        customForward(n, inConns);
        double outDown = n->output;

        c->from->output = origInput;

        double inputGrad = (outUp - outDown) / (2.0 * h);
        c->from->error += inputGrad * n->error;
    }

    // ── Step 5: restore correct output after all nudging ──────
    customForward(n, inConns);
}
