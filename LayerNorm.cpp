#include "LayerNorm.h"

// =============================================================
//  Constructor
// =============================================================
LayerNorm::LayerNorm(int index, int n, double epsilon)
    : Layer(index, Activation::LINEAR)
    , epsilon_(epsilon)
    , gamma_    (n, 1.0)   // scale  init = 1
    , beta_     (n, 0.0)   // shift  init = 0
    , gammaGrad_(n, 0.0)
    , betaGrad_ (n, 0.0)
    , gammaAdam_(n)
    , betaAdam_ (n)
    , normed_   (n, 0.0)
    , invStd_   (1.0)
{
    createNeurons(n, "L" + std::to_string(index) + "-LN");
    height_ = 1;
    width_ = n;
    channels_ = 1;
}

// =============================================================
//  forwardNorm
//
//  Called by Network::forward() with the previous layer's outputs.
//
//  Steps:
//    1. Compute mean over all N inputs
//    2. Compute variance over all N inputs
//    3. Normalise each input: x̂[i] = (x[i]-mean) / sqrt(var+ε)
//    4. Scale and shift:       out[i] = γ[i]*x̂[i] + β[i]
//
//  Cache normed_[] and invStd_ for use in backward().
// =============================================================
void LayerNorm::forwardNorm(const std::vector<double>& prev)
{
    int n = (int)neurons_.size();
    assert((int)prev.size() == n);

    // Step 1 — mean
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += prev[i];
    mean /= n;

    // Step 2 — variance
    double var = 0.0;
    for (int i = 0; i < n; i++)
    {
        double d = prev[i] - mean;
        var += d * d;
    }
    var /= n;

    // Step 3 — normalise and cache
    invStd_ = 1.0 / std::sqrt(var + epsilon_);
    for (int i = 0; i < n; i++)
    {
        normed_[i] = (prev[i] - mean) * invStd_;
    }

    // Step 4 — scale and shift
    for (int i = 0; i < n; i++)
    {
        neurons_[i]->output = gamma_[i] * normed_[i] + beta_[i];
    }
}

// =============================================================
//  forward — no-op, Network calls forwardNorm() instead
// =============================================================
void LayerNorm::forward()
{
    // no-op — use forwardNorm()
}

// =============================================================
//  backward
//
//  Collects error from next layer via outConns, then computes
//  gradients for γ, β, and propagates error back to prevLayer_.
//
//  Let δ[i]  = dL/d(out[i])   (from next layer)
//  Let x̂[i]  = normed_[i]
//  Let std   = 1/invStd_
//
//  Gradients for parameters:
//    dL/dγ[i] = δ[i] * x̂[i]
//    dL/dβ[i] = δ[i]
//
//  Gradient back to inputs:
//    dL/dx[i] = (γ[i] * invStd) *
//               (δ[i]
//                - (1/N) Σ δ[j]
//                - x̂[i] * (1/N) Σ δ[j]*x̂[j])
//
//  The last two terms are the corrections that account for the
//  fact that normalisation couples all inputs together through
//  the shared mean and variance.
// =============================================================
void LayerNorm::backward()
{
    int n = (int)neurons_.size();

    // ── collect error from next layer via outConns ────────────
    for (int i = 0; i < n; i++)
    {
        neurons_[i]->error = 0.0;
        for (auto* c : neurons_[i]->outConns)
        {
            double w = (c->filter != nullptr)
                       ? c->filter->weight(c->filterSlot)
                       : c->weight;
            neurons_[i]->error += w * c->to->error;
        }
    }

    // δ[i] = neurons_[i]->error
    // ── gradients for γ and β ─────────────────────────────────
    for (int i = 0; i < n; i++)
    {
        gammaGrad_[i] += neurons_[i]->error * normed_[i];
        betaGrad_[i]  += neurons_[i]->error;
    }

    // ── propagate error back to prevLayer_ ────────────────────
    if (prevLayer_ == nullptr) return;

    // precompute the two correction terms (scalars)
    double sumDelta      = 0.0;   // Σ δ[j]
    double sumDeltaNorm  = 0.0;   // Σ δ[j] * x̂[j]
    for (int i = 0; i < n; i++)
    {
        sumDelta     += neurons_[i]->error;
        sumDeltaNorm += neurons_[i]->error * normed_[i];
    }
    double invN = 1.0 / n;

    for (int i = 0; i < n; i++)
    {
        double dx = gamma_[i] * invStd_ *
                    (neurons_[i]->error
                     - invN * sumDelta
                     - normed_[i] * invN * sumDeltaNorm);

        // accumulate into previous layer's neuron error
        prevLayer_->neurons_[i]->error += dx;
    }
}

// =============================================================
//  step
//  Update γ and β using Adam.
// =============================================================
void LayerNorm::step(double lr, int t)
{
    for (int i = 0; i < (int)neurons_.size(); i++)
    {
        gammaAdam_[i].update(gamma_[i], gammaGrad_[i], lr, t);
        betaAdam_[i].update (beta_[i],  betaGrad_[i],  lr, t);
    }
}

// =============================================================
//  zero_grad
// =============================================================
void LayerNorm::zero_grad()
{
    std::fill(gammaGrad_.begin(), gammaGrad_.end(), 0.0);
    std::fill(betaGrad_.begin(),  betaGrad_.end(),  0.0);
    for (auto* n : neurons_)
    {
        n->error = 0.0;
    }
}
