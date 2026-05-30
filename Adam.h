#pragma once
#include <cmath>

// =============================================================
//  AdamState
//  One instance per learnable parameter (weight or bias).
//  Holds the running averages m and v used by the Adam optimiser.
//
//  update() applies one Adam step to the parameter and returns
//  the step size taken.
//
//  Formula:
//    m      = β₁·m + (1-β₁)·gradient        smoothed gradient
//    v      = β₂·v + (1-β₂)·gradient²       smoothed gradient²
//    m_hat  = m / (1 - β₁ᵗ)                 bias correction
//    v_hat  = v / (1 - β₂ᵗ)                 bias correction
//    param -= lr · m_hat / (√v_hat + ε)      weight update
// =============================================================

class AdamState
{
public:
    AdamState();

    // apply one Adam step to param, return the step taken
    double update(double& param, double gradient, double lr, int t);

private:
    double m_;   // first moment  (smoothed gradient)
    double v_;   // second moment (smoothed gradient²)

    static constexpr double BETA1   = 0.9;
    static constexpr double BETA2   = 0.999;
    static constexpr double EPSILON = 1e-8;
};
