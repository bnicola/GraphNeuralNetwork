#pragma once
#include "Layer.h"
#include "Adam.h"
#include <vector>
#include <cmath>
#include <numeric>
#include <cassert>

// =============================================================
//  LayerNorm  (PyTorch: nn.LayerNorm)
//
//  Normalises across all neurons within a single sample.
//  Essential building block for transformers.
//
//  FORWARD:
//    For a layer with N neurons, given inputs x[0..N-1]:
//
//    mean    = (1/N) Σ x[i]
//    var     = (1/N) Σ (x[i] - mean)²
//    norm[i] = (x[i] - mean) / sqrt(var + ε)
//    out[i]  = γ[i] * norm[i] + β[i]
//
//    γ (gamma) = learnable scale, initialised to 1.0
//    β (beta)  = learnable shift, initialised to 0.0
//
//  BACKWARD:
//    Let δ[i] = dL/d(out[i])   (error collected from next layer)
//    Let x̂[i] = norm[i]        (the normalised value)
//    Let std  = sqrt(var + ε)
//
//    dL/dγ[i] = δ[i] * x̂[i]   (gradient for gamma)
//    dL/dβ[i] = δ[i]            (gradient for beta)
//
//    dL/dx[i] = (γ[i] / std) * (δ[i]
//               - (1/N) Σ δ[j]
//               - x̂[i] * (1/N) Σ δ[j]*x̂[j])
//
//  LEARNABLE PARAMETERS:
//    gamma_[i]      initialised to 1.0
//    beta_[i]       initialised to 0.0
//    gammaAdam_[i]  one AdamState per gamma
//    betaAdam_[i]   one AdamState per beta
//
//  DESIGN — mirrors Dropout:
//    - No inConns/outConns wired to previous layer
//    - Reads prevLayer_ directly (set via setPrevLayer())
//    - Writes to its own neurons_[i]->output
//    - N = size() = previous layer size (one-to-one mapping)
//
//  USAGE IN NETWORK:
//    net.addLinear(512, Activation::LINEAR);
//    net.addLayerNorm();    // normalises the 512 outputs
// =============================================================

class LayerNorm : public Layer
{
public:
    // n         = number of neurons (same as previous layer size)
    // epsilon   = small constant for numerical stability (default 1e-5)
    LayerNorm(int index, int n, double epsilon = 1e-5);

    // Called by Network::forward() — reads prevLayer_ outputs directly
    void forwardNorm(const std::vector<double>& prevOutputs);

    void forward()              override;   // no-op — use forwardNorm()
    void backward()             override;
    void step(double lr, int t) override;
    void zero_grad()            override;

    std::string typeName() const override { return "LayerNorm"; }

    void setPrevLayer(Layer* prev) { prevLayer_ = prev; }

    // expose for save/load
    double gamma(int i) const { return gamma_[i]; }
    double beta (int i) const { return beta_[i];  }
    void   setGamma(int i, double v) { gamma_[i] = v; }
    void   setBeta (int i, double v) { beta_[i]  = v; }

private:
    double               epsilon_;
    std::vector<double>  gamma_;        // learnable scale,  init=1.0
    std::vector<double>  beta_;         // learnable shift,  init=0.0
    std::vector<double>  gammaGrad_;    // dL/dγ accumulator
    std::vector<double>  betaGrad_;     // dL/dβ accumulator
    std::vector<AdamState> gammaAdam_;
    std::vector<AdamState> betaAdam_;

    // cached from forward pass — needed for backward
    std::vector<double>  normed_;       // x̂[i] = (x[i]-mean)/std
    double               invStd_;       // 1 / sqrt(var + ε)

    Layer* prevLayer_ = nullptr;
};
