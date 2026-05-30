#pragma once
#include "Linear.h"
#include "Dropout.h"
#include "Residual.h"
#include "Conv1d.h"
#include "Conv2d.h"
#include "Maxpool.h"
#include "LayerNorm.h"
#include "Softmax.h"
#include <vector>
#include <random>

// =============================================================
//  Network  (PyTorch: nn.Module / nn.Sequential)
//
//  Owns all layers and connections.
//  Provides a PyTorch-style interface:
//
//    net.addLinear(n, act)           add a fully connected layer
//    net.addDropout(p)               add a dropout layer
//    net.addResidual(n, act, src)    add a residual layer
//
//    net.forward(inputs)             run forward pass
//    net.backward(targets)           run backward pass, return loss
//    net.step(lr)                    update all weights via Adam
//    net.zero_grad()                 clear all gradients
//    net.train(inputs, targets, lr)  forward + backward + step
//    net.predict(inputs)             forward only, no dropout
//    net.summary()                   print layer table
//
//  Weight initialisation: Xavier  w ~ N(0, sqrt(2/(fan_in+fan_out)))
//  Loss: MSE  L = (1/N) Σ (output - target)²
//  Optimiser: Adam (β₁=0.9, β₂=0.999, ε=1e-8)
// =============================================================

class Network
{
public:
    explicit Network(unsigned seed = 42);
    ~Network();

    // ── layer builders ───────────────────────────────────
    void addLinear   (int n, Activation act);
    void addDropout  (double p = 0.2);
    void addResidual (int n, Activation act, int skipFromIdx);
    void addConv1D   (int kernelSize, Activation act, int stride = 1, int numFilters = 1);
    void addConv2D   (int inputHeight, int inputWidth, int kernelHeight, int kernelWidth, int strideH, int strideW, int numFilters, Activation act, double initStd);
    void addMaxPool2D(int inputH, int inputW, int numChannels, int poolH, int poolW, int strideH = 2, int strideW = 2);
    void addLayerNorm(double epsilon);
    void addSoftmax();

    // ── PyTorch-style interface ───────────────────────────
    std::vector<double> forward  (const std::vector<double>& inputs,
                                  bool isTraining = true);
    double              backward (const std::vector<double>& targets, int batchSize = 1);
    void                step     (double lr);
    void                zero_grad();

    double              train   (const std::vector<double>& inputs,
                                 const std::vector<double>& targets,
                                 double lr);

    double             train(const std::vector<std::vector<double>>& inputs, 
                             const std::vector<std::vector<double>>& targets, 
                             double lr);
    std::vector<double> predict (const std::vector<double>& inputs);

    void summary() const;

    bool save(const std::string& filename) const;
    bool load(const std::string& filename);

    // read-only access to layers (for demos / inspection)
    const std::vector<Layer*>& layers() const { return layers_; }

private:
    std::vector<Layer*>      layers_;
    std::vector<Connection*> allConns_;
    std::mt19937             rng_;
    int                      t_;          // global Adam step counter

    // wiring helpers
    void wireDense   (Layer* prev, Layer* curr);
    void wireSkip    (Layer* src,  Layer* dst);
    void wireConv1D  (Layer* prev, Conv1DLayer* curr, int numFilters);
    void wireConv2D(Layer* prev, Conv2DLayer* curr);

    std::vector<double> getOutputs() const;
    double              xavier     (int fanIn, int fanOut) const;
};
