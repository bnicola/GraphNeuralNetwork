#pragma once
#include "Adam.h"
#include <vector>
#include <string>

// =============================================================
//  Filter
//
//  This is the heart of Conv — the shared weight mechanism.
//
//  WHY IT EXISTS:
//    In a normal dense layer every connection owns its own weight.
//    In Conv, all output neurons share the SAME K weights.
//    We cannot put the weight on the Connection because the same
//    weight would get updated N times per backward pass (once per
//    output neuron that uses it), tripling/quadrupling the gradient.
//
//    Instead, the Filter owns the K weights.
//    Each Connection just stores which slot it uses (filterSlot).
//    During backward, all connections route their gradient INTO
//    the filter's accumulator for that slot.
//    After all neurons have done backward, one single Adam update
//    is applied per slot.
//
//  WHAT IT HOLDS:
//    weights[k]    — the actual learnable value for slot k
//    gradients[k]  — accumulated dL/dw[k] from all output neurons
//    adam[k]       — one AdamState per slot
//
//  EXAMPLE (K=3, 3 output neurons):
//
//    output neuron 0:  conn0(slot=0), conn1(slot=1), conn2(slot=2)
//    output neuron 1:  conn3(slot=0), conn4(slot=1), conn5(slot=2)
//    output neuron 2:  conn6(slot=0), conn7(slot=1), conn8(slot=2)
//
//    During backward:
//      gradients[0] += delta_0 * x0   (from neuron 0)
//      gradients[0] += delta_1 * x1   (from neuron 1)
//      gradients[0] += delta_2 * x2   (from neuron 2)
//      ... same for slots 1 and 2
//
//    Then step() applies ONE update:
//      weights[0] -= Adam(gradients[0])
//      weights[1] -= Adam(gradients[1])
//      weights[2] -= Adam(gradients[2])
//
//  This is exactly what PyTorch does internally for nn.Conv1d.
// =============================================================

class Filter
{
public:
    // kernelSize = number of weight slots (K)
    // initStd    = standard deviation for random weight init
    Filter(int kernelSize, double initStd);

    // apply one Adam step to each weight slot
    void step(double lr, int t);

    // clear all gradient accumulators before next sample
    void zero_grad();

    // read-only access
    int    size()               const { return kernelSize_; }
    double weight  (int slot)   const { return weights_[slot]; }
    double gradient(int slot)   const { return gradients_[slot]; }

    // called by Neuron::backward() to accumulate gradient
    void accumulateGrad(int slot, double grad);

private:
    int                  kernelSize_;
    std::vector<double>  weights_;     // weights_[0..K-1]
    std::vector<double>  gradients_;   // gradients_[0..K-1]
    std::vector<AdamState> adam_;      // one per slot
};
