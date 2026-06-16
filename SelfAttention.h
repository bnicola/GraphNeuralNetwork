#pragma once
#include "Layer.h"
#include <vector>
#include <cmath>
#include <cassert>

// =============================================================
//  SelfAttention  (single-head, neuron-framework version)
//
//  This layer ONLY implements the attention mechanism itself:
//    - scaled dot-product between Q and K neurons
//    - softmax over scores
//    - weighted sum of V neurons → context
//
//  The four linear projections (Wq, Wk, Wv, Wo) are normal
//  Linear layers in the Network, wired by wireDense as usual.
//  Their weights are trained automatically by the existing
//  neuron backward/step machinery — no custom code needed here.
//
//  Expected layer stack (set up by addSelfAttention):
//
//    LayerQ  : Linear(seqLen * dHead)  — query projection
//    LayerK  : Linear(seqLen * dHead)  — key   projection
//    LayerV  : Linear(seqLen * dHead)  — value projection
//    THIS    : SelfAttention           — softmax + weighted sum
//    LayerO  : Linear(seqLen * dModel) — output projection
//
//  Shape convention (same as Embedding):
//    neurons_ are flat: neurons_[pos * dHead + h]
//    width()    = seqLen
//    channels() = dHead   (input and output of this layer)
//
//  Forward:
//    scores[q][k] = dot(Q[q], K[k]) / sqrt(dHead)
//    attn[q]      = softmax(scores[q])
//    out[q][h]    = sum_k  attn[q][k] * V[k][h]
//
//  Backward:
//    Collects errors from outConns (wired to LayerO).
//    Backprops through weighted sum → dAttn, dV
//    Backprops through softmax     → dScores
//    Backprops through dot-product → errors into LayerQ, LayerK neurons
//    dV errors are pushed into LayerV neurons directly.
// =============================================================

class SelfAttention : public Layer
{
public:
  // layerQ, layerK, layerV : the three projection Linear layers
  //   each must have size() == seqLen * dHead
  SelfAttention(int index,
                Layer* layerQ,
                Layer* layerK,
                Layer* layerV,
                int seqLen,
                int dHead);

  ~SelfAttention() override {}

  void forward()              override;
  void backward()             override;
  void step(double, int)      override {}   // no own weights
  void zero_grad()            override;

  std::string typeName() const override { return "SelfAttn"; }

  int channels() const override { return dHead_;  }
  int height()   const override { return 1;       }
  int width()    const override { return seqLen_; }

private:
  Layer* layerQ_;
  Layer* layerK_;
  Layer* layerV_;
  int    seqLen_;
  int    dHead_;

  // forward cache
  std::vector<std::vector<double>> Q_;      // [seqLen][dHead]
  std::vector<std::vector<double>> K_;      // [seqLen][dHead]
  std::vector<std::vector<double>> V_;      // [seqLen][dHead]
  std::vector<std::vector<double>> scores_; // [seqLen][seqLen]
  std::vector<std::vector<double>> attn_;   // [seqLen][seqLen]

  static std::vector<double> softmax(const std::vector<double>& v);
  static double dot(const std::vector<double>& a,
                    const std::vector<double>& b);
};
