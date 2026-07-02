#pragma once
#include "Layer.h"
#include "Attention.h"
#include "Neuron.h"
#include "Connection.h"
#include <vector>
#include <string>
#include <cmath>
#include <cassert>

// =============================================================
//  AttentionScore
//
//  Second stage of attention. Sits after AttentionLayer (Q,K,V).
//
//  AttentionLayer stores neurons_ flat in THREE consecutive blocks:
//    Block 0 (Q):  neurons_[outC * seqLen + t]              outC=0..outputChannels-1
//    Block 1 (K):  neurons_[projSize + outC * seqLen + t]
//    Block 2 (V):  neurons_[2*projSize + outC * seqLen + t]
//    where projSize = outputChannels * seqLen
//
//  We use helper functions qNeuron(t,z), kNeuron(t,z), vNeuron(t,z)
//  to access the right neuron for token t, dim z.
//
//  Topology (coordinate system):
//    X-axis = token positions  (seqLen)
//    Z-axis = outputChannels   (going into page)
//
//  THREE internal neuron groups:
//
//  ── Group 1: Score neurons  (seqLen × seqLen) ─────────────────
//    score[q][k] = dot(Q[q], K[k]) × scale
//    flat: scoreNeurons_[q * seqLen + k]
//    Z-axis consumed by dot product — ONE number per (q,k) pair
//
//  ── Group 2: Softmax neurons  (seqLen × seqLen) ───────────────
//    attn[q][k] = softmax of row q
//    flat: attnNeurons_[q * seqLen + k]
//    seqLen independent softmaxes — one per query row
//
//  ── Group 3: Output neurons  (seqLen × outputChannels) ────────
//    out[q][z] = sum_k  attn[q][k] × V[k][z]
//    flat: neurons_[q * outputChannels + z]
//    Z-axis COMES BACK through V
//    These are neurons_ — visible to next layer
//
//  Shape:
//    width()    = seqLen
//    channels() = outputChannels  (same as AttentionLayer output)
//    size()     = seqLen × outputChannels
// =============================================================

class AttentionScore : public Layer
{
public:
  AttentionScore(int             index,
                 AttentionLayer* prev,
                 int             seqLen,
                 int             outputChannels);

  ~AttentionScore() override;

  void forward()               override;
  void backward()              override;
  void step(double lr, int t)  override {}   // no own weights
  void zero_grad()             override;

  std::string typeName() const override { return "AttnScore"; }

  int width()    const override { return seqLen_;          }
  int channels() const override { return outputChannels_;  }
  int height()   const override { return 1;                }

private:
  AttentionLayer* prev_;
  int             seqLen_;
  int             outputChannels_;
  double          scale_;

  // internal neuron groups
  std::vector<Neuron*> scoreNeurons_;   // seqLen × seqLen
  std::vector<Neuron*> attnNeurons_;    // seqLen × seqLen
  // neurons_ = output  seqLen × outputChannels  (base class)

  // ── index helpers ─────────────────────────────────────────────
  // Access Q, K, V neurons in AttentionLayer's flat neurons_ block
  // Layout in AttentionLayer: (outC * seqLen) + t
  // Block offsets: Q=0, K=projSize, V=2*projSize
  // where projSize = outputChannels * seqLen

  inline Neuron* qNeuron(int t, int z) const
  {
    int projSize = outputChannels_ * seqLen_;
    return prev_->neurons_[z * seqLen_ + t];
  }

  inline Neuron* kNeuron(int t, int z) const
  {
    int projSize = outputChannels_ * seqLen_;
    return prev_->neurons_[projSize + z * seqLen_ + t];
  }

  inline Neuron* vNeuron(int t, int z) const
  {
    int projSize = outputChannels_ * seqLen_;
    return prev_->neurons_[2 * projSize + z * seqLen_ + t];
  }

  void buildScoreNeurons();
  void buildAttnNeurons();
  void buildOutputNeurons();
};
