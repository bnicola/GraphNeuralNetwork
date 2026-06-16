#include "AttentionScore.h"
#include <algorithm>

// =============================================================
//  Constructor
// =============================================================
AttentionScore::AttentionScore(int             index,
                               AttentionLayer* prev,
                               int             seqLen,
                               int             outputChannels)
  : Layer(index, Activation::LINEAR)
  , prev_           (prev)
  , seqLen_         (seqLen)
  , outputChannels_ (outputChannels)
  , scale_          (1.0 / std::sqrt(static_cast<double>(outputChannels)))
{
  assert(prev          != nullptr);
  assert(seqLen        >= 1);
  assert(outputChannels >= 1);
  // prev->neurons_ must have 3 projection blocks
  assert((int)prev->neurons_.size() == 3 * seqLen * outputChannels);

  buildScoreNeurons();
  buildAttnNeurons();
  buildOutputNeurons();

  width_    = seqLen_;
  channels_ = outputChannels_;
  height_   = 1;
}

// =============================================================
//  Destructor
// =============================================================
AttentionScore::~AttentionScore()
{
  for (auto* n : scoreNeurons_) delete n;
  for (auto* n : attnNeurons_)  delete n;
  // neurons_ deleted by base class
}

// =============================================================
//  buildScoreNeurons — seqLen × seqLen score neurons
// =============================================================
void AttentionScore::buildScoreNeurons()
{
  std::string prefix = "L" + std::to_string(index) + "-Score";
  for (int q = 0; q < seqLen_; q++)
  {
    for (int k = 0; k < seqLen_; k++)
    {
      std::string name = prefix + "[q=" + std::to_string(q) + "][k=" + std::to_string(k) + "]";
      scoreNeurons_.push_back(new Neuron(name, Activation::LINEAR));
    }
  }
}

// =============================================================
//  buildAttnNeurons — seqLen × seqLen softmax neurons
// =============================================================
void AttentionScore::buildAttnNeurons()
{
  std::string prefix = "L" + std::to_string(index) + "-Attn";
  for (int q = 0; q < seqLen_; q++)
  {
    for (int k = 0; k < seqLen_; k++)
    {
      std::string name = prefix + "[q=" + std::to_string(q) + "][k=" + std::to_string(k) + "]";
      attnNeurons_.push_back(new Neuron(name, Activation::LINEAR));
    }
   }
}

// =============================================================
//  buildOutputNeurons
//
//  Creates seqLen × outputChannels output neurons.
//  Wires V neurons → output neurons (non-trainable)
//  so errors flow back into V via outConns in backward.
//
//  Output flat layout: neurons_[q * outputChannels + z]
//  V access via vNeuron(k, z) helper.
// =============================================================
void AttentionScore::buildOutputNeurons()
{
  std::string prefix = "L" + std::to_string(index) + "-Out";
  createNeurons(seqLen_ * outputChannels_, prefix);

  for (int q = 0; q < seqLen_; q++)
  {
    for (int z = 0; z < outputChannels_; z++)
    {
      Neuron* outNeur = neurons_[q * outputChannels_ + z];

      for (int k = 0; k < seqLen_; k++)
      {
        Connection* c = new Connection();
        c->from = vNeuron(k, z);   // V[k][z]
        c->to = outNeur;
        c->weight = 1.0;
        c->trainable = false;

        vNeuron(k, z)->outConns.push_back(c);
        outNeur->inConns.push_back(c);
        //allConns_.push_back(c);
      }
    }
  }
}

// =============================================================
//  forward
//
//  Step 1 — dot product Q[q] · K[k] × scale → scoreNeurons_
//  Step 2 — softmax per row                  → attnNeurons_
//  Step 3 — weighted sum attn × V            → neurons_ (output)
// =============================================================
void AttentionScore::forward()
{
  // ── Step 1: dot products ──────────────────────────────────────
  // For each (q, k) pair on X-axis:
  //   score[q][k] = sum_z  Q[q][z] × K[k][z]  × scale
  //   Z-axis (outputChannels) consumed here → ONE number per pair

  for (int q = 0; q < seqLen_; q++)
  {
    for (int k = 0; k < seqLen_; k++)
    {
      double dot = 0.0;
      for (int z = 0; z < outputChannels_; z++) // Z-axis
      {
        dot += qNeuron(q, z)->output * kNeuron(k, z)->output;
      }

      scoreNeurons_[q * seqLen_ + k]->output = dot * scale_;
    }
  }

  // ── Step 2: softmax per row ───────────────────────────────────
  // seqLen independent softmaxes — one per query row q
  // Each row turns raw scores into attention percentages

  for (int q = 0; q < seqLen_; q++)
  {
    // numerical stability — subtract max before exp
    double maxVal = scoreNeurons_[q * seqLen_]->output;
    for (int k = 1; k < seqLen_; k++)
    {
      maxVal = std::max(maxVal, scoreNeurons_[q * seqLen_ + k]->output);
    }

    double sum = 0.0;
    for (int k = 0; k < seqLen_; k++)
    {
      double e = std::exp(scoreNeurons_[q * seqLen_ + k]->output - maxVal);
      attnNeurons_[q * seqLen_ + k]->output = e;
      sum += e;
    }

    for (int k = 0; k < seqLen_; k++)
    {
      attnNeurons_[q * seqLen_ + k]->output /= (sum + 1e-15);
    }
  }

  // ── Step 3: weighted sum × V ──────────────────────────────────
  // Z-axis COMES BACK through V
  // out[q][z] = sum_k  attn[q][k] × V[k][z]

  for (int q = 0; q < seqLen_; q++)
  {
    for (int z = 0; z < outputChannels_; z++)    // Z-axis back
    {
      double sum = 0.0;
      for (int k = 0; k < seqLen_; k++)
      {
        sum += attnNeurons_[q * seqLen_ + k]->output * vNeuron(k, z)->output;
      }

      neurons_[q * outputChannels_ + z]->output = sum;
    }
  }
}

// =============================================================
//  backward
//
//  Step 3 backward: output errors → dAttn + push dV into V neurons
//  Step 2 backward: dAttn → dScore  (softmax gradient per row)
//  Step 1 backward: dScore → push dQ into Q neurons
//                            push dK into K neurons
//
//  After this, AttentionLayer::backward() runs automatically
//  and walks inConns of Q, K, V neurons propagating back
//  through Wq, Wk, Wv filters. Nothing extra needed here.
// =============================================================
void AttentionScore::backward()
{
  // ── collect errors from next layer ───────────────────────────
  for (int q = 0; q < seqLen_; q++)
  {
    for (int z = 0; z < outputChannels_; z++)
    {
      int    idx = q * outputChannels_ + z;
      double err = 0.0;
      for (auto* c : neurons_[idx]->outConns)
        err += c->weight * c->to->error;
      neurons_[idx]->error = err;
    }
  }

  // ── Step 3 backward: weighted sum ────────────────────────────
  // out[q][z] = sum_k  attn[q][k] × V[k][z]
  // dAttn[q][k] += dOut[q][z] × V[k][z]    (sum over z)
  // dV[k][z]    += dOut[q][z] × attn[q][k]  (sum over q)

  std::vector<double> dAttn(seqLen_ * seqLen_, 0.0);

  for (int q = 0; q < seqLen_; q++)
  {
    for (int z = 0; z < outputChannels_; z++)
    {
      double dOut = neurons_[q * outputChannels_ + z]->error;

      for (int k = 0; k < seqLen_; k++)
      {
        dAttn[q * seqLen_ + k] += dOut * vNeuron(k, z)->output;
        vNeuron(k, z)->error += dOut * attnNeurons_[q * seqLen_ + k]->output;
      }
    }
  }

  // ── Step 2 backward: softmax per row ─────────────────────────
  // dScore[q][k] = attn[q][k] × (dAttn[q][k] - dotTerm)
  // dotTerm = sum_j( dAttn[q][j] × attn[q][j] )

  std::vector<double> dScore(seqLen_ * seqLen_, 0.0);

  for (int q = 0; q < seqLen_; q++)
  {
    double dotTerm = 0.0;
    for (int j = 0; j < seqLen_; j++)
    {
      dotTerm += dAttn[q * seqLen_ + j] * attnNeurons_[q * seqLen_ + j]->output;
    }

    for (int k = 0; k < seqLen_; k++)
    {
      dScore[q * seqLen_ + k] = attnNeurons_[q * seqLen_ + k]->output * (dAttn[q * seqLen_ + k] - dotTerm);
    }
  }

  // ── Step 1 backward: dot product ─────────────────────────────
  // score[q][k] = sum_z  Q[q][z] × K[k][z] × scale
  // dQ[q][z] += dScore[q][k] × scale × K[k][z]
  // dK[k][z] += dScore[q][k] × scale × Q[q][z]

  for (int q = 0; q < seqLen_; q++)
  {
    for (int k = 0; k < seqLen_; k++)
    {
      double dS = dScore[q * seqLen_ + k] * scale_;

      for (int z = 0; z < outputChannels_; z++)
      {
        qNeuron(q, z)->error += dS * kNeuron(k, z)->output;
        kNeuron(k, z)->error += dS * qNeuron(q, z)->output;
      }
    }
  }
}

// =============================================================
//  zero_grad
// =============================================================
void AttentionScore::zero_grad()
{
  for (auto* n : scoreNeurons_) n->zero_grad();
  for (auto* n : attnNeurons_)  n->zero_grad();
  for (auto* n : neurons_)      n->zero_grad();
}
