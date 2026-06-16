#include "SelfAttention.h"
#include <algorithm>
#include <numeric>

// =============================================================
//  Constructor
// =============================================================
SelfAttention::SelfAttention(int index,
                             Layer* layerQ,
                             Layer* layerK,
                             Layer* layerV,
                             int seqLen,
                             int dHead)
  : Layer(index, Activation::LINEAR),
    layerQ_ (layerQ),
    layerK_ (layerK),
    layerV_ (layerV),
    seqLen_ (seqLen),
    dHead_  (dHead)
{
  assert(layerQ->size() == seqLen * dHead && "LayerQ size mismatch");
  assert(layerK->size() == seqLen * dHead && "LayerK size mismatch");
  assert(layerV->size() == seqLen * dHead && "LayerV size mismatch");

  Q_      .assign(seqLen_, std::vector<double>(dHead_,  0.0));
  K_      .assign(seqLen_, std::vector<double>(dHead_,  0.0));
  V_      .assign(seqLen_, std::vector<double>(dHead_,  0.0));
  scores_ .assign(seqLen_, std::vector<double>(seqLen_, 0.0));
  attn_   .assign(seqLen_, std::vector<double>(seqLen_, 0.0));

  // Output neurons: seqLen * dHead  (same shape as Q/K/V layers)
  createNeurons(seqLen_ * dHead_, "L" + std::to_string(index) + "-SA");
  width_    = seqLen_;
  channels_ = dHead_;
}

// =============================================================
//  Helpers
// =============================================================
double SelfAttention::dot(const std::vector<double>& a,
                          const std::vector<double>& b)
{
  double s = 0.0;
  for (int i = 0; i < (int)a.size(); ++i) s += a[i] * b[i];
  return s;
}

std::vector<double> SelfAttention::softmax(const std::vector<double>& v)
{
  double maxV = *std::max_element(v.begin(), v.end());
  std::vector<double> out(v.size());
  double sum = 0.0;
  for (int i = 0; i < (int)v.size(); ++i)
  {
    out[i] = std::exp(v[i] - maxV);
    sum += out[i];
  }
  for (auto& val : out) val /= sum;
  return out;
}

// =============================================================
//  Forward
//
//  Reads Q, K, V from the three projection layers.
//  Computes attention and writes context to own neurons_.
// =============================================================
void SelfAttention::forward()
{
  const double scale = 1.0 / std::sqrt(static_cast<double>(dHead_));

  // Step 1 — read Q, K, V from their projection layers
  for (int pos = 0; pos < seqLen_; ++pos)
  {
    for (int h = 0; h < dHead_; ++h)
    {
      int idx = pos * dHead_ + h;
      Q_[pos][h] = layerQ_->neurons_[idx]->output;
      K_[pos][h] = layerK_->neurons_[idx]->output;
      V_[pos][h] = layerV_->neurons_[idx]->output;
    }
  }

  // Step 2 — scaled dot-product scores + softmax
  for (int q = 0; q < seqLen_; ++q)
  {
    for (int k = 0; k < seqLen_; ++k)
	{
      scores_[q][k] = dot(Q_[q], K_[k]) * scale;
	}
    attn_[q] = softmax(scores_[q]);
  }

  // Step 3 — context = weighted sum of V
  // out[q][h] = sum_k  attn[q][k] * V[k][h]
  for (int q = 0; q < seqLen_; ++q)
  {
    for (int h = 0; h < dHead_; ++h)
    {
      double val = 0.0;
      for (int k = 0; k < seqLen_; ++k)
	  {
        val += attn_[q][k] * V_[k][h];
	  }
      neurons_[q * dHead_ + h]->output = val;
    }
  }
}

// =============================================================
//  Backward
//
//  Only three things to backprop through here:
//    1. Weighted sum  → dAttn, dV
//    2. Softmax       → dScores
//    3. Dot-product   → errors into layerQ_, layerK_ neurons
//
//  dV errors go into layerV_ neurons.
//  layerQ_, layerK_, layerV_ Linear layers then backprop
//  through their own connections automatically.
// =============================================================
void SelfAttention::backward()
{
  const double scale = 1.0 / std::sqrt(static_cast<double>(dHead_));

  // ── collect errors from next layer via outConns ──────────────────────────
  // dOut[q][h] = error arriving at neurons_[q*dHead+h]
  std::vector<std::vector<double>> dOut(seqLen_, std::vector<double>(dHead_, 0.0));
  for (int q = 0; q < seqLen_; ++q)
  {
    for (int h = 0; h < dHead_; ++h)
    {
      int idx = q * dHead_ + h;
      neurons_[idx]->error = 0.0;
      for (auto* c : neurons_[idx]->outConns)
      {
        double w = (c->filter != nullptr)
                   ? c->filter->weight(c->filterSlot)
                   : c->weight;
        neurons_[idx]->error += w * c->to->error;
      }
      dOut[q][h] = neurons_[idx]->error;
    }
  }
  
  // ── Step 1: backward through weighted sum ────────────────────────────────
  // out[q][h] = sum_k  attn[q][k] * V[k][h]
  // dL/dattn[q][k] = sum_h  dOut[q][h] * V[k][h]
  // dL/dV[k][h]   += sum_q  dOut[q][h] * attn[q][k]
  std::vector<std::vector<double>> dAttn(seqLen_, std::vector<double>(seqLen_, 0.0));

  for (int q = 0; q < seqLen_; ++q)
    for (int k = 0; k < seqLen_; ++k)
      for (int h = 0; h < dHead_; ++h)
      {
        dAttn[q][k] += dOut[q][h] * V_[k][h];
        // push dV directly into layerV_ neurons
        layerV_->neurons_[k * dHead_ + h]->error += dOut[q][h] * attn_[q][k];
      }

  // ── Step 2: backward through softmax ─────────────────────────────────────
  // dscores[q][k] = attn[q][k] * (dAttn[q][k] - dot(dAttn[q], attn[q]))
  std::vector<std::vector<double>> dScores(seqLen_, std::vector<double>(seqLen_, 0.0));
  for (int q = 0; q < seqLen_; ++q)
  {
    double dotDA = dot(dAttn[q], attn_[q]);
    for (int k = 0; k < seqLen_; ++k)
      dScores[q][k] = attn_[q][k] * (dAttn[q][k] - dotDA);
  }

  // ── Step 3: backward through scaled dot-product ──────────────────────────
  // scores[q][k] = dot(Q[q], K[k]) * scale
  // dL/dQ[q][h] += sum_k  dScores[q][k] * scale * K[k][h]
  // dL/dK[k][h] += sum_q  dScores[q][k] * scale * Q[q][h]
  // Push directly into layerQ_ and layerK_ neuron errors
  for (int q = 0; q < seqLen_; ++q)
    for (int k = 0; k < seqLen_; ++k)
      for (int h = 0; h < dHead_; ++h)
      {
        layerQ_->neurons_[q * dHead_ + h]->error += dScores[q][k] * scale * K_[k][h];
        layerK_->neurons_[k * dHead_ + h]->error += dScores[q][k] * scale * Q_[q][h];
      }

  // layerQ_, layerK_, layerV_ Linear layers now have their neuron errors set.
  // Their own backward() walks inConns and propagates further back + accumulates
  // weight gradients automatically — nothing more needed here.
}

// =============================================================
//  zero_grad  — clear own neuron state only (no weights here)
// =============================================================
void SelfAttention::zero_grad()
{
  for (auto* n : neurons_)
    n->zero_grad();
}
