#pragma once
#include "NumericalGradOp.h"
#include <vector>

class Neuron;
class Connection;

// =============================================================
//  ScoreMulOp
//
//  An attention-style score neuron.
//
//  In attention, a score is computed as:
//    score = dot(Q, K) / sqrt(d)
//
//  This Op models ONE score neuron that receives two groups
//  of inputs via its inConns:
//    first  half of inConns = Q values  (query)
//    second half of inConns = K values  (key)
//
//  forward:
//    dot = sum(Q[i] * K[i])  for i in 0..d-1
//    output = dot / sqrt(d)
//
//  No backward code needed — NumericalGradOp handles it
//  by nudging. This means you can experiment with any
//  score function and get correct gradients for free.
//
//  WIRING:
//    The neuron must have 2*d inConns.
//    First d  connections = Q neurons (w=1, trainable=false)
//    Second d connections = K neurons (w=1, trainable=false)
//    The Op reads outputs directly — weights are pass-through.
//
//  WHY NOT USE WEIGHTS HERE?
//    Q and K are already the result of learned projections
//    in the previous layer. The score neuron just computes
//    the dot product — no additional weights needed.
// =============================================================

class ScoreMulOp : public NumericalGradOp
{
public:
    int d;   // dimension of Q and K — must equal inConns.size() / 2

    explicit ScoreMulOp(int d) : d(d) {}

    void customForward(Neuron* n,
                       const std::vector<Connection*>& inConns) override;
};
