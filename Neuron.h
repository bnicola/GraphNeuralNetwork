#pragma once
#include "Activation.h"
#include "Adam.h"
#include "Connection.h"
#include "Filter.h"
#include "MicroOp.h"
#include <vector>
#include <string>

// =============================================================
//  Neuron  — updated with optional MicroOp graph
//
//  TWO MODES:
//
//  Mode 1 — Standard (microOps_ is empty):
//    forward()  and backward() work exactly as original.
//    Nothing changes for Linear, Conv, Residual, etc.
//
//  Mode 2 — MicroOp graph (microOps_ is not empty):
//    forward()  walks microOps_ left to right
//    backward() walks microOps_ right to left
//    Gradients and sensitivities collected automatically
//
//  HOW TO USE MODE 2:
//    Call buildGraph() to build standard weighted sum graph.
//    OR call beginGraph()/op()/endGraph() to build any graph.
//
//  The user of framework never sees MicroOp.
//  They just call network.addLinear() or network.addCustomLayer().
// =============================================================

class Neuron
{
public:
  Neuron(const std::string& name, Activation act);
  ~Neuron();

  void forward();
  void backward();
  void step(double lr, int t);
  void zero_grad();

  // ==========================================================
  //  Graph building API
  //
  //  buildStandardGraph() -- builds the standard weighted sum
  //                          + activation as a MicroOp graph
  //                          call after all inConns are wired
  //
  //  For arbitrary operations:
  //    Call addInput(inConn index) to get an INPUT node
  //    Call addParam(value ptr)    to get a PARAM node
  //    Call add/mul/tanh/relu/etc  to build the operation
  //    Call setOutput(node)        to mark the final node
  // ==========================================================
  void buildStandardGraph();

  MicroOp* addInput(int inConnIndex);
  MicroOp* addParam(double* valuePtr);
  MicroOp* add(MicroOp* a, MicroOp* b);
  MicroOp* mul(MicroOp* a, MicroOp* b);
  MicroOp* tanh_op(MicroOp* a);
  MicroOp* relu_op(MicroOp* a);
  MicroOp* exp_op(MicroOp* a);
  MicroOp* neg_op(MicroOp* a);
  MicroOp* sq_op(MicroOp* a);
  void     setOutput(MicroOp* node);

  std::string id;
  Activation  act;
  double      z;
  double      output;
  double      error;
  double      bias;
  double      biasGradient;
  AdamState   biasAdam;

  std::vector<Connection*> inConns;
  std::vector<Connection*> outConns;

private:
  // MicroOp graph -- empty means standard mode
  std::vector<MicroOp*>  microOps_;    // ops in forward order
  std::vector<MicroOp*>  inputNodes_;  // one per inConn
  std::vector<MicroOp*>  paramNodes_;  // weights and bias
  std::vector<double*>   paramPtrs_;   // pointer to actual value
  MicroOp* outputNode_ = nullptr;

  void forwardStandard();
  void backwardStandard();
  void forwardGraph();
  void backwardGraph();

  void clearGraph();
};
