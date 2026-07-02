#include "Neuron.h"
#include <cassert>

// =============================================================
//  Constructor / Destructor
// =============================================================

Neuron::Neuron(const std::string& name, Activation a)
  : id(name), act(a),
  z(0.0), output(0.0), error(0.0),
  bias(0.0), biasGradient(0.0),
  outputNode_(nullptr)
{
}

Neuron::~Neuron()
{
  clearGraph();
}

void Neuron::clearGraph()
{
  for (auto* op : microOps_)  delete op;
  for (auto* op : inputNodes_) delete op;
  microOps_.clear();
  inputNodes_.clear();
  paramNodes_.clear();
  paramPtrs_.clear();
  outputNode_ = nullptr;
}

// =============================================================
//  forward
//  If graph exists: use it. Otherwise: original code.
// =============================================================

void Neuron::forward()
{
  if (inConns.empty()) return;

  if (!microOps_.empty())
    forwardGraph();
  else
    forwardStandard();
}

// =============================================================
//  backward
//  If graph exists: use it. Otherwise: original code.
// =============================================================

void Neuron::backward()
{
  if (inConns.empty()) return;

  if (!microOps_.empty())
    backwardGraph();
  else
    backwardStandard();
}

// =============================================================
//  forwardStandard -- ORIGINAL forward()
// =============================================================

void Neuron::forwardStandard()
{
  z = bias;
  for (auto* c : inConns)
  {
    if (c->trainable == false) continue;
    double w = (c->filter != nullptr)
      ? c->filter->weight(c->filterSlot)
      : c->weight;
    z += w * c->from->output;
  }

  output = applyAct(act, z);

  for (auto* c : inConns)
  {
    if (c->trainable == false)
      output += c->weight * c->from->output;
  }
}

// =============================================================
//  backwardStandard -- ORIGINAL backward()
// =============================================================

void Neuron::backwardStandard()
{
  if (!outConns.empty())
  {
    error = 0.0;
    for (auto* c : outConns)
    {
      double w = (c->filter != nullptr)
        ? c->filter->weight(c->filterSlot)
        : c->weight;
      error += w * c->to->error;
    }
  }

  double delta = error * applyActDeriv(act, z);

  for (auto* c : inConns)
  {
    if (c->trainable == false) continue;
    double grad = delta * c->from->output;
    if (c->filter != nullptr)
      c->filter->accumulateGrad(c->filterSlot, grad);
    else
      c->gradient += grad;
  }
  biasGradient += delta;

  for (auto* c : inConns)
  {
    if (c->trainable == false)
      c->gradient += error * c->from->output;
  }
}

// =============================================================
//  forwardGraph
//
//  1. Sync input node values from inConns
//  2. Sync param node values from their pointers
//  3. Walk microOps_ left to right calling forward()
//  4. Write outputNode_->val to neuron->output
// =============================================================

void Neuron::forwardGraph()
{
  // sync input values from previous layer
  for (int i = 0; i < (int)inputNodes_.size(); i++)
    inputNodes_[i]->val = inConns[i]->from->output;

  // sync param values from actual weight/bias memory
  for (int i = 0; i < (int)paramNodes_.size(); i++)
    paramNodes_[i]->val = *paramPtrs_[i];

  // run every op in forward order
  for (auto* op : microOps_)
    op->forward();

  output = outputNode_->val;
  z = outputNode_->val;   // keep z in sync for compatibility
}

// =============================================================
//  backwardGraph
//
//  1. Collect error from outConns (SAME as standard backward)
//  2. Zero all micro gradients
//  3. Seed outputNode_->grad with error
//  4. Walk microOps_ right to left calling backward()
//  5. Collect weight gradients from param nodes
//  6. Store input sensitivities on inConns
//     (previous layer uses these to collect its own error)
// =============================================================

void Neuron::backwardGraph()
{
  // ── step 1: collect error from next layer ─────────────────
  // IDENTICAL to standard backward — unchanged
  if (!outConns.empty())
  {
    error = 0.0;
    for (auto* c : outConns)
    {
      double w = (c->filter != nullptr)
        ? c->filter->weight(c->filterSlot)
        : c->weight;
      error += w * c->to->error;
    }
  }

  // ── step 2: zero all micro gradients ──────────────────────
  for (auto* op : microOps_)  op->grad = 0.0;
  for (auto* op : inputNodes_) op->grad = 0.0;
  for (auto* op : paramNodes_) op->grad = 0.0;

  // ── step 3: seed output gradient with collected error ──────
  outputNode_->grad = error;

  // ── step 4: walk backwards ─────────────────────────────────
  for (int i = (int)microOps_.size() - 1; i >= 0; i--)
    microOps_[i]->backward();

  // ── step 5: collect weight gradients from param nodes ──────
  // paramNodes_[0]      = bias
  // paramNodes_[1..N]   = weights for inConns[0..N-1]
  biasGradient += paramNodes_[0]->grad;

  for (int i = 0; i < (int)inConns.size(); i++)
  {
    auto* c = inConns[i];
    if (!c->trainable) continue;

    double grad = paramNodes_[i + 1]->grad;

    if (c->filter != nullptr)
      c->filter->accumulateGrad(c->filterSlot, grad);
    else
      c->gradient += grad;
  }

  // ── step 6: store input sensitivities on inConns ──────────
  // previous layer neurons walk their outConns and read
  // c->sensitivity to collect their own error
  // sensitivity = d(this.output)/d(input[i])
  //             = inputNodes_[i]->grad / error
  // (inputNodes_[i]->grad already has: grad * d(output)/d(input))
  // so sensitivity = inputNodes_[i]->grad / error
  if (error != 0.0)
  {
    for (int i = 0; i < (int)inConns.size(); i++)
    {
      if (!inConns[i]->trainable) continue;
      inConns[i]->sensitivity = inputNodes_[i]->grad / error;
    }
  }
}

// =============================================================
//  step -- original
// =============================================================

void Neuron::step(double lr, int t)
{
  for (auto* c : inConns)
  {
    if (c->trainable && c->filter == nullptr)
      c->adam.update(c->weight, c->gradient, lr, t);
  }
  biasAdam.update(bias, biasGradient, lr, t);
}

// =============================================================
//  zero_grad -- original
// =============================================================

void Neuron::zero_grad()
{
  error = 0.0;
  biasGradient = 0.0;
  for (auto* c : inConns)
    c->gradient = 0.0;
}

// =============================================================
//  Graph building API
// =============================================================

MicroOp* Neuron::addInput(int inConnIndex)
{
  auto* node = new MicroOp(MicroOp::INPUT);
  node->val = inConns[inConnIndex]->from->output;
  inputNodes_.push_back(node);
  return node;
}

MicroOp* Neuron::addParam(double* valuePtr)
{
  auto* node = new MicroOp(MicroOp::PARAM);
  node->val = *valuePtr;
  paramNodes_.push_back(node);
  paramPtrs_.push_back(valuePtr);
  return node;
}

MicroOp* Neuron::add(MicroOp* a, MicroOp* b)
{
  auto* node = new MicroOp(MicroOp::ADD, a, b);
  microOps_.push_back(node);
  return node;
}

MicroOp* Neuron::mul(MicroOp* a, MicroOp* b)
{
  auto* node = new MicroOp(MicroOp::MUL, a, b);
  microOps_.push_back(node);
  return node;
}

MicroOp* Neuron::tanh_op(MicroOp* a)
{
  auto* node = new MicroOp(MicroOp::TANH, a);
  microOps_.push_back(node);
  return node;
}

MicroOp* Neuron::relu_op(MicroOp* a)
{
  auto* node = new MicroOp(MicroOp::RELU, a);
  microOps_.push_back(node);
  return node;
}

MicroOp* Neuron::exp_op(MicroOp* a)
{
  auto* node = new MicroOp(MicroOp::EXP, a);
  microOps_.push_back(node);
  return node;
}

MicroOp* Neuron::neg_op(MicroOp* a)
{
  auto* node = new MicroOp(MicroOp::NEG, a);
  microOps_.push_back(node);
  return node;
}

MicroOp* Neuron::sq_op(MicroOp* a)
{
  auto* node = new MicroOp(MicroOp::SQ, a);
  microOps_.push_back(node);
  return node;
}

void Neuron::setOutput(MicroOp* node)
{
  outputNode_ = node;
}

// =============================================================
//  buildStandardGraph
//
//  Builds the standard weighted sum + activation as a MicroOp
//  graph. Call this after all inConns are wired.
//
//  Graph for 2 inputs:
//    bias(PARAM) ──► ADD ──► sum0 ──► ADD ──► z ──► ACT ──► output
//                   ▲               ▲
//              MUL(w0,x0)      MUL(w1,x1)
//
//  This is equivalent to existing forward/backward.
//  The graph form makes it inspectable and extensible.
// =============================================================

void Neuron::buildStandardGraph()
{
  clearGraph();

  // bias param node -- always first in paramNodes_
  MicroOp* biasNode = addParam(&bias);

  // start running sum from bias
  MicroOp* running = biasNode;
  bool firstConn = true;

  for (int i = 0; i < (int)inConns.size(); i++)
  {
    auto* c = inConns[i];
    if (!c->trainable) continue;

    // input node for x[i]
    MicroOp* xNode = addInput(i);

    // weight param node for w[i]
    MicroOp* wNode = addParam(&c->weight);

    // w[i] * x[i]
    MicroOp* product = mul(wNode, xNode);

    // add to running sum
    if (firstConn)
    {
      // first term: running = bias + w0*x0
      running = add(running, product);
      firstConn = false;
    }
    else
    {
      // subsequent terms: running = running + wi*xi
      running = add(running, product);
    }
  }

  // activation on final sum
  MicroOp* actNode = nullptr;
  switch (act)
  {
  case Activation::TANH:   actNode = tanh_op(running); break;
  case Activation::RELU:   actNode = relu_op(running); break;
  case Activation::LINEAR: actNode = running;           break;
  default:                 actNode = running;           break;
  }

  setOutput(actNode);
}
