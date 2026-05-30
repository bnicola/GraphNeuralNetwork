#include "Neuron.h"


Neuron::Neuron(const std::string& name, Activation a)
  : id(name), act(a),
  z(0.0), output(0.0), error(0.0),
  bias(0.0), biasGradient(0.0)
{}

// =============================================================
//  forward
//
//  The only change vs before:
//    When reading the weight on an inConn, check if it is a
//    conv connection. If so, read from the filter. Otherwise
//    read from the connection itself.
//
//  Everything else — splitting trainable/non-trainable for
//  skip connections, applying activation, adding skip after
//  activation — is identical.
//
//  Graph view:
//    Normal:  z += c->weight              * c->from->output
//    Conv:    z += c->filter->weight(k)   * c->from->output
//    Skip:    output += 1.0              * c->from->output  (after act)
// =============================================================
void Neuron::forward()
{
  if (inConns.empty()) return;

  // trainable inConns feed into z (pre-activation)
  z = bias;
  for (auto* c : inConns) 
  {
    if (c->trainable == false)
    {
      continue;   // skip connections handled below
    }

    // read weight: from filter if conv, from connection if dense
    double w = (c->filter != nullptr)  ? c->filter->weight(c->filterSlot) : c->weight;

    z += w * c->from->output;
  }

  output = applyAct(act, z);

  // non-trainable (skip/residual) connections added after activation
  // weight=1.0 so this is just: output += source->output
  for (auto* c : inConns)
  {
    if (c->trainable == false)
    {
      output += c->weight * c->from->output;
    }
  }
}

// =============================================================
//  backward
//
//  The only change vs before:
//    When accumulating gradient for an inConn, check if conv.
//    If conv  → route to filter->accumulateGrad(slot, grad)
//    If dense → accumulate on c->gradient as before
//    If skip  → accumulate on c->gradient (never applied in step)
//
//  Error collection via outConns is UNCHANGED — we still read
//  c->weight. For conv outConns we read filter->weight(slot).
// =============================================================
void Neuron::backward()
{
  if (inConns.empty())
  {
    return;
  }

  // collect error from next layer via outConns
  if (outConns.empty() == false)
  {
    error = 0.0;
    for (auto* c : outConns)
    {
      // read weight: filter or connection
      double w = (c->filter != nullptr) ? c->filter->weight(c->filterSlot) : c->weight;
      error += w * c->to->error;
    }
  }

  // trainable connections: gradient uses act'(z)
  double delta = error * applyActDeriv(act, z);
  for (auto* c : inConns)
  {
    if (c->trainable == false)
    {
      continue;   // skip handled below
    }

    double grad = delta * c->from->output;

    if (c->filter != nullptr)
    {
      // conv: route to shared filter accumulator
      //   multiple connections with same slot ADD their grad here
      c->filter->accumulateGrad(c->filterSlot, grad);
    }
    else
    {
      // dense: accumulate on connection itself as before
      c->gradient += grad;      
    }
  }
  biasGradient += delta;

  // skip connections: no act' (added after activation)
  // gradient accumulates here but is never applied in step()
  for (auto* c : inConns)
  {
    if (c->trainable == false)
    {
      c->gradient += error * c->from->output;
    }
  }
}

// =============================================================
//  step
//  Update only connections that:
//    - are trainable (not skip)
//    - do NOT have a filter (conv weights updated by filter->step())
// =============================================================
void Neuron::step(double lr, int t)
{
  for (auto* c : inConns)
  {
    if (c->trainable && c->filter == nullptr)
    {
      // conv connections skipped here — filter->step() handles them
      c->adam.update(c->weight, c->gradient, lr, t);      
    }
  }
  
  biasAdam.update(bias, biasGradient, lr, t);
}

// =============================================================
//  zero_grad — unchanged, clears all inConn gradients
// =============================================================
void Neuron::zero_grad()
{
  error        = 0.0;
  biasGradient = 0.0;
  for (auto* c : inConns)
  {
    // note: filter gradients are cleared by Conv1DLayer::zero_grad()
    // not here, because one filter is shared across many neurons
    c->gradient = 0.0;   
  }
}
