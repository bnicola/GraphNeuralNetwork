#include "Softmax.h"

Softmax::Softmax(int idx, int numNeurons)
  : Layer(idx, Activation::LINEAR)
{
  createNeurons(numNeurons, "L"+std::to_string(idx)+"-S");
}

void Softmax::forward()
{
  // Step 1 — compute z = Σ w*x + bias  for each neuron
  for (auto* n : neurons_)
  {
    n->z = n->bias;
    for (auto* c : n->inConns)
      n->z += c->weight * c->from->output;
  }

  // Step 2 — numerical stability: subtract max
  double maxZ = neurons_[0]->z;
  for (auto* n : neurons_)
    if (n->z > maxZ) maxZ = n->z;

  // Step 3 — exponentiate and sum
  double sum = 0.0;
  for (auto* n : neurons_)
  {
    n->output = std::exp(n->z - maxZ);
    sum += n->output;
  }

  // Step 4 — normalise
  for (auto* n : neurons_)
    n->output /= sum;
}

void Softmax::backward()
{
  for (auto* n : neurons_)
  {
    n->backward();
  }
}

void Softmax::step(double lr, int t)
{
  for (auto* n : neurons_)
  {
    n->step(lr, t);
  }
}

void Softmax::zero_grad()
{
  for (auto* n : neurons_)
  {
    n->zero_grad();
  }
}
