#include "Residual.h"

Residual::Residual(int index, int numNeurons, Activation act, int skipFromIdx)
  : Layer(index, act)
  , skipFromIdx_(skipFromIdx)
{
  createNeurons(numNeurons, "L" + std::to_string(index) + "-R");
}

void Residual::forward()
{
  for (auto* n : neurons_)
  {
    n->forward();
  }
}

void Residual::backward()
{
  for (auto* n : neurons_)
  {
    n->backward();
  }
}

void Residual::step(double lr, int t)
{
  for (auto* n : neurons_)
  {
    n->step(lr, t);
  }
}

void Residual::zero_grad()
{
  for (auto* n : neurons_)
  {
    n->zero_grad();
  }
}