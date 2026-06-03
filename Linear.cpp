#include "Linear.h"

Linear::Linear(int idx, int n, Activation a)
  : Layer(idx, a)
{
  channels_ = 1;
  height_ = 1;
  width_ = n;
  createNeurons(n, "L"+std::to_string(idx)+"-N");
}

Linear::Linear(int idx, int height, int width, int channels, Activation a)
  : Layer(idx, a)
{
  channels_ = channels;
  height_ = height;
  width_ = width;
  int numOfNeurons = channels_ * height_ * width_;
  createNeurons(numOfNeurons, "L" + std::to_string(idx) + "-N");
}

void Linear::forward()
{
  for (auto* n : neurons_)
  {
    n->forward();
  }
}

void Linear::backward()
{
  for (auto* n : neurons_)
  {
    n->backward();
  }
}

void Linear::step(double lr, int t)
{
  for (auto* n : neurons_)
  {
    n->step(lr, t);
  }
}

void Linear::zero_grad()
{
  for (auto* n : neurons_)
  {
    n->zero_grad();
  }
}
