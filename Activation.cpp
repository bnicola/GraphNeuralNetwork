#include "Activation.h"

double applyAct(Activation a, double z)
{
  switch (a) 
  {
  case Activation::RELU:    return z > 0.0 ? z : 0.0;
  case Activation::SIGMOID: return 1.0 / (1.0 + std::exp(-z));
  case Activation::TANH:    return std::tanh(z);
  default:                  return z;
  }
}

double applyActDeriv(Activation a, double z)
{
  switch (a)
  {
  case Activation::RELU:    return z > 0.0 ? 1.0 : 0.0;
  case Activation::SIGMOID: { double s = 1.0/(1.0+std::exp(-z)); return s*(1.0-s); }
  case Activation::TANH:    { double t = std::tanh(z); return 1.0 - t*t; }
  default:                  return 1.0;
  }
}

std::string actName(Activation a)
{
  switch (a)
  {
  case Activation::RELU:    return "relu";
  case Activation::SIGMOID: return "sigmoid";
  case Activation::TANH:    return "tanh";
  default:                  return "linear";
  }
}
