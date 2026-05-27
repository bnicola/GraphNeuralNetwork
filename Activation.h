#pragma once
#include <string>
#include <cmath>

// =============================================================
//  Activation
//  Supported: LINEAR, RELU, SIGMOID, TANH
// =============================================================

enum class Activation 
{ 
  LINEAR, 
  RELU,
  LEAKY_RELU,
  SIGMOID, 
  TANH 
};

double      applyAct      (Activation a, double z);
double      applyActDeriv (Activation a, double z);
std::string actName       (Activation a);
