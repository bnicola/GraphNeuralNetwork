#pragma once
#include "Layer.h"
#include <random>

class Softmax : public Layer
{
public:
  Softmax(int index, int numNeurons);
  
  void forward()              override;   // no-op
  void backward()             override;
  void step(double lr, int t) override;   // no-op — no weights
  void zero_grad()            override;

  std::string typeName() const override { return "Softmax"; }

};
