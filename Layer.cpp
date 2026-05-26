#include "Layer.h"

Layer::Layer(int idx, Activation a)
  : index(idx), act(a)
{
}

Layer::~Layer()
{
  for (auto* n : neurons_)
  {
    delete n;
  }
}

void Layer::createNeurons(int n, const std::string& prefix)
{
  for (int i = 0; i < n; i++) 
  {
    std::string name = prefix + std::to_string(i);
    neurons_.push_back(new Neuron(name, act));
  }
}
