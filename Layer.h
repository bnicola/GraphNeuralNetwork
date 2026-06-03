#pragma once
#include "Activation.h"
#include "Neuron.h"
#include <string>
#include <vector>

class Layer
{
public:
  Layer(int index, Activation act);
  virtual ~Layer();

  virtual void forward() = 0;
  virtual void backward() = 0;
  virtual void step(double lr, int t) = 0;
  virtual void zero_grad() = 0;

  virtual std::string typeName() const = 0;
  virtual int skipFrom() const { return -1; }

  int size() const { return (int)neurons_.size(); }

  // ==================== SHAPE SYSTEM ====================
  virtual int channels() const { return channels_; }
  virtual int height()   const { return height_; }
  virtual int width()    const { return width_; }

  int index;
  Activation act;
  std::vector<Neuron*> neurons_;

protected:
  void createNeurons(int n, const std::string& prefix);

  // Shape members (protected so derived classes can set them)
  int channels_ = 1;
  int height_ = 1;
  int width_ = 1;   // For flattened layers: width = total neurons
};