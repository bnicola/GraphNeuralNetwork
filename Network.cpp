#include "Network.h"


// =============================================================
//  Constructor / Destructor
// =============================================================
Network::Network(unsigned seed)
  : rng_(seed), t_(0)
{}

Network::~Network()
{
  for (auto* c : allConns_) delete c;
  for (auto* l : layers_)   delete l;
}

// =============================================================
//  addLinear
//  Creates a fully connected layer and wires it densely
//  to the previous layer.
// =============================================================
void Network::addLinear(int n, Activation act)
{
  int    idx   = (int)layers_.size();
  auto*  layer = new Linear(idx, n, act);
  layers_.push_back(layer);

  if (layers_.size() > 1)
  {
    wireDense(layers_[layers_.size() - 2], layer);
  }
}

void Network::addLinear(int height, int width, int channels, Activation act)
{
  int    idx = (int)layers_.size();
  auto* layer = new Linear(idx, height, width, channels, act);
  layers_.push_back(layer);

  if (layers_.size() > 1)
  {
    wireDense(layers_[layers_.size() - 2], layer);
  }
}
// =============================================================
//  addDropout
//  Creates a dropout layer with pass-through connections
//  (weight=1) from the previous layer.
// =============================================================
void Network::addDropout(double p)
{
  assert(!layers_.empty() && "add an input layer first");

  int    idx   = (int)layers_.size();
  int    n     = layers_.back()->size();
  auto*  layer = new Dropout(idx, n, p, rng_);
  layers_.push_back(layer);

  // one-to-one pass-through connections from previous layer
  Layer* prev = layers_[layers_.size()-2];
  for (int i = 0; i < n; i++)
  {
    auto* c   = new Connection();
    c->from   = prev->neurons_[i];
    c->to     = layer->neurons_[i];
    c->weight = 1.0;
    prev->neurons_[i]->outConns.push_back(c);
    layer->neurons_[i]->inConns.push_back(c);
    allConns_.push_back(c);
  }
}

// =============================================================
//  addResidual
//  Creates a Residual layer and wires two paths automatically:
//    path 1: dense connections from previous layer  (normal weights)
//    path 2: identity skip from layers_[skipFromIdx] (no weight)
//
//  The skip source layer must have the same size as this layer.
// =============================================================
void Network::addResidual(int skipFromIdx, Activation act)
{
  assert(!layers_.empty() && "add an input layer first");
  assert(skipFromIdx >= 0 && skipFromIdx < (int)layers_.size() && "skipFromIdx out of range");
 
  Layer* skip = layers_[skipFromIdx];
  int skippedLayerNumNeurons = skip->size();

  int   idx   = (int)layers_.size();
  auto* layer = new Residual(idx, skippedLayerNumNeurons, act, skipFromIdx);
  layers_.push_back(layer);

  // path 1 — dense from previous layer
  wireDense(layers_[layers_.size()-2], layer);

  // path 2 — skip from source layer (wired automatically here)
  wireSkip(layers_[skipFromIdx], layer);
}

// =============================================================
//  addSoftmax
//  Creates a softmax layer and wires with previous layer 
//  automatically.
// =============================================================
void Network::addSoftmax()
{
  assert(!layers_.empty() && "add an input layer first");

  int idx           = (int)layers_.size();
  int prevLayerSize = layers_.back()->size();
  Layer* layer      = new Softmax(idx, prevLayerSize);
  layers_.push_back(layer);

  Layer* prev = layers_[layers_.size() - 2];
  for (int i = 0; i < prevLayerSize; i++)
  {
    Connection* c = new Connection();
    c->from       = prev->neurons_[i];
    c->to         = layer->neurons_[i];
    c->trainable  = false;
    c->weight     = 1.0;

    prev->neurons_[i]->outConns.push_back(c);
    layer->neurons_[i]->inConns.push_back(c);
    allConns_.push_back(c);
  }
}

// =============================================================
//  addConv1D
//
//  Creates a Conv1DLayer and wires it to the previous layer.
//
//  Output size = prevSize - kernelSize + 1
//  Each output neuron i gets K connections from input[i..i+K-1]
//  all pointing to the same Filter, each with its filterSlot.
//
//  initStd uses sqrt(1/kernelSize) — a common conv init.
// =============================================================
void Network::addConv1D(int kernelSize, Activation act, int stride, int numFilters)
{
  assert(!layers_.empty());
  Layer* prev = layers_.back();
  int idx = (int)layers_.size();
  double initStd = std::sqrt(1.0 / (kernelSize * prev->channels()));

  auto* layer = new Conv1DLayer(idx, prev, kernelSize, stride, numFilters, act, initStd);
  layers_.push_back(layer);
  wireConv1D(prev, layer, numFilters);
}

// =============================================================
//  wireConv1D  (private)
//
//  This is the key wiring that makes Conv different from Dense.
//
//  For each output neuron i (0 .. outputSize-1):
//    For each filter slot k (0 .. K-1):
//      Create a Connection from input[i+k] → output[i]
//      Set c->filter     = layer's Filter (shared)
//      Set c->filterSlot = k
//
//  So output neuron 0 connects to input[0,1,2] with slots [0,1,2]
//     output neuron 1 connects to input[1,2,3] with slots [0,1,2]
//     output neuron 2 connects to input[2,3,4] with slots [0,1,2]
//
//  The same filter slot k is used by ALL output neurons for
//  their k-th connection. That is weight sharing.
//
//  Neuron::forward()  reads  filter->weight(k) for each conn.
//  Neuron::backward() calls  filter->accumulateGrad(k, grad).
// =============================================================
void Network::wireConv1D(Layer* prev, Conv1DLayer* curr, int numFilters)
{
  int K          = curr->kernelSize();
  int stride     = curr->stride();
  int inChannels = prev->channels();
  int inSize     = prev->width();        // Important: use shape info

  int outSize    = curr->width();        // should be pre-computed in Conv1DLayer

  for (int f = 0; f < numFilters; f++)   // output channels
  {
    for (int o = 0; o < outSize; o++)  // output positions
    {
      int dstIdx = (f * outSize) + o;
      Neuron* dst = curr->neurons_[dstIdx];

      for (int ic = 0; ic < inChannels; ic++)   // input channels
      {
        for (int k = 0; k < K; k++)             // kernel positions
        {
          int srcPos = (o * stride) + k;
          if (srcPos >= inSize) continue;

          int srcIdx = (ic * inSize) + srcPos;
          Neuron* src = prev->neurons_[srcIdx];

          int filterSlot = (ic * K) + k;

          auto* c = new Connection();
          c->from = src;
          c->to = dst;
          c->filter = curr->filters_[f];
          c->filterSlot = filterSlot;
          c->trainable = true;

          src->outConns.push_back(c);
          dst->inConns.push_back(c);
          allConns_.push_back(c);
        }
      }
    }
  }
}

// =============================================================
// addConv2D
// =============================================================
void Network::addConv2D(int kernelHeight, int kernelWidth,
                        int strideH, int strideW,
                        int numFilters,
                        Activation act,
                        double initStd)
{
  assert(!layers_.empty() && "add an input layer first");

  Layer* prev          = layers_.back();
  int    inputHeight   = prev->height();
  int    inputWidth    = prev->width();
  int    inputChannels = prev->channels();

  int    layerIndex    = (int)layers_.size();

  auto* layer = new Conv2DLayer(layerIndex, layers_.back(),
                                kernelHeight, kernelWidth,
                                strideH, strideW,
                                numFilters, act, initStd);

  layers_.push_back(layer);

  wireConv2D(layers_[layers_.size() - 2], layer);
}

// =============================================================
// wireConv2D
// =============================================================
void Network::wireConv2D(Layer* prev, Conv2DLayer* curr)
{
  int inputHeight = curr->inputHeight();
  int inputWidth = curr->inputWidth();
  int kernelHeight = curr->kernelHeight();
  int kernelWidth = curr->kernelWidth();
  int strideHeight = curr->strideH();
  int strideWidth = curr->strideW();
  int numOutputFilters = curr->numFilters();
  int inputChannels = prev->size() / (inputHeight * inputWidth);
  int outputHeight = curr->outputHeight();
  int outputWidth = curr->outputWidth();

  int outputNeuronsPerFilter = outputHeight * outputWidth;

  for (int outputFilter = 0; outputFilter < numOutputFilters; outputFilter++)
  {
    for (int outputRow = 0; outputRow < outputHeight; outputRow++)
    {
      for (int outputCol = 0; outputCol < outputWidth; outputCol++)
      {
        int outputNeuronIndex = outputFilter * outputNeuronsPerFilter
          + outputRow * outputWidth
          + outputCol;
        Neuron* dst = curr->neurons_[outputNeuronIndex];

        for (int inputChannel = 0; inputChannel < inputChannels; inputChannel++)
        {
          for (int kernelRow = 0; kernelRow < kernelHeight; kernelRow++)
          {
            for (int kernelCol = 0; kernelCol < kernelWidth; kernelCol++)
            {
              int sourceRow = outputRow * strideHeight + kernelRow;
              int sourceCol = outputCol * strideWidth + kernelCol;

              if (sourceRow >= inputHeight || sourceCol >= inputWidth)
                continue;

              // Source neuron: channel inputChannel, pixel (sourceRow, sourceCol)
              int sourceNeuronIndex = inputChannel * (inputHeight * inputWidth)
                + sourceRow * inputWidth
                + sourceCol;
              Neuron* src = prev->neurons_[sourceNeuronIndex];

              // Filter slot encodes both the input channel and kernel position.
              // Fixed:    (inputChannel * kH * kW) + kh * kW + kw
              int filterSlot = inputChannel * (kernelHeight * kernelWidth)
                + kernelRow * kernelWidth
                + kernelCol;

              auto* c = new Connection();
              c->from = src;
              c->to = dst;
              c->filter = curr->filters_[outputFilter];
              c->filterSlot = filterSlot;
              c->trainable = true;

              src->outConns.push_back(c);
              dst->inConns.push_back(c);
              allConns_.push_back(c);
            }
          }
        }
      }
    }
  }
}

// =============================================================
//  addMaxPool2D
//  No connections created — MaxPool accesses the previous layer
//  neurons directly via prevLayer_ pointer.
//  User provides input dimensions explicitly.
// =============================================================
void Network::addMaxPool2D(int poolH, int poolW,
                           int strideH, int strideW)
{
  assert(!layers_.empty() && "add an input layer first");

  Layer* prev        = layers_.back();
  int    inputHeight = prev->height();
  int    inputWidth  = prev->width();
  int    numChannels = prev->channels();

  assert(layers_.back()->size() == numChannels * inputHeight * inputWidth
    && "addMaxPool2D: previous layer size does not match dimensions");

  int   idx = (int)layers_.size();
  auto* layer = new MaxPool2DLayer(idx,
                                   inputHeight, inputWidth,
                                   numChannels,
                                   poolH, poolW,
                                   strideH, strideW);

  // give MaxPool direct access to previous layer neurons
  layer->setPrevLayer(layers_.back());

  layers_.push_back(layer);

  // wire output neurons to next layer via dense connections
  // MaxPool output neurons need outConns so backward() can
  // collect error from the next layer the normal way
  // BUT we do NOT wire inConns — MaxPool reads prev directly
  // We do this by simply pushing the layer with no inConn wiring
  // outConns will be created when the NEXT layer is added via wireDense
}

// =============================================================
//  addMaxPool1D
//  No connections created — MaxPool accesses the previous layer
//  neurons directly via prevLayer_ pointer.
//  User provides input dimensions explicitly.
// =============================================================
void Network::addMaxPool1D(int poolSize, int stride)
{
  assert(!layers_.empty() && "add an input layer first");

  int   idx = (int)layers_.size();
  Layer* prev = layers_.back();
  int inputSize = prev->width();
  int numChannels = prev->channels();
  auto* layer = new MaxPool1DLayer(idx, inputSize, numChannels, poolSize, stride);

  assert(prev->size() == numChannels * inputSize
    && "addMaxPool1D: previous layer size does not match dimensions");
  // give MaxPool direct access to previous layer neurons
  layer->setPrevLayer(layers_.back());

  layers_.push_back(layer);
}

void Network::addLayerNorm(double epsilon)
{
  assert(!layers_.empty() && "add an input layer first");

  int   idx = (int)layers_.size();
  int   n = layers_.back()->size();

  auto* layer = new LayerNorm(idx, n, epsilon);

  // give LayerNorm direct access to previous layer's neurons
  // so backward() can write errors back — same pattern as MaxPool
  layer->setPrevLayer(layers_.back());

  layers_.push_back(layer);

  // Wire pass-through output connections (weight=1, not trainable)
  // so the NEXT layer can collect errors via outConns normally.
  // This mirrors how Dropout is wired.
  Layer* prev = layers_[layers_.size() - 2];
  for (int i = 0; i < n; i++)
  {
    auto* c = new Connection();
    c->from = prev->neurons_[i];
    c->to = layer->neurons_[i];
    c->weight = 1.0;
    c->trainable = false;   // pass-through — not a learnable weight

    prev->neurons_[i]->outConns.push_back(c);
    layer->neurons_[i]->inConns.push_back(c);
    allConns_.push_back(c);
  }
}

// =============================================================
//  forward
//  Propagates inputs left to right through all layers.
//  Dropout uses forwardWithMask(); all others use forward().
// =============================================================
std::vector<double> Network::forward(const std::vector<double>& inputs, bool isTraining)
{
  assert(!layers_.empty());
  assert((int)inputs.size() == layers_[0]->size());

  // set input layer values
  for (int i = 0; i < (int)inputs.size(); i++)
  {
    layers_[0]->neurons_[i]->output = inputs[i];
  }

  // propagate left to right
  for (int i = 1; i < (int)layers_.size(); i++)  
  {
    if (auto* dl = dynamic_cast<Dropout*>(layers_[i])) 
    {
      std::vector<double> prev;
      for (auto* n : layers_[i - 1]->neurons_)
      {
        prev.push_back(n->output);
      }
      dl->forwardWithMask(prev, isTraining);
    }
    else
    {
      layers_[i]->forward();
    }
  }
  
  return getOutputs();
}

// =============================================================
//  backward
//  Seeds error at output layer, then propagates right to left.
//  Loss: MSE = (1/N) Σ (output - target)²
//  Returns the loss value.
// =============================================================
double Network::backward(const std::vector<double>& targets, int batchSize)
{
  assert(!layers_.empty());
  Layer* out = layers_.back();
  assert((int)targets.size() == out->size());

  // seed error at output neurons
  double loss = 0.0;
  for (int i = 0; i < out->size(); i++)
  {
    double diff = out->neurons_[i]->output - targets[i];
    loss += diff * diff;
    out->neurons_[i]->error = diff / batchSize;   // ∂L/∂output = prediction - target
  }
  loss /= out->size();

  // propagate right to left
  for (int i = (int)layers_.size() - 1; i >= 1; i--)
  {
    layers_[i]->backward();
  }

  return loss;
}

// =============================================================
//  step
//  Increments global step counter t_ and updates all weights.
//  Skip connection weights are updated inside each neuron's
//  step() because they live in inConns.
// =============================================================
void Network::step(double lr)
{
  t_++;
  for (int i = 1; i < (int)layers_.size(); i++)
  {
    layers_[i]->step(lr, t_);
  }
}

// =============================================================
//  zero_grad
//  Clears all errors and gradients before next sample.
// =============================================================
void Network::zero_grad()
{
  for (auto* l : layers_)
  {
    l->zero_grad();
  }
}

// =============================================================
//  train
//  One full training step: zero_grad → forward → backward → step
// =============================================================
double Network::train(const std::vector<double>& inputs,
                       const std::vector<double>& targets,
                       double lr)
{
  zero_grad();
  forward(inputs, true);
  double loss = backward(targets);
  step(lr);
  return loss;
}

// =============================================================
//  train  (batch overload)
//
//  Processes all samples in the batch before updating weights.
//
//  What happens:
//    zero_grad()                    clear once before the batch
//    for each sample:
//      forward()                    compute outputs
//      backward(targets, N)         accumulate gradients / N
//    step()                         ONE weight update after all N
//
//  Gradient scaling:
//    backward() divides the seeded error by batchSize (N).
//    After N samples the accumulated gradient is the AVERAGE
//    over the batch — same scale as a single sample update.
//    This means you can use the same lr for any batch size.
//
//  Returns average loss over the batch.
// =============================================================
double Network::train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, double lr)
{
  assert(inputs.size() == targets.size() && "train: inputs and targets must have same batch size");
  assert(!inputs.empty() && "train: batch cannot be empty");

  // The size of our batch.
  int    N = (int)inputs.size();
  double loss = 0.0;

  zero_grad();   // clear once before the whole batch

  for (int i = 0; i < N; i++)
  {
    forward(inputs[i], true);

    // pass N so each sample's gradient contribution is divided by N
    // gradients accumulate across all N samples
    // after the loop the total = average gradient over the batch
    loss += backward(targets[i], N);
  }

  step(lr);   // ONE weight update using fully accumulated gradients

  return loss / N;
}

// =============================================================
//  predict
//  Forward pass with no dropout (isTraining=false).
// =============================================================
std::vector<double> Network::predict(const std::vector<double>& inputs)
{
  zero_grad();
  return forward(inputs, false);
}

// =============================================================
//  summary
//  Prints a table of all layers.
// =============================================================
void Network::summary() const
{
  std::cout << "\n=== Network Summary ===\n";
  std::cout << std::left
    << std::setw(6)  << "Layer"
    << std::setw(12) << "Type"
    << std::setw(8)  << "Nodes"
    << std::setw(12) << "Activation"
    << "Notes\n";
  std::cout << std::string(54, '-') << "\n";

  for (auto* l : layers_) 
  {
    std::string notes;
    if (auto* dl = dynamic_cast<Dropout*>(l))
    {
      notes = "p=" + std::to_string(dl->rate()).substr(0, 4);
    }
    if (auto* rl = dynamic_cast<Residual*>(l))
    {
      notes = "skip from L" + std::to_string(rl->skipFrom());
    }
    if (auto* cl = dynamic_cast<Conv1DLayer*>(l))
    {
      notes = "kernel=" + std::to_string(cl->kernelSize())
        + "  params=" + std::to_string(cl->kernelSize());
    }
    if (auto* cl = dynamic_cast<Conv2DLayer*>(l))
    {
      notes = "kernelHeight = " + std::to_string(cl->kernelHeight()) + ", kernelWidth = " + std::to_string(cl->kernelWidth())
        + "  filters=" + std::to_string(cl->numFilters());
    }

    std::cout << std::setw(6)  << ("L"+std::to_string(l->index))
      << std::setw(12) << l->typeName()
      << std::setw(8)  << l->size()
      << std::setw(12) << actName(l->act)
      << notes << "\n";
  }
  
  std::cout << std::string(54, '-') << "\n\n";
}

// =============================================================
//  wireDense  (private)
//  Fully connects every neuron in prev to every neuron in curr.
//  Weights initialised with Xavier: N(0, sqrt(2/(fanIn+fanOut)))
// =============================================================
void Network::wireDense(Layer* prev, Layer* curr)
{
  double std = xavier(prev->size(), curr->size());
  std::normal_distribution<double> dist(0.0, std);

  for (auto* src : prev->neurons_)
  {
    for (auto* dst : curr->neurons_)
    {
      auto* c = new Connection();
      c->from = src;
      c->to = dst;
      c->weight = dist(rng_);
      src->outConns.push_back(c);
      dst->inConns.push_back(c);
      allConns_.push_back(c);
    }
  }
}

//  Creates a normal Connection from src neuron i to dst neuron i
//  with weight=1.0 and trainable=false.
//
//  Because it is a real Connection in inConns/outConns:
//    forward()   adds  w*x = 1*x = x  after activation automatically
//    backward()  passes error back at full strength automatically
//    step()      skips it because trainable=false — weight stays 1.0
//
//  No special cases needed anywhere. Everything just works.
// =============================================================
void Network::wireSkip(Layer* src, Layer* dst)
{
  assert(src->size() == dst->size() && "wireSkip: layers must have the same size");

  for (int i = 0; i < src->size(); i++) 
  {
    auto* c       = new Connection();
    c->from       = src->neurons_[i];
    c->to         = dst->neurons_[i];
    c->weight     = 1.0;
    c->trainable  = false;   // fixed — never updated by Adam
    src->neurons_[i]->outConns.push_back(c);
    dst->neurons_[i]->inConns.push_back(c);
    allConns_.push_back(c);
  }
}

// =============================================================
//  getOutputs  (private)
// =============================================================
std::vector<double> Network::getOutputs() const
{
  std::vector<double> out;
  for (auto* n : layers_.back()->neurons_)
  {
    out.push_back(n->output);
  }
 
  return out;
}

// =============================================================
//  xavier  (private)
// =============================================================
double Network::xavier(int fanIn, int fanOut) const
{
  return std::sqrt(2.0 / (fanIn + fanOut));
}


// =============================================================
//  Network::save()
//
//  File format (plain text, version 2):
//
//    version 2
//    layers N
//
//    layer 0 Linear    size act
//    layer 1 Conv1D    size kernelSize stride numFilters act
//    layer 2 Conv1D    size kernelSize stride numFilters act
//    layer 3 Linear    size act
//    layer 4 Dropout   size rate
//    layer 5 Residual  size act skipFromIdx
//    layer 6 MaxPool2D size inputH inputW numChans poolH poolW strideH strideW
//    layer 7 Softmax   size
//
//    weights
//    L0 bias   v0 v1 v2 ...          (one per neuron)
//    L0 dense  v0 v1 v2 ...          (all inConn weights, neuron-major order)
//    L1 filter 0  v0 v1 v2 ...       (filter index then K weight values)
//    L1 filter 1  v0 v1 v2 ...
//    L1 bias   v0 v1 v2 ...
//    ...
//
//  NOTES:
//    - MaxPool2D and Dropout have no weights — only architecture line saved.
//    - Softmax saves bias only (connections are fixed weight=1, not trainable).
//    - Conv dense connections route through Filter, not Connection::weight.
//    - Residual saves bias + dense inConn weights (skip conns are fixed w=1).
// =============================================================

// =============================================================
//  Network::save()
//
//  File format (plain text, version 2):
//
//    version 2
//    layers N
//
//    layer 0 Linear    size act
//    layer 1 Conv1D    size kernelSize stride numFilters act
//    layer 2 Conv1D    size kernelSize stride numFilters act
//    layer 3 Linear    size act
//    layer 4 Dropout   size rate
//    layer 5 Residual  size act skipFromIdx
//    layer 6 MaxPool2D size inputH inputW numChans poolH poolW strideH strideW
//    layer 7 Softmax   size
//
//    weights
//    L0 bias   v0 v1 v2 ...          (one per neuron)
//    L0 dense  v0 v1 v2 ...          (all inConn weights, neuron-major order)
//    L1 filter 0  v0 v1 v2 ...       (filter index then K weight values)
//    L1 filter 1  v0 v1 v2 ...
//    L1 bias   v0 v1 v2 ...
//    ...
//
//  NOTES:
//    - MaxPool2D and Dropout have no weights — only architecture line saved.
//    - Softmax saves bias only (connections are fixed weight=1, not trainable).
//    - Conv dense connections route through Filter, not Connection::weight.
//    - Residual saves bias + dense inConn weights (skip conns are fixed w=1).
// =============================================================

bool Network::save(const std::string& filename) const
{
  std::ofstream f(filename);
  if (!f.is_open())
  {
    std::cerr << "Network::save — cannot open: " << filename << "\n";
    return false;
  }

  f << std::setprecision(10);   // enough precision to round-trip doubles

  // ── header ───────────────────────────────────────────────
  f << "version 2\n";
  f << "layers " << layers_.size() << "\n\n";

  // ── architecture lines ───────────────────────────────────
  for (auto* l : layers_)
  {
    f << "layer " << l->index << " ";

    if (auto* dl = dynamic_cast<Dropout*>(l))
    {
      f << "Dropout " << l->size() << " " << dl->rate() << "\n";
    }
    else if (auto* rl = dynamic_cast<Residual*>(l))
    {
      f << "Residual " << l->size() << " " << actName(l->act)
        << " " << rl->skipFrom() << "\n";
    }
    else if (auto* cl = dynamic_cast<Conv1DLayer*>(l))
    {
      f << "Conv1D " << l->size()
        << " " << cl->kernelSize()
        << " " << cl->stride()
        << " " << (int)cl->filters_.size()
        << " " << actName(l->act) << "\n";
    }
    else if (auto* cl = dynamic_cast<Conv2DLayer*>(l))
    {
      f << "Conv2D " << l->size()
        << " " << cl->inputHeight()
        << " " << cl->inputWidth()
        << " " << cl->kernelHeight()
        << " " << cl->kernelWidth()
        << " " << cl->strideH()
        << " " << cl->strideW()
        << " " << cl->numFilters()
        << " " << actName(l->act) << "\n";
    }
    else if (auto* ml = dynamic_cast<MaxPool2DLayer*>(l))
    {
      f << "MaxPool2D " << l->size()
        << " " << ml->inputH()
        << " " << ml->inputW()
        << " " << ml->numChans()
        << " " << ml->poolH()
        << " " << ml->poolW()
        << " " << ml->strideH()
        << " " << ml->strideW() << "\n";
    }
    else if (dynamic_cast<Softmax*>(l))
    {
      f << "Softmax " << l->size() << "\n";
    }
    else   // Linear (default)
    {
      f << "Linear " << l->size() << " " << actName(l->act) << "\n";
    }
  }

  // ── weights ──────────────────────────────────────────────
  f << "\nweights\n";

  for (auto* l : layers_)
  {
    std::string tag = "L" + std::to_string(l->index);

    // ── Conv1D / Conv2D — save filter weights then biases ──
    if (auto* cl = dynamic_cast<Conv1DLayer*>(l))
    {
      for (int fi = 0; fi < (int)cl->filters_.size(); fi++)
      {
        f << tag << " filter " << fi;
        for (int k = 0; k < cl->filters_[fi]->size(); k++)
          f << " " << cl->filters_[fi]->weight(k);
        f << "\n";
      }
      // biases — one per output neuron
      f << tag << " bias";
      for (auto* n : l->neurons_) f << " " << n->bias;
      f << "\n";
      continue;
    }

    if (auto* cl = dynamic_cast<Conv2DLayer*>(l))
    {
      for (int fi = 0; fi < (int)cl->filters_.size(); fi++)
      {
        f << tag << " filter " << fi;
        for (int k = 0; k < cl->filters_[fi]->size(); k++)
          f << " " << cl->filters_[fi]->weight(k);
        f << "\n";
      }
      f << tag << " bias";
      for (auto* n : l->neurons_) f << " " << n->bias;
      f << "\n";
      continue;
    }

    // ── MaxPool2D / Dropout — no weights ──────────────────
    if (dynamic_cast<MaxPool2DLayer*>(l)) continue;
    if (dynamic_cast<Dropout*>(l))        continue;

    // ── Linear / Residual / Softmax — bias + dense weights ─
    // bias line
    f << tag << " bias";
    for (auto* n : l->neurons_) f << " " << n->bias;
    f << "\n";

    // Softmax has no trainable dense weights (fixed pass-through)
    if (dynamic_cast<Softmax*>(l)) continue;

    // dense weight line — all inConn weights, neuron-major order
    // skip connections (trainable=false) are fixed at 1.0, not saved
    f << tag << " dense";
    for (auto* n : l->neurons_)
    {
      for (auto* c : n->inConns)
      {
        if (c->trainable && c->filter == nullptr)
          f << " " << c->weight;
      }
    }
    f << "\n";
  }

  std::cout << "Network saved to: " << filename << "\n";
  return true;
}

// =============================================================
//  Network::load()
//
//  Sequence:
//    1. Parse architecture lines → call the same add*() methods
//       used when building the network manually. This guarantees
//       identical wiring.
//    2. Parse weight lines → overwrite biases and filter/conn weights.
//
//  The network must be empty before calling load().
//  Any existing layers/connections are cleared first.
// =============================================================

// ── helpers ──────────────────────────────────────────────────

// Parse activation name string → enum
static Activation parseAct(const std::string& s)
{
  if (s == "relu")       return Activation::RELU;
  if (s == "leaky_relu") return Activation::LEAKY_RELU;
  if (s == "sigmoid")    return Activation::SIGMOID;
  if (s == "tanh")       return Activation::TANH;
  return Activation::LINEAR;
}

// Read a whitespace-separated list of doubles from a stream
static std::vector<double> readDoubles(std::istringstream& ss)
{
  std::vector<double> vals;
  double v;
  while (ss >> v) vals.push_back(v);
  return vals;
}

bool Network::load(const std::string& filename)
{
  std::ifstream f(filename.c_str());

  if (!f.is_open())
  {
    std::cerr << "Network::load - cannot open: "
      << filename << "\n";
    return false;
  }

  // ------------------------------------------
  // clear existing network
  // ------------------------------------------
  for (size_t i = 0; i < allConns_.size(); i++)
    delete allConns_[i];

  for (size_t i = 0; i < layers_.size(); i++)
    delete layers_[i];

  allConns_.clear();
  layers_.clear();
  t_ = 0;

  std::string line;

  // ==========================================
  // PASS 1 : architecture
  // ==========================================
  while (std::getline(f, line))
  {
    if (line.empty() || line[0] == '#')
      continue;

    if (line == "weights")
      break;

    std::istringstream ss(line);

    std::string token;
    ss >> token;

    if (token == "version")
      continue;

    if (token == "layers")
      continue;

    if (token != "layer")
      continue;

    int idx;
    std::string type;

    ss >> idx >> type;

    if (type == "Linear")
    {
      int size;
      std::string act;

      ss >> size >> act;

      // -----------------------------------
      // first layer = input layer
      // no dense wiring should occur
      // -----------------------------------
      if (layers_.empty())
      {
        int idx = (int)layers_.size();

        Layer* layer = new Linear(idx, size, parseAct(act));

        layers_.push_back(layer);
      }
      else
      {
        addLinear(size, parseAct(act));
      }
    }
    else if (type == "Dropout")
    {
      int size;
      double rate;

      ss >> size >> rate;

      addDropout(rate);
    }
    else if (type == "Residual")
    {
      int size;
      int skipIdx;
      std::string act;

      ss >> size >> act >> skipIdx;

      addResidual(skipIdx, parseAct(act));
    }
    else if (type == "Conv1D")
    {
      int size;
      int kernelSize;
      int stride;
      int numFilters;

      std::string act;

      ss >> size
        >> kernelSize
        >> stride
        >> numFilters
        >> act;

      addConv1D(kernelSize, parseAct(act), stride, numFilters);
    }
    else if (type == "Conv2D")
    {
      int size;
      int inputH;
      int inputW;
      int kernelH;
      int kernelW;
      int strideH;
      int strideW;
      int numFilters;

      std::string act;

      ss >> size
        >> inputH
        >> inputW
        >> kernelH
        >> kernelW
        >> strideH
        >> strideW
        >> numFilters
        >> act;

      addConv2D(kernelH, kernelW, strideH, strideW, numFilters, parseAct(act), std::sqrt(1.0 / (kernelH * kernelW)));
    }
    else if (type == "MaxPool2D")
    {
      int size;
      int inputH;
      int inputW;
      int numCh;
      int poolH;
      int poolW;
      int strideH;
      int strideW;

      ss >> size
        >> inputH
        >> inputW
        >> numCh
        >> poolH
        >> poolW
        >> strideH
        >> strideW;

      addMaxPool2D(numCh, poolH, poolW, strideH);
    }
    else if (type == "Softmax")
    {
      addSoftmax();
    }
  }

  // ==========================================
  // PASS 2 : weights
  // ==========================================
  std::map<int, std::vector<double> > denseWeights;

  while (std::getline(f, line))
  {
    if (line.empty() || line[0] == '#')
      continue;

    std::istringstream ss(line);

    std::string tag;
    std::string kind;

    ss >> tag >> kind;

    if (tag.empty() || tag[0] != 'L')
      continue;

    int layerIdx = std::atoi(tag.substr(1).c_str());

    Layer* l = layers_[layerIdx];

    if (kind == "bias")
    {
      std::vector<double> vals = readDoubles(ss);

      for (int i = 0; i < l->size(); i++)
      {
        l->neurons_[i]->bias = vals[i];
      }
    }
    else if (kind == "dense")
    {
      denseWeights[layerIdx] = readDoubles(ss);
    }
    else if (kind == "filter")
    {
      int filterIdx;

      ss >> filterIdx;

      std::vector<double> vals = readDoubles(ss);

      Conv1DLayer* c1 = dynamic_cast<Conv1DLayer*>(l);

      Conv2DLayer* c2 = dynamic_cast<Conv2DLayer*>(l);

      Filter* filter = nullptr;

      if (c1)
        filter = c1->filters_[filterIdx];

      if (c2)
        filter = c2->filters_[filterIdx];

      if (!filter)
        return false;

      for (int k = 0; k < filter->size(); k++)
      {
        filter->setWeight(k,  vals[k]);
      }
    }
  }

  // ==========================================
  // apply dense weights
  // ==========================================
  std::map<int, std::vector<double> >::iterator it;

  for (it = denseWeights.begin(); it != denseWeights.end(); ++it)
  {
    int layerIdx = it->first;

    std::vector<double>& vals = it->second;

    Layer* l = layers_[layerIdx];

    int cursor = 0;

    for (size_t ni = 0; ni < l->neurons_.size(); ni++)
    {
      Neuron* n = l->neurons_[ni];

      for (size_t ci = 0; ci < n->inConns.size(); ci++)
      {
        Connection* c = n->inConns[ci];

        if (c->trainable && c->filter == nullptr)
        {
          c->weight = vals[cursor++];
        }
      }
    }
  }

  std::cout << "Network loaded from: " << filename << "\n";

  return true;
}