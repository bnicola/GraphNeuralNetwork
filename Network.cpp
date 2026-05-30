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
void Network::addResidual(int n, Activation act, int skipFromIdx)
{
  assert(!layers_.empty() && "add an input layer first");
  assert(skipFromIdx >= 0 && skipFromIdx < (int)layers_.size() && "skipFromIdx out of range");
  assert(layers_[skipFromIdx]->size() == n && "Residual layer size must match skip source layer size");

  int   idx   = (int)layers_.size();
  auto* layer = new Residual(idx, n, act, skipFromIdx);
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
  assert(!layers_.empty() && "add an input layer first");
  int    idx = (int)layers_.size();
  int    prevSize = layers_.back()->size();
  double initStd = std::sqrt(1.0 / kernelSize);

  auto* layer = new Conv1DLayer(idx, prevSize, kernelSize, stride, numFilters, act, initStd);
  layers_.push_back(layer);

  wireConv1D(layers_[layers_.size() - 2], layer, numFilters);
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
  int K = curr->kernelSize();
  // curr->size() gives the total number of neueons.
  int outputSize = curr->size() / numFilters;
  for (int f = 0; f < numFilters; f++)
  {
    for (int i = 0; i < (outputSize); i++)
    {
      Neuron* dst = curr->neurons_[f * outputSize + i];

      int stride = curr->stride();
      for (int k = 0; k < K; k++)
      {
        // input neuron at position i+k feeds into output neuron i
        Neuron* src = prev->neurons_[(i * stride) + k];

        auto* c = new Connection();
        c->from = src;
        c->to = dst;
        c->filter = curr->filters_[f];   // shared filters
        c->filterSlot = k;               // which weight slot
        c->trainable = true;
        // c->weight not used — weight lives in filter
        // c->gradient not used — grad goes to filter->accumulateGrad

        src->outConns.push_back(c);
        dst->inConns.push_back(c);
        allConns_.push_back(c);
      }
    }
  }
}

// =============================================================
// addConv2D
// =============================================================
void Network::addConv2D(int inputHeight, int inputWidth,
  int kernelHeight, int kernelWidth,
  int strideH, int strideW,
  int numFilters,
  Activation act,
  double initStd)
{
  assert(!layers_.empty() && "add an input layer first");

  int idx = (int)layers_.size();
  auto* layer = new Conv2DLayer(idx, inputHeight, inputWidth, kernelHeight, kernelWidth, strideH, strideW, numFilters, act, initStd);
  layers_.push_back(layer);

  wireConv2D(layers_[layers_.size() - 2], layer);
}

// =============================================================
// wireConv2D
// =============================================================
void Network::wireConv2D(Layer* prev, Conv2DLayer* curr)
{
  int inH = curr->inputHeight();
  int inW = curr->inputWidth();
  int kH = curr->kernelHeight();
  int kW = curr->kernelWidth();
  int sH = curr->strideH();
  int sW = curr->strideW();
  int numFilters = curr->numFilters();
  int outH = curr->outputHeight();
  int outW = curr->outputWidth();

  int neuronsPerFilter = outH * outW;

  for (int f = 0; f < numFilters; f++)
  {
    for (int oh = 0; oh < outH; oh++)
    {
      for (int ow = 0; ow < outW; ow++)
      {
        int dstIdx = (f * neuronsPerFilter) + oh * outW + ow;
        Neuron* dst = curr->neurons_[dstIdx];

        for (int kh = 0; kh < kH; kh++)
        {
          for (int kw = 0; kw < kW; kw++)
          {
            int srcH = oh * sH + kh;
            int srcW = ow * sW + kw;

            if (srcH >= inH || srcW >= inW)
              continue;

            int srcIdx = srcH * inW + srcW;
            Neuron* src = prev->neurons_[srcIdx];

            auto* c = new Connection();
            c->from = src;
            c->to = dst;
            c->filter = curr->filters_[f];
            c->filterSlot = kh * kW + kw;
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

// =============================================================
//  addMaxPool2D
//  No connections created — MaxPool accesses the previous layer
//  neurons directly via prevLayer_ pointer.
//  User provides input dimensions explicitly.
// =============================================================
void Network::addMaxPool2D(int inputH, int inputW, int numChannels,             
                           int poolH, int poolW,
                           int strideH, int strideW)
{
  assert(!layers_.empty() && "add an input layer first");
  assert(layers_.back()->size() == numChannels * inputH * inputW
    && "addMaxPool2D: previous layer size does not match dimensions");

  int   idx = (int)layers_.size();
  auto* layer = new MaxPool2DLayer(idx,
                                   inputH, inputW, numChannels,
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
//
//bool Network::load(const std::string& filename)
//{
//  std::ifstream f(filename);
//  if (!f.is_open())
//  {
//    std::cerr << "Network::load — cannot open: " << filename << "\n";
//    return false;
//  }
//
//  // ── clear existing network ────────────────────────────────
//  for (auto* c : allConns_) delete c;
//  for (auto* l : layers_)   delete l;
//  allConns_.clear();
//  layers_.clear();
//  t_ = 0;
//
//  std::string line;
//  int version = 1;
//
//  // ── pass 1: architecture ─────────────────────────────────
//  while (std::getline(f, line))
//  {
//    if (line.empty() || line[0] == '#') continue;
//    if (line == "weights") break;           // stop at weights section
//
//    std::istringstream ss(line);
//    std::string token;
//    ss >> token;
//
//    if (token == "version") { ss >> version; continue; }
//    if (token == "layers") { continue; }   // count not needed — just read
//    if (token != "layer") { continue; }
//
//    int idx;
//    std::string type;
//    ss >> idx >> type;
//
//    if (type == "Linear")
//    {
//      int size; std::string act;
//      ss >> size >> act;
//      addLinear(size, parseAct(act));
//    }
//    else if (type == "Dropout")
//    {
//      double rate;
//      ss >> rate;          // size is implicit — same as previous layer
//      addDropout(rate);
//    }
//    else if (type == "Residual")
//    {
//      int size, skipIdx; std::string act;
//      ss >> size >> act >> skipIdx;
//      addResidual(size, parseAct(act), skipIdx);
//    }
//    else if (type == "Conv1D")
//    {
//      // size kernelSize stride numFilters act
//      int size, kernelSize, stride, numFilters;
//      std::string act;
//      ss >> size >> kernelSize >> stride >> numFilters >> act;
//      addConv1D(kernelSize, parseAct(act), stride, numFilters);
//    }
//    else if (type == "Conv2D")
//    {
//      // size inputH inputW kernelH kernelW strideH strideW numFilters act
//      int size, inputH, inputW, kernelH, kernelW, strideH, strideW, numFilters;
//      std::string act;
//      ss >> size >> inputH >> inputW
//        >> kernelH >> kernelW
//        >> strideH >> strideW
//        >> numFilters >> act;
//      addConv2D(inputH, inputW, kernelH, kernelW,
//        strideH, strideW, numFilters,
//        parseAct(act), std::sqrt(1.0 / (kernelH * kernelW)));
//    }
//    else if (type == "MaxPool2D")
//    {
//      // size inputH inputW numChans poolH poolW strideH strideW
//      int size, inputH, inputW, numChans, poolH, poolW, strideH, strideW;
//      ss >> size >> inputH >> inputW >> numChans
//        >> poolH >> poolW >> strideH >> strideW;
//      addMaxPool2D(inputH, inputW, numChans, poolH, poolW, strideH, strideW);
//    }
//    else if (type == "Softmax")
//    {
//      addSoftmax();
//    }
//    else
//    {
//      std::cerr << "Network::load — unknown layer type: " << type << "\n";
//      return false;
//    }
//  }
//
//  if (layers_.empty())
//  {
//    std::cerr << "Network::load — no layers found in file\n";
//    return false;
//  }
//
//  // ── pass 2: weights ───────────────────────────────────────
//  // We track per-layer cursors for dense weights so we can
//  // fill them in neuron-major, connection-major order —
//  // the same order save() wrote them.
//
//  // For each layer, a flat list of dense weight values ready to apply
//  std::unordered_map<int, std::vector<double>> denseWeights; // layerIdx → values
//  std::unordered_map<int, std::vector<double>> biasValues;   // layerIdx → values
//
//  while (std::getline(f, line))
//  {
//    if (line.empty() || line[0] == '#') continue;
//
//    std::istringstream ss(line);
//    std::string tag, kind;
//    ss >> tag >> kind;
//
//    // tag = "L3", extract index
//    if (tag.empty() || tag[0] != 'L') continue;
//    int layerIdx = std::stoi(tag.substr(1));
//
//    if (layerIdx < 0 || layerIdx >= (int)layers_.size())
//    {
//      std::cerr << "Network::load — layer index out of range: " << layerIdx << "\n";
//      continue;
//    }
//
//    Layer* l = layers_[layerIdx];
//
//    if (kind == "bias")
//    {
//      auto vals = readDoubles(ss);
//      if ((int)vals.size() != l->size())
//      {
//        std::cerr << "Network::load — bias count mismatch on L"
//          << layerIdx << " (got " << vals.size()
//          << ", expected " << l->size() << ")\n";
//        return false;
//      }
//      for (int i = 0; i < l->size(); i++)
//        l->neurons_[i]->bias = vals[i];
//    }
//    else if (kind == "dense")
//    {
//      // store for second pass — we need all biases loaded first
//      // (not strictly required, but keeps the code clean)
//      denseWeights[layerIdx] = readDoubles(ss);
//    }
//    else if (kind == "filter")
//    {
//      int filterIdx;
//      ss >> filterIdx;
//      auto vals = readDoubles(ss);
//
//      Filter* filter = nullptr;
//      if (auto* cl = dynamic_cast<Conv1DLayer*>(l))
//      {
//        if (filterIdx < (int)cl->filters_.size())
//          filter = cl->filters_[filterIdx];
//      }
//      else if (auto* cl = dynamic_cast<Conv2DLayer*>(l))
//      {
//        if (filterIdx < (int)cl->filters_.size())
//          filter = cl->filters_[filterIdx];
//      }
//
//      if (!filter)
//      {
//        std::cerr << "Network::load — filter not found: L"
//          << layerIdx << " filter " << filterIdx << "\n";
//        return false;
//      }
//
//      if ((int)vals.size() != filter->size())
//      {
//        std::cerr << "Network::load — filter weight count mismatch on L"
//          << layerIdx << " filter " << filterIdx << "\n";
//        return false;
//      }
//
//      // Filter doesn't expose a setter, so we update via accumulateGrad
//      // trick: zero weights, then use the Adam state — actually we need
//      // direct write access. We expose it via a friend or by using
//      // the fact that Filter::weights_ is private.
//      // CLEANEST SOLUTION: add Filter::setWeight(int slot, double val)
//      // See note below — this requires one small addition to Filter.h
//      for (int k = 0; k < filter->size(); k++)
//        filter->setWeight(k, vals[k]);
//    }
//  }
//
//  // ── apply dense weights ───────────────────────────────────
//  for (auto& [layerIdx, vals] : denseWeights)
//  {
//    Layer* l = layers_[layerIdx];
//    int cursor = 0;
//    for (auto* n : l->neurons_)
//    {
//      for (auto* c : n->inConns)
//      {
//        if (c->trainable && c->filter == nullptr)
//        {
//          if (cursor >= (int)vals.size())
//          {
//            std::cerr << "Network::load — dense weight underflow on L"
//              << layerIdx << "\n";
//            return false;
//          }
//          c->weight = vals[cursor++];
//        }
//      }
//    }
//    if (cursor != (int)vals.size())
//    {
//      std::cerr << "Network::load — dense weight count mismatch on L"
//        << layerIdx << " (used " << cursor
//        << ", had " << vals.size() << ")\n";
//      return false;
//    }
//  }
//
//  std::cout << "Network loaded from: " << filename
//    << " (" << layers_.size() << " layers)\n";
//  return true;
//}
