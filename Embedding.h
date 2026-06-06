#pragma once
#include "Layer.h"
#include <random>
#include <set>

class Embedding : public Layer
{
public:
  Embedding(int index, int seqLength, int vocabSize, int embeddingDim, Layer* prev);

  ~Embedding();

  void forward() override;
  void backward() override;
  void step(double lr, int t) override;
  void zero_grad() override;

  std::string typeName() const override { return "Embedding"; }

  // Shape
  int channels() const override { return embeddingDim_; }
  int height()   const override { return 1; }
  int width()    const override { return inputSeqLen_; }

public:
  int vocabSize_    = 1;
  int embeddingDim_ = 1;
  int inputSeqLen_  = 1;
  
  Layer* prev_;

  std::vector<std::vector<double>> weights_;
  std::vector<std::vector<double>> gradients_;
  std::vector<std::vector<AdamState>> adam_;
  std::set<int> usedTokens_;

  std::mt19937 rng_;
};