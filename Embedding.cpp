#include "Embedding.h"
#include <assert.h>

Embedding::Embedding(int index, int seqLength, int vocabSize, int embeddingDim, Layer* prev)
  :Layer(index, Activation::LINEAR),
  inputSeqLen_(seqLength),
  vocabSize_(vocabSize),
  embeddingDim_(embeddingDim),
  prev_(prev),
  rng_(42)
{
  assert(vocabSize >= 1    && "Vocublary should be a positive number");
  assert(embeddingDim >= 1 && "Embeddings should be a positive number");
  assert(seqLength >= 1    && "Sequence length  should be a positive number");
  
  weights_.resize(vocabSize_, std::vector<double>(embeddingDim_));
  gradients_.resize(vocabSize_, std::vector<double>(embeddingDim_, 0.0));
  adam_.resize(vocabSize_, std::vector<AdamState>(embeddingDim_));

  // Xavier-like or small random init
  std::normal_distribution<double> dist(0.0, 0.01); // or better scaling
  for (int v = 0; v < vocabSize_; ++v)
  {
    for (int d = 0; d < embeddingDim_; ++d)
    {
      weights_[v][d] = dist(rng_);
    }
  }

  createNeurons(seqLength * embeddingDim, "L" + std::to_string(index) + "-E");
}

Embedding::~Embedding()
{}

void Embedding::forward()
{
  usedTokens_.clear();
  for (int i = 0; i < inputSeqLen_; i++)
  {
    int token_id = (int)prev_->neurons_[i]->output;
    usedTokens_.insert(token_id);
    for (int j = 0; j < embeddingDim_; j++)
    {
      int index = i * embeddingDim_ + j;
      neurons_[index]->output = weights_[token_id][j];
    }
  }
}

void Embedding::backward()
{
  // Propagate the error from the following layer to us.
  for (auto* n : neurons_)
  {
    n->error = 0.0;
    for (auto* c : n->outConns)
    {
      double w = (c->filter != nullptr) ? c->filter->weight(c->filterSlot) : c->weight;
      n->error += w * c->to->error;
    }
  }

  // Updating the embeddings of our used tokens
  // by the errors propagated to our neurons from 
  // the following layer.

  // token ID   |  Embeddings
  //    0       | 0.1  -0.2  0.5,  -0.1
  //    1       | 0.3   0.5  -0.4, -0.3
  //    2       | 0.7  -0.8  0.6,  -0.5
  //    3       | 0.2  -0.3  -0.4, -0.9
  //    ..      | 0.3   0.9  0.8,  -0.8
  //    ..      | 0.8   0.1  0.9,  -0.7
  //  vocabSize | 0.9  -0.4  -0.1, -0.6

  // Our neurons are flattened like this (assume 3 tokens with 2 embeddings
  // N0 Toke0_Emb0
  // N1 Toke0_Emb1
  // N2 Toke1_Emb0
  // N3 Toke1_Emb1
  // N4 Toke2_Emb0
  // N5 Toke2_Emb1

  for (int pos = 0; pos < inputSeqLen_; ++pos)
  {
    int token = (int)prev_->neurons_[pos]->output;
    for (int d = 0; d < embeddingDim_; ++d)
    {
      int neuronIdx = pos * embeddingDim_ + d;
      gradients_[token][d] += neurons_[neuronIdx]->error;
    }
  }
}

void Embedding::step(double lr, int t)
{
  for (int token : usedTokens_)
  {
    for (int d = 0; d < embeddingDim_; ++d)
    {
      adam_[token][d].update(weights_[token][d], gradients_[token][d], lr, t);
    }
  }
}

void Embedding::zero_grad()
{
  usedTokens_.clear();
  for (int i = 0; i < vocabSize_; i++)
  {
    for (int e = 0; e < embeddingDim_; e++)
    {
      gradients_[i][e] = 0;
    }
   }
}

