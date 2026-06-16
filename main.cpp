// word2vec.cpp
//
// Skip-Gram Word2Vec using the neural network library.
//
// Architecture (per training pair):
//   Input:  one-hot vector [vocabSize]       <- center word
//   Linear(vocabSize, LINEAR)                <- input layer (receives one-hot)
//   Linear(EMB_DIM,   LINEAR)                <- ** THIS IS THE EMBEDDING TABLE **
//   Linear(vocabSize, LINEAR)                <- output projection (logits)
//   Softmax                                  <- probability over vocab
//   Target: one-hot vector [vocabSize]       <- context word (MSE loss)
//
// After training the output layer is discarded.
// The weights of layer index 1 (EMB_DIM x vocabSize) are the word vectors.
// Row i of that weight matrix = embedding for word i.
//
// Compile (adjust paths to your library):
//   g++ -std=c++17 -O2 -I. word2vec.cpp \
//       Network.cpp Linear.cpp Softmax.cpp Neuron.cpp Connection.cpp \
//       Adam.cpp Layer.cpp Activation.cpp Embedding.cpp Residual.cpp \
//       Dropout.cpp LayerNorm.cpp Filter.cpp Conv1d.cpp Conv2d.cpp \
//       AttentionLayer.cpp AttentionScore.cpp Maxpool1D.cpp Maxpool2D.cpp \
//       SelfAttention.cpp \
//       -o word2vec
//
// Run:
//   ./word2vec corpus.txt

#include "Network.h"
#include "Vocabulary.h"
#include "Linear.h"       // for casting layers() to extract weights

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

// ================================================================
//  Hyperparameters — tweak these
// ================================================================

static constexpr int    EMB_DIM = 64;    // embedding vector size
static constexpr int    WINDOW_SIZE = 5;     // context window (each side)
static constexpr double LR = 0.025; // Adam learning rate
static constexpr int    EPOCHS = 3;
static constexpr int    BATCH_SIZE = 1024;   // was 124
static constexpr int    PRINT_EVERY = 20;
static constexpr int    MIN_FREQ = 1;     // drop words rarer than this

// ================================================================
//  load_corpus
//  Reads a plain-text file and returns all words as a flat token list.
//  Uses the same lowercase+strip logic as Vocabulary::tokenise().
// ================================================================

static std::vector<std::string> load_corpus(const std::string& path)
{
  std::ifstream f(path);
  if (!f.is_open())
  {
    std::cerr << "Cannot open corpus file: " << path << "\n";
    std::exit(1);
  }

  std::vector<std::string> tokens;
  std::string line;
  while (std::getline(f, line))
  {
    // Reuse Vocabulary's tokeniser (lowercase + strip punctuation)
    for (auto& tok : Vocabulary::tokenise(line))
      tokens.push_back(tok);
  }
  return tokens;
}

// ================================================================
//  build_pairs
//  Given a flat token list and a Vocabulary, generates all
//  (center_word_id, context_word_id) training pairs within the window.
//  Words with unknown ids (<UNK>=1) are skipped as centers.
// ================================================================

static std::vector<std::pair<int, int>> build_pairs(
  const std::vector<std::string>& tokens,
  const Vocabulary& vocab,
  int windowSize)
{
  std::vector<std::pair<int, int>> pairs;
  pairs.reserve(tokens.size() * windowSize * 2);

  for (int i = 0; i < (int)tokens.size(); ++i)
  {
    int centerId = vocab.index(tokens[i]);
    if (centerId <= 1) continue;   // skip <PAD> and <UNK>

    int lo = std::max(0, i - windowSize);
    int hi = std::min((int)tokens.size() - 1, i + windowSize);

    for (int j = lo; j <= hi; ++j)
    {
      if (j == i) continue;
      int ctxId = vocab.index(tokens[j]);
      if (ctxId <= 1) continue;  // skip unknown context words

      pairs.emplace_back(centerId, ctxId);
    }
  }
  return pairs;
}

// ================================================================
//  make_one_hot
//  Returns a dense float vector of length vocabSize with a 1.0
//  at position idx and 0.0 everywhere else.
// ================================================================

static std::vector<double> make_one_hot(int idx, int vocabSize)
{
  std::vector<double> v(vocabSize, 0.0);
  v[idx] = 1.0;
  return v;
}

// ================================================================
//  cosine_similarity
// ================================================================

static double cosine_sim(const std::vector<double>& a,
  const std::vector<double>& b)
{
  assert(a.size() == b.size());
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (int i = 0; i < (int)a.size(); ++i)
  {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  double denom = std::sqrt(na) * std::sqrt(nb);
  return (denom < 1e-12) ? 0.0 : dot / denom;
}

// ================================================================
//  extract_embeddings
//
//  After training, the embedding matrix lives in the connections of
//  layer index 1 (the EMB_DIM-wide Linear layer).
//
//  Your library wires layers with Connection objects whose `weight`
//  field holds the trained value.  For the embedding layer:
//    - It has EMB_DIM neurons.
//    - Each neuron has vocabSize inConns.
//    - inConn[v]->weight is the weight from input neuron v to
//      embedding neuron e.
//
//  So embedding[v][e] = layers()[1]->neurons_[e]->inConns[v]->weight
//  We transpose this to get embedding[v] as a vector of length EMB_DIM.
// ================================================================

static std::vector<std::vector<double>> extract_embeddings(
  const Network& net,
  int vocabSize,
  int embDim)
{
  // Layer 0 = input Linear(vocabSize)
  // Layer 1 = embedding Linear(embDim)  <-- weights we want
  const Layer* embLayer = net.layers()[1];
  assert((int)embLayer->neurons_.size() == embDim);

  std::vector<std::vector<double>> emb(vocabSize, std::vector<double>(embDim, 0.0));

  for (int e = 0; e < embDim; ++e)
  {
    const Neuron* n = embLayer->neurons_[e];
    assert((int)n->inConns.size() == vocabSize);

    for (int v = 0; v < vocabSize; ++v)
    {
      emb[v][e] = n->inConns[v]->weight;
    }
  }
  return emb;
}

// ================================================================
//  nearest_neighbors
//  Returns the top-K most similar words to `query` by cosine sim.
// ================================================================

static std::vector<std::pair<double, std::string>> nearest_neighbors(
  const std::string& query,
  const Vocabulary& vocab,
  const std::vector<std::vector<double>>& emb,
  int k = 5)
{
  int qid = vocab.index(query);
  if (qid <= 1)
  {
    std::cerr << "  Word '" << query << "' not in vocabulary.\n";
    return {};
  }

  std::vector<std::pair<double, std::string>> scores;
  scores.reserve(vocab.size());

  for (int v = 2; v < vocab.size(); ++v)  // skip <PAD>=0, <UNK>=1
  {
    if (v == qid) continue;
    double sim = cosine_sim(emb[qid], emb[v]);
    scores.emplace_back(sim, vocab.word(v));
  }

  std::sort(scores.begin(), scores.end(),
    [](auto& a, auto& b) { return a.first > b.first; });

  if ((int)scores.size() > k) scores.resize(k);
  return scores;
}

// ================================================================
//  main
// ================================================================

int main(int argc, char** argv)
{
  // ── 1. Load corpus ──────────────────────────────────────────
  std::string corpusPath = (argc > 1) ? argv[1] : "corpus.txt";
  std::cout << "Loading corpus: " << corpusPath << " ...\n";
  std::vector<std::string> tokens = load_corpus(corpusPath);
  std::cout << "  Total tokens : " << tokens.size() << "\n";

  // ── 2. Build vocabulary ─────────────────────────────────────
  //  Vocabulary::build() takes a list of sentences, but we can wrap
  //  each token as a single-word "sentence" to register all words.
  //  Alternatively, join into one string — both work identically.
  Vocabulary vocab;
  vocab.build(tokens);   // pass flat token list — build() iterates words
  const int V = vocab.size();
  std::cout << "  Vocabulary   : " << V << " tokens (incl. <PAD>,<UNK>)\n\n";

  if (V < 3)
  {
    std::cerr << "Corpus too small — need at least one real word.\n";
    return 1;
  }

  // ── 3. Generate (center, context) training pairs ────────────
  std::cout << "Generating skip-gram pairs (window=" << WINDOW_SIZE << ")...\n";
  auto pairs = build_pairs(tokens, vocab, WINDOW_SIZE);
  std::cout << "  Training pairs: " << pairs.size() << "\n\n";

  if (pairs.empty())
  {
    std::cerr << "No training pairs found. Check your corpus.\n";
    return 1;
  }

  // ── 4. Build network ────────────────────────────────────────
  //
  //  Input layer  : Linear(V)          receives one-hot center word
  //  Embedding    : Linear(EMB_DIM)    <-- the embedding table
  //  Output proj  : Linear(V)          predicts context word logits
  //  Softmax                           turns logits into probabilities
  //
  //  MSE loss against one-hot target drives the whole thing.
  //  The Softmax + MSE combination works for small vocabularies;
  //  for V > 10K switch to negative sampling (see future version).

  Network net(42);
  net.load("word2vec_model");
  //net.addLinear(V, Activation::LINEAR);   // [0] input (one-hot)
  //net.addLinear(EMB_DIM, Activation::LINEAR);   // [1] embedding  ← keep this
  //net.addLinear(V, Activation::LINEAR);   // [2] output projection
  //net.addSoftmax();                              // [3] probabilities
  net.summary();

  // ── 5. Training ─────────────────────────────────────────────
  const int numPairs = (int)pairs.size();
  std::mt19937 rng(42);
  std::vector<int> order(numPairs);
  std::iota(order.begin(), order.end(), 0);

  std::cout << "\nTraining for " << EPOCHS << " epoch(s)"
    << "  (batch=" << BATCH_SIZE
    << ", lr=" << LR << ", embDim=" << EMB_DIM << ") ...\n\n";

  auto wallStart = std::chrono::steady_clock::now();

  bool train = false;
  if (train)
  {
    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
      std::shuffle(order.begin(), order.end(), rng);

      double epochLoss = 0.0;
      int    numBatches = 0;

      // Build batch inputs/targets then call the batch train() overload
      std::vector<std::vector<double>> batchIn, batchTgt;
      batchIn.reserve(BATCH_SIZE);
      batchTgt.reserve(BATCH_SIZE);

      for (int i = 0; i < numPairs; ++i)
      {
        int centerId = pairs[order[i]].first;
        int contextId = pairs[order[i]].second;

        batchIn.push_back(make_one_hot(centerId, V));
        batchTgt.push_back(make_one_hot(contextId, V));

        bool lastSample = (i == numPairs - 1);
        bool batchFull = ((int)batchIn.size() == BATCH_SIZE);

        if (batchFull || lastSample)
        {
          // net.train(batch) does forward+backward+step for all
          // samples, accumulates gradients, then applies one Adam update
          double loss = net.train(batchIn, batchTgt, LR);
          epochLoss += loss;
          ++numBatches;

          batchIn.clear();
          batchTgt.clear();

          // Progress line every PRINT_EVERY batches
          if (numBatches % PRINT_EVERY == 0)
          {
            double avgLoss = epochLoss / numBatches;
            double pct = 100.0 * i / numPairs;

            auto now = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(now - wallStart).count();

            std::cout << "  epoch " << std::setw(2) << epoch + 1
              << "  [" << std::setw(5) << std::fixed
              << std::setprecision(1) << pct << "%]"
              << "  avg_loss=" << std::setprecision(5) << avgLoss
              << "  elapsed=" << std::setprecision(1) << secs << "s\n";
          }
        }
      }

      double avgLoss = (numBatches > 0) ? epochLoss / numBatches : 0.0;
      auto now = std::chrono::steady_clock::now();
      double s = std::chrono::duration<double>(now - wallStart).count();

      std::cout << "\n=== Epoch " << epoch + 1 << " complete"
        << "  avg_loss=" << std::fixed << std::setprecision(5) << avgLoss
        << "  time=" << std::setprecision(1) << s << "s ===\n\n";
    }
  }

  // ── 6. Save trained network ─────────────────────────────────
  //  net.save() serialises ALL layers including the embedding weights,
  //  so you can reload and call extract_embeddings() later.
  //net.save("word2vec_model");
  std::cout << "Model saved to 'word2vec_model'.\n\n";

  // ── 7. Extract embedding table ──────────────────────────────
  //  Pull word vectors out of layer [1]'s connection weights.
  auto embeddings = extract_embeddings(net, V, EMB_DIM);

  // ── 8. Save embedding table to a plain text file ────────────
  //  Format:  <word> <f1> <f2> ... <fEMB_DIM>
  //  Compatible with most embedding viewers and the Python
  //  gensim.models.KeyedVectors.load_word2vec_format() loader.
  {
    std::ofstream out("embeddings.txt");
    // Header line: numWords embDim  (gensim expects this)
    out << (V - 2) << " " << EMB_DIM << "\n";   // skip <PAD> and <UNK>
    for (int v = 2; v < V; ++v)
    {
      out << vocab.word(v);
      for (int e = 0; e < EMB_DIM; ++e)
        out << " " << std::fixed << std::setprecision(6) << embeddings[v][e];
      out << "\n";
    }
    std::cout << "Embeddings saved to 'embeddings.txt'.\n\n";
  }

  // ── 9. Quick similarity demo ─────────────────────────────────
  //  Print nearest neighbours for a handful of words so you can
  //  sanity-check the embeddings immediately after training.
  std::cout << "─────────────────────────────────────────\n";
  std::cout << "Nearest neighbours (cosine similarity)\n";
  std::cout << "─────────────────────────────────────────\n";

  // Pick up to 6 "interesting" words from the vocabulary
  // (skip the first two special tokens <PAD> and <UNK>)
  std::vector<std::string> probes;
  int step = std::max(1, (V - 2) / 120);
  for (int v = 2; v < V && (int)probes.size() < 120; v += step)
    probes.push_back(vocab.word(v));

  for (auto& word : probes)
  {
    std::cout << "\n  '" << word << "':\n";
    auto neighbors = nearest_neighbors(word, vocab, embeddings, 6);
    for (int n = 0; n < (int)neighbors.size(); ++n)
    {
      std::string w = neighbors[n].second;
      double      sim = neighbors[n].first;
      std::cout << "    " << std::setw(20) << std::left << w
        << "  sim=" << std::fixed << std::setprecision(4) << sim << "\n";
    }
  }

  std::cout << "\nDone.\n";
  return 0;
}
