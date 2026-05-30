// appliance_main.cpp
//
// Natural language appliance controller
// Pipeline: sentence → tokenise → encode → Conv1D network → binary outputs
//
// Build:
//   g++ -std=c++17 -O2 -o appliance appliance_main.cpp \
//       Activation.cpp Adam.cpp Connection.cpp Filter.cpp Layer.cpp \
//       Linear.cpp Dropout.cpp Residual.cpp Conv1d.cpp Conv2d.cpp \
//       Maxpool.cpp Softmax.cpp Neuron.cpp Network.cpp
// =============================================================

#include "Network.h"
#include "Vocabulary.h"
#include "ApplianceDataset.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

// =============================================================
//  Config
// =============================================================
static const int SEQ_LEN      = 12;    // max tokens per sentence
static const int HIDDEN       = 32;    // hidden layer width
static const int EPOCHS       = 300;
static const double LR        = 0.005;
static const double THRESHOLD = 0.5;  // output > 0.5 → ON

// =============================================================
//  printState — show appliance on/off state nicely
// =============================================================
void printState(const std::vector<double>& outputs)
{
  std::cout << "\n  Appliance states:\n";
  for (int i = 0; i < NUM_APPLIANCES; i++)
  {
    bool on = outputs[i] > THRESHOLD;
    std::cout << "    " << std::left << std::setw(20)
      << APPLIANCE_NAMES[i]
      << (on ? "[ ON  ]" : "[ off ]")
        << "  (" << std::fixed << std::setprecision(2)
        << outputs[i] << ")\n";
  }
  std::cout << "\n";
}

// =============================================================
//  evaluate — accuracy over dataset
// =============================================================
double evaluate(Network& net, const Vocabulary& vocab, const std::vector<Sample>& data)
{
  int correct = 0, total = 0;
  for (const auto& s : data)
  {
    auto input = vocab.encode(s.sentence, SEQ_LEN);
    auto pred = net.predict(input);

    bool sampleCorrect = true;
    for (int i = 0; i < NUM_APPLIANCES; i++)
    {
      bool predOn = pred[i] > THRESHOLD;
      bool targetOn = s.labels[i] > THRESHOLD;
      if (predOn != targetOn) { sampleCorrect = false; break; }
    }
    if (sampleCorrect) correct++;
    total++;
  }
  return 100.0 * correct / total;
}

// =============================================================
//  main
// =============================================================
int main()
{
  std::cout << "=== Natural Language Appliance Controller ===\n\n";

  // ── 1. Build dataset ─────────────────────────────────────
  auto dataset = buildDataset();
  std::cout << "Dataset: " << dataset.size() << " samples\n";

  // ── 2. Build vocabulary ──────────────────────────────────
  Vocabulary vocab;
  std::vector<std::string> sentences;
  for (const auto& s : dataset) sentences.push_back(s.sentence);
  vocab.build(sentences);
  std::cout << "Vocabulary size: " << vocab.size() << " tokens\n";
  std::cout << "Sequence length: " << SEQ_LEN << "\n\n";

  // ── 3. Build network ─────────────────────────────────────
  //
  //  Input:   SEQ_LEN neurons (one per token position)
  //  Conv1D:  kernel=3 → detects 3-word phrases like "turn on the"
  //  Conv1D:  kernel=3 → composes phrase-level features
  //  Linear:  HIDDEN neurons — global reasoning
  //  Linear:  NUM_APPLIANCES neurons with SIGMOID — one per appliance
  //
  Network net(42);
  net.addLinear(SEQ_LEN, Activation::LINEAR);
  net.addConv1D(3, Activation::RELU, 1, 4);   // 4 filters, kernel=3
  net.addConv1D(3, Activation::RELU, 1, 4);   // 4 filters, kernel=3
  net.addConv1D(3, Activation::RELU, 1, 4);   // 4 filters, kernel=3
  //net.addConv1D(3, Activation::RELU, 1, 4);   // 4 filters, kernel=3
  net.addLinear(HIDDEN, Activation::RELU);
  net.addLinear(NUM_APPLIANCES, Activation::SIGMOID);

  net.summary();

  // ── 4. Training ──────────────────────────────────────────
  std::cout << "=== Training ===\n";

  for (int epoch = 0; epoch < EPOCHS; epoch++)
  {
    double totalLoss = 0.0;

    // shuffle dataset each epoch
    std::vector<int> idx(dataset.size());
    std::iota(idx.begin(), idx.end(), 0);
    // simple Fisher-Yates with fixed seed offset
    for (int i = (int)idx.size() - 1; i > 0; i--)
    {
      int j = (epoch * 1000 + i * 7) % (i + 1);
      std::swap(idx[i], idx[j]);
    }

    for (int i : idx)
    {
      auto input = vocab.encode(dataset[i].sentence, SEQ_LEN);
      double loss = net.train(input, dataset[i].labels, LR);
      totalLoss += loss;
    }

    if ((epoch + 1) % 50 == 0)
    {
      double acc = evaluate(net, vocab, dataset);
      std::cout << "Epoch " << std::setw(4) << epoch + 1
        << " | Loss: " << std::fixed << std::setprecision(4)
        << totalLoss / dataset.size()
        << " | Accuracy: " << std::setprecision(1)
        << acc << "%\n";
    }
  }

  std::cout << "\n=== Training complete ===\n";
  double finalAcc = evaluate(net, vocab, dataset);
  std::cout << "Final accuracy: " << finalAcc << "%\n\n";

  // ── 5. Interactive command line ───────────────────────────
  std::cout << "=== Appliance Controller Ready ===\n";
  std::cout << "Type a command (or 'quit' to exit)\n";
  std::cout << "Examples:\n";
  std::cout << "  turn on the lights\n";
  std::cout << "  lights and fan on\n";
  std::cout << "  switch off the ac\n";
  std::cout << "  turn on everything\n\n";

  std::string command;
  while (true)
  {
    std::cout << "> ";
    std::getline(std::cin, command);

    if (command == "quit" || command == "exit") break;
    if (command.empty()) continue;

    auto input = vocab.encode(command, SEQ_LEN);
    auto outputs = net.predict(input);

    std::cout << "\nCommand: \"" << command << "\"";
    printState(outputs);
  }

  std::cout << "Goodbye.\n";
  return 0;
}
