#include "Network.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

void demoXOR();
void demoResidual();
void demoDeepSkip();
void demoConv1D();

// =============================================================
//  Demo 1 — XOR with Linear + Dropout
// =============================================================
void demoXOR()
{
  std::cout << "=== Demo 1: XOR  (Linear + Dropout) ===\n";

  Network nn(42);
  nn.addLinear (2,  Activation::LINEAR);
  nn.addLinear (8,  Activation::TANH);
  nn.addDropout(0.1);
  nn.addLinear (1,  Activation::SIGMOID);
  nn.summary();

  std::vector<std::vector<double>> X = {{0,0},{0,1},{1,0},{1,1}};
  std::vector<std::vector<double>> Y = {{0},  {1},  {1},  {0}};

  std::cout << std::fixed << std::setprecision(6);
  for (int epoch = 0; epoch < 10000; epoch++)
  {
    double loss = 0.0;
    for (int i = 0; i < 4; i++)
    {
      loss += nn.train(X[i], Y[i], 0.001);
    }
    if (epoch % 2000 == 0)
    {
      std::cout << "  epoch " << std::setw(5) << epoch
        << "  loss = " << loss / 4.0 << "\n";
    }
  }

  std::cout << "\nFinal predictions:\n";
  for (int i = 0; i < 4; i++) 
  {
    auto pred = nn.predict(X[i]);
    std::cout << "  " << X[i][0] << " XOR " << X[i][1]
      << "  ->  " << std::setprecision(4) << pred[0]
      << "  (target=" << Y[i][0] << ")\n";
  }
}

// =============================================================
//  Demo 2 — Residual skip one layer behind
//  L0(2) -> L1(8) -> Residual(8, skip=L1) -> L3(1)
// =============================================================
void demoResidual()
{
  std::cout << "\n=== Demo 2: Residual skip one layer ===\n";

  Network nn(42);
  nn.addLinear  (2, Activation::LINEAR);
  nn.addLinear  (8, Activation::TANH);
  nn.addResidual(8, Activation::TANH, 1);  // skip from L1
  nn.addLinear  (1, Activation::SIGMOID);
  nn.summary();

  std::vector<std::vector<double>> X = {{0,0},{0,1},{1,0},{1,1}};
  std::vector<std::vector<double>> Y = {{0},  {1},  {1},  {0}};

  std::cout << std::fixed << std::setprecision(6);
  for (int epoch = 0; epoch < 10000; epoch++) 
  {
    double loss = 0.0;
    for (int i = 0; i < 4; i++)
    {
      loss += nn.train(X[i], Y[i], 0.001);
    }
    if (epoch % 2000 == 0)
    {
      std::cout << "  epoch " << std::setw(5) << epoch
        << "  loss = " << loss / 4.0 << "\n";
    }
  }

  std::cout << "\nFinal predictions:\n";
  for (int i = 0; i < 4; i++)
  {
    auto pred = nn.predict(X[i]);
    std::cout << "  " << X[i][0] << " XOR " << X[i][1] << "  --> " << std::setprecision(4) << pred[0] << "  (target=" << Y[i][0] << ")\n";
  }
}

// =============================================================
//  Demo 3 — Deep network, skip across 3 layers
//  L0(4) -> L1(4) -> L2(4) -> L3(4) -> Residual(4, skip=L0) -> L5(1)
// =============================================================
void demoDeepSkip()
{
  std::cout << "\n=== Demo 3: Deep residual — skip across 3 layers ===\n";

  Network nn(42);
  nn.addLinear  (4, Activation::LINEAR);
  nn.addLinear  (4, Activation::TANH);
  nn.addLinear  (4, Activation::TANH);
  nn.addLinear  (4, Activation::TANH);
  nn.addResidual(4, Activation::TANH, 0);  // skip from L0
  nn.addLinear  (1, Activation::SIGMOID);
  nn.summary();

  // odd parity: output 1 if odd number of 1s in input
  std::vector<std::vector<double>> X = 
  {
    {0,0,0,0},{0,0,0,1},{0,0,1,0},{0,0,1,1},
    {0,1,0,0},{0,1,0,1},{0,1,1,0},{0,1,1,1},
    {1,0,0,0},{1,0,0,1},{1,0,1,0},{1,0,1,1},
    {1,1,0,0},{1,1,0,1},{1,1,1,0},{1,1,1,1},
  };
  
  std::vector<std::vector<double>> Y;
  for (auto& x : X)
  {
    Y.push_back({ (double)((int)(x[0] + x[1] + x[2] + x[3]) % 2) });
  }

  std::cout << std::fixed << std::setprecision(6);
  for (int epoch = 0; epoch < 15000; epoch++) 
  {
    double loss = 0.0;
    for (int i = 0; i < 16; i++)
    {
      loss += nn.train(X[i], Y[i], 0.001);
    }
    if (epoch % 3000 == 0)
    {
      std::cout << "  epoch " << std::setw(5) << epoch << "  loss = " << loss / 16.0 << "\n";
    }
  }

  std::cout << "\nSample predictions:\n";
  for (int i = 0; i < 4; i++) 
  {
    auto pred = nn.predict(X[i]);
    std::cout << "  [" << X[i][0]<<","<<X[i][1]<<"," << X[i][2]<<","<<X[i][3] << "]" << " --> " << std::setprecision(4) << pred[0]
      << "  (target=" << Y[i][0] << ")\n";
  }
}

// =============================================================
//  Main
// =============================================================
// =============================================================
//  Demo 4 — Conv1D
//  input(7) → Conv1D(kernel=3) → Linear(1, sigmoid)
//
//  Task: left-heavy input [1,1,1,0,0,0,0] → 1
//        right-heavy input [0,0,0,0,1,1,1] → 0
//
//  The conv filter learns to detect the pattern position.
// =============================================================
void demoConv1D()
{
  std::cout << "\n=== Demo 4: Conv1D ===\n";

  Network nn(42);
  nn.addLinear (7, Activation::LINEAR);    // input: 7 neurons
  nn.addConv1D (3, Activation::TANH);      // conv: kernel=3, output=5
  nn.addLinear(7, Activation::TANH);
  nn.addResidual(5, Activation::TANH, 1);
  nn.addLinear (1, Activation::SIGMOID);   // output
  nn.summary();

  std::vector<std::vector<double>> X = 
  {
    {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0},   // left heavy  → 1
    {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0},   // right heavy → 0
    {1.0, 0.8, 0.6, 0.0, 0.0, 0.0, 0.0},   // left        → 1
    {0.0, 0.0, 0.0, 0.0, 0.6, 0.8, 1.0},   // right       → 0
  };
  
  std::vector<std::vector<double>> Y = {{1},{0},{1},{0}};

  std::cout << std::fixed << std::setprecision(6);
  for (int epoch = 0; epoch < 4000; epoch++)
  {
    double loss = 0.0;
    for (int i = 0; i < 4; i++)
    {
      loss += nn.train(X[i], Y[i], 0.001);    
    }
    if (epoch % 2000 == 0)
    {
      std::cout << "  epoch " << std::setw(5) << epoch << "  loss = " << loss / 4.0 << "\n";
    }
  }
  nn.save("xor.model");
  std::cout << "\nFinal predictions:\n";

  for (int i = 0; i < 4; i++) 
  {
    auto pred = nn.predict(X[i]);
    std::cout << "  -> " << std::setprecision(4) << pred[0] << "  (target=" << Y[i][0] << ")\n";
  }
}

int main()
{
  //demoXOR();
  //demoResidual();
  //demoDeepSkip();
  demoConv1D();
  return 0;
}
