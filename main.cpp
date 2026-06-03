#include "Network.h"
#include <iostream>
#include <iomanip>
#include <vector>

int Conv1DExample()
{
  std::cout << "=== Conv1D Test with Stride ===\n\n";

  Network net(42);

  // New shape-aware way
  net.addLinear(7, Activation::LINEAR);           // input: 7 features (flattened)
  net.addLinear(120, Activation::LEAKY_RELU);
  net.addDropout(0.3);
  //net.addResidual(7, Activation::RELU, 0);

  // New simplified Conv1D - no need to pass input size!
  net.addConv1D(3, Activation::LEAKY_RELU, 1, 25);

  // New simplified MaxPool1D
  net.addMaxPool1D(2, 1);   // poolSize=2, stride=1

  net.addResidual(7, Activation::TANH, 0);
  //net.addDropout(0.3);
  net.addLinear(2, Activation::LINEAR);
  net.addSoftmax();

  net.summary();

  // Test data
  std::vector<std::vector<double>> X = {
      {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
      {1.0, 0.8, 0.6, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.6, 0.8, 1.0}
  };

  std::vector<std::vector<double>> Y = { {1, 0}, {0, 1}, {1, 0}, {0, 1} };

  std::cout << "Training Conv1D...\n";
  for (int epoch = 0; epoch < 1000; ++epoch)   // reduced for faster testing
  {
    double loss = 0.0;
    for (size_t i = 0; i < X.size(); ++i)
    {
      loss += net.train(X[i], Y[i], 0.0005);
    }
    if (epoch % 200 == 0)
    {
      std::cout << "Epoch " << epoch << " - Loss: "
        << std::setprecision(6) << loss / 4.0 << "\n";
    }
  }

  std::cout << "\nFinal Predictions:\n";
  for (size_t i = 0; i < X.size(); ++i)
  {
    auto pred = net.predict(X[i]);
    std::cout << "Input " << i << " -> " << pred[0] << ", " << pred[1]
      << " (target = " << Y[i][0] << ", " << Y[i][1] << ")\n";
  }
  return 0;
}

int Conv2DExample()
{
  std::cout << "=== Simple Conv2D + MaxPool Test ===\n\n";

  Network net(42);

  // Input: 28x28 image flattened
  net.addLinear(784, Activation::LINEAR);
  net.addConv2D(28, 28, 3, 3, 1, 1, 8, Activation::RELU, 0.3);
  net.addMaxPool2D(26, 26, 8, 2, 2, 2, 2);
  net.addConv2D(13, 13, 3, 3, 1, 1, 16, Activation::RELU, 0.3);

  // MaxPool 2: 11x11 input, pool 2x2, stride 2x2
  // output: 16 x 5 x 5 = 400 neurons
  // note: (11-2)/2 + 1 = 5
  net.addMaxPool2D(11, 11, 16, 2, 2, 2, 2);
  net.addLinear(10, Activation::LINEAR);
  net.addSoftmax();

  net.summary();

  // Dummy test data
  std::vector<double> input(784, 0.5);
  std::vector<double> target = { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };  // class 4

  std::cout << "\nTraining...\n";
  for (int i = 0; i < 100; i++)
  {
    double loss = net.train(input, target, 0.1);
    std::cout << "Step : " << i << ",  Loss = " << loss << std::endl;
  }

  std::cout << "\nPrediction (probabilities):\n";
  auto pred = net.predict(input);
  for (size_t i = 0; i < pred.size(); ++i)
  {
    std::cout << "  Class " << i << ": " << pred[i] << "\n";
  }

  return 0;
}

int main()
{
  Conv1DExample();
  //Conv2DExample();
  return 0;
}