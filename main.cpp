#include "Network.h"
#include <iostream>
#include <iomanip>
#include <vector>

int Conv1DExample()
{
  std::cout << "=== Conv1D Test with Stride ===\n\n";

  Network net(42);

  net.addLinear(7, Activation::LINEAR);    // input: 7 features (flattened)
  net.addLinear(40, Activation::LEAKY_RELU);
  net.addDropout(0.3);
  net.addConv1D(3, Activation::LEAKY_RELU, 1, 64);
  net.addMaxPool1D(2, 1);   // poolSize=2, stride=1
  net.addResidual(0, Activation::TANH);
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
  for (int epoch = 0; epoch < 200; ++epoch)   // reduced for faster testing
  {
    double loss = 0.0;
    for (size_t i = 0; i < X.size(); ++i)
    {
      loss += net.train(X[i], Y[i], 0.0005);
    }
    if (epoch % 10 == 0)
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

  // Input: 28x28 image 1 channel(binary or grey scale image).
  net.addLinear(28, 28, 1, Activation::LINEAR);
  net.addConv2D(3, 3, 1, 1, 8, Activation::RELU, 0.3);
  net.addMaxPool2D(2, 2, 2, 2);
  net.addConv2D(3, 3, 1, 1, 16, Activation::RELU, 0.3);
  net.addMaxPool2D(2, 2, 2, 2);
  net.addDropout(0.2);
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
  Conv2DExample();
  return 0;
}