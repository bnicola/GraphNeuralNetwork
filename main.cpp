#include "Network.h"
#include <iostream>
#include <iomanip>
#include <vector>

int Conv1DExample()
{
  std::cout << "=== Conv1D Test with Stride ===\n\n";

  Network net(42);

  net.addLinear(7,   Activation::LINEAR); // input
  net.addLinear(120, Activation::LEAKY_RELU);   
  net.addDropout(0.5);
  net.addResidual(7, Activation::RELU, 0);
  net.addConv1D(3,   Activation::LEAKY_RELU, 1, 25);
  net.addResidual(7, Activation::TANH, 3);
  net.addDropout(0.5);
  net.addLinear(2,   Activation::LINEAR);
  net.addSoftmax();

  net.summary();

  // Test data: left-heavy vs right-heavy patterns
  std::vector<std::vector<double>> X = {
      {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0},   // Left  → 1
      {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0},   // Right → 0
      {1.0, 0.8, 0.6, 0.0, 0.0, 0.0, 0.0},   // Left  → 1
      {0.0, 0.0, 0.0, 0.0, 0.6, 0.8, 1.0}    // Right → 0
  };

  std::vector<std::vector<double>> Y = { {1, 0}, {0, 1}, {1, 0}, {0, 1} };

  std::cout << "Training Conv1D...\n";
  for (int epoch = 0; epoch < 2000; ++epoch)
  {
    double loss = 0.0;
    for (size_t i = 0; i < X.size(); ++i)
    {
      loss += net.train(X[i], Y[i], 0.0001);
    }
    if (epoch % 200 == 0)
    {
      std::cout << "Epoch " << epoch << " - Loss: "
        << std::setprecision(8) << loss / 4.0 << "\n";
      net.save("Conv.model");
    }
  }

  std::cout << "\nFinal Predictions:\n";
  for (size_t i = 0; i < X.size(); ++i)
  {
    auto pred = net.predict(X[i]);
    std::cout << "Input " << i << " -> " << std::fixed << std::setprecision(4)
      << pred[0] << " (target = " << Y[i][0] << ")\n";
  }

  return 0;
}

int Conv2DExample()
{
  std::cout << "=== Simple Conv2D Test ===\n\n";

  Network net(42);

  // Input: 28x28 image flattened (like MNIST)
  net.addLinear(784, Activation::LINEAR);

  // First Conv2D: 28x28 → smaller feature maps
  net.addConv2D(28, 28, 3, 3, 1, 1, 8, Activation::RELU, 0.3);

  // Second Conv2D: further feature extraction
  net.addConv2D(26, 26, 3, 3, 1, 1, 16, Activation::RELU, 0.3);

  // Final classification head
  net.addLinear(10, Activation::LINEAR);   // 10 classes
  net.addSoftmax();

  net.summary();

  // Dummy test data (one sample)
  std::vector<double> input(784, 0.0);     // flattened 28x28
  // You can fill input with real data later

  std::vector<double> target = { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };  // class 4

  std::cout << "\nRunning one training step...\n";
  double loss = 0.0;
  for (int i = 0; i < 140; i++)
  {
    loss = net.train(input, target, 0.1);
    std::cout << "Loss = " << loss << std::endl;
  }

  std::cout << "Loss = " << std::fixed << std::setprecision(6) << loss << "\n";

  std::cout << "\nPrediction (probabilities):\n";
  auto pred = net.predict(input);
  for (size_t i = 0; i < pred.size(); ++i)
  {
    std::cout << "Class " << i << ": " << std::fixed << std::setprecision(4)
      << pred[i] << "\n";
  }

  return 0;
}


int main()
{
  //Conv1DExample();
  Conv2DExample();

  return 0;
}