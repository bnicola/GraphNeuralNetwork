#include "Network.h"
//#include "MSELoss.h"
//#include "BCELoss.h"
//#include "CrossEntropyLoss.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include "Network.h"
#include <iostream>
#include <iomanip>

int main()
{
  std::cout << "=== Conv1D Test with Stride ===\n\n";

  Network net(42);

  net.addLinear(7, Activation::LINEAR); // input
  net.addLinear(8, Activation::TANH);   
  net.addConv1D(3, Activation::TANH, 2);
  net.addLinear(2, Activation::LINEAR);
  net.addResidual(8, Activation::TANH, 1);
  net.addLinear(2, Activation::LINEAR);
  net.addLinear(8, Activation::TANH);
  net.addLinear(2, Activation::LINEAR);
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