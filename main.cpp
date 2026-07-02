#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include "Network.h"
#include "Activation.h"

// =============================================================
//  Image classifier for 64x64 RGB images
//
//  Larger than 32x32 — tests framework with real-ish scale.
//  Still manageable on CPU in reasonable time.
//
//  If you want 128x128 — change IMG_H, IMG_W and reduce
//  filter counts to 4, 8, 16 to keep memory under control.
//
//  If you want 224x224 — reduce to 1 or 2 conv blocks and
//  expect very slow training. Good for correctness testing
//  but not for actual experiments.
//
//  DUMMY DATA:
//    6 classes with distinct colour and brightness patterns.
//    Each 64x64x3 image has noise added so no two are identical.
//    A working network should reach 80%+ accuracy within 20 epochs.
// =============================================================

const int IMG_H = 96;
const int IMG_W = 96;
const int IMG_C = 3;
const int NUM_CLASSES = 6;
const int NUM_SAMPLES = 300;
const int NUM_EPOCHS = 3;
const double LR = 0.00005;

// =============================================================
//  generateImage
//
//  6 visually distinct patterns:
//    0 — all dark
//    1 — all bright
//    2 — red dominant
//    3 — green dominant
//    4 — blue dominant
//    5 — checkerboard (alternating dark/bright blocks)
// =============================================================

std::vector<double> generateImage(int classIdx, std::mt19937& rng)
{
  int total = IMG_H * IMG_W * IMG_C;
  std::vector<double> img(total, 0.0);

  std::uniform_real_distribution<double> noise(-0.05, 0.05);

  for (int h = 0; h < IMG_H; h++)
    for (int w = 0; w < IMG_W; w++)
    {
      int base = (h * IMG_W + w) * IMG_C;

      double r = 0.5, g = 0.5, b = 0.5;

      switch (classIdx)
      {
      case 0:
        // dark
        r = 0.05; g = 0.05; b = 0.05;
        break;

      case 1:
        // bright
        r = 0.95; g = 0.95; b = 0.95;
        break;

      case 2:
        // red
        r = 0.90; g = 0.05; b = 0.05;
        break;

      case 3:
        // green
        r = 0.05; g = 0.90; b = 0.05;
        break;

      case 4:
        // blue
        r = 0.05; g = 0.05; b = 0.90;
        break;

      case 5:
        // checkerboard — 8x8 blocks
      {
        int blockH = h / 8;
        int blockW = w / 8;
        double val = ((blockH + blockW) % 2 == 0) ? 0.9 : 0.1;
        r = val; g = val; b = val;
      }
      break;
      }

      img[base + 0] = std::max(0.0, std::min(1.0, r + noise(rng)));
      img[base + 1] = std::max(0.0, std::min(1.0, g + noise(rng)));
      img[base + 2] = std::max(0.0, std::min(1.0, b + noise(rng)));
    }

  return img;
}

std::vector<double> makeTarget(int classIdx)
{
  std::vector<double> t(NUM_CLASSES, 0.0);
  t[classIdx] = 1.0;
  return t;
}

// =============================================================
//  buildNetwork
//
//  64x64x3 input
//
//  Block 1: Conv 3x3x8   -> 62x62x8
//           MaxPool 2x2  -> 31x31x8
//
//  Block 2: Conv 3x3x16  -> 29x29x16
//           MaxPool 2x2  -> 14x14x16
//
//  Block 3: Conv 3x3x32  -> 12x12x32
//           MaxPool 2x2  -> 6x6x32 = 1152
//
//  Dense:   1152 -> 128 -> NUM_CLASSES
// =============================================================

Network buildNetwork()
{
  Network net(42);

  // input layer
  net.addLinear(IMG_H, IMG_W, IMG_C, Activation::LINEAR);

  // block 1
  // 64x64x3 -> 62x62x8
  net.addConv2D(5, 5, 1, 1, 8, Activation::RELU);
  net.addMaxPool2D(4, 4, 4, 4);

  // block 2
  // 31x31x8 -> 29x29x16
  net.addConv2D(3, 3, 1, 1, 16, Activation::RELU);
  // 29x29x16 -> 14x14x16
  net.addMaxPool2D(2, 2, 2, 2);

  // block 3
  // 14x14x16 -> 12x12x32
  net.addConv2D(3, 3, 1, 1, 32, Activation::RELU);
  // 12x12x32 -> 6x6x32 = 1152
  net.addMaxPool2D(2, 2, 2, 2);
 
  // dense head
  net.addLinear(128, Activation::RELU);
  net.addResidual(0, Activation::LEAKY_RELU);
  net.addDropout(0.3);

  net.addLinear(NUM_CLASSES, Activation::LINEAR);
  net.addSoftmax();

  return net;
}

// =============================================================
//  main
// =============================================================

int main()
{
  std::mt19937 rng(234);

  // generate dataset
  printf("Generating %d images at %dx%d...\n", NUM_SAMPLES, IMG_H, IMG_W);

  std::vector<std::vector<double>> images(NUM_SAMPLES);
  std::vector<int>                 labels(NUM_SAMPLES);

  for (int i = 0; i < NUM_SAMPLES; i++)
  {
    labels[i] = i % NUM_CLASSES;
    images[i] = generateImage(labels[i], rng);
  }

  printf("Each image: %d values\n", IMG_H * IMG_W * IMG_C);
  printf("Dataset:    %.1f MB\n\n",
    (double)NUM_SAMPLES * IMG_H * IMG_W * IMG_C * 8 / 1e6);

  //build network
  Network net = buildNetwork();
  bool loaded = net.load("large_classifier.bin");
  if (!loaded)
  {    
    printf("Network built.\n");
    printf("Training on %d samples for %d epochs...\n\n", NUM_SAMPLES, NUM_EPOCHS);

    // training loop
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++)
    {
      double totalLoss = 0.0;
      int    correct = 0;

      net.SetTraining(true);

      // shuffle
      std::vector<int> order(NUM_SAMPLES);
      for (int i = 0; i < NUM_SAMPLES; i++) order[i] = i;
      std::shuffle(order.begin(), order.end(), rng);

      for (int s : order)
      {
        std::vector<double> target = makeTarget(labels[s]);

        net.zero_grad();

        std::vector<double> output = net.forward(images[s], true);

        double loss = net.backward(target, 1);
        totalLoss += loss;

        std::cout << "loss = " << loss << std::endl;

        net.step(LR);

        // prediction
        int pred = 0;
        for (int c = 1; c < NUM_CLASSES; c++)
          if (output[c] > output[pred]) pred = c;
        if (pred == labels[s]) correct++;
      }

      printf("Epoch %2d  loss=%.4f  acc=%5.1f%%\n",
        epoch + 1,
        totalLoss / NUM_SAMPLES,
        100.0 * correct / NUM_SAMPLES);
    }

    // evaluation
    printf("\nFinal evaluation (no dropout):\n");
  }
  net.SetTraining(false);

  int correct = 0;
  for (int i = 0; i < NUM_SAMPLES; i++)
  {
    std::vector<double> output = net.forward(images[i], false);
    int pred = 0;
    for (int c = 1; c < NUM_CLASSES; c++)
      if (output[c] > output[pred]) pred = c;
    if (pred == labels[i]) correct++;
  }

  printf("Accuracy:        %.1f%%\n", 100.0 * correct / NUM_SAMPLES);
  printf("Random baseline: %.1f%%\n", 100.0 / NUM_CLASSES);

  net.save("large_classifier.bin");
  printf("\nSaved to large_classifier.bin\n");

  return 0;
}