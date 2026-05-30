#include "Adam.h"


AdamState::AdamState()
  : m_(0.0), v_(0.0)
{
}

double AdamState::update(double& param, double gradient, double lr, int t)
{
  // Step 1 — update smoothed gradient
  m_ = BETA1 * m_ + (1.0 - BETA1) * gradient;
  
  // Step 2 — update smoothed gradient²
   v_ = BETA2 * v_ + (1.0 - BETA2) * gradient * gradient;
   
   // Step 3 — bias correction (fixes cold start from zero)
   double m_hat = m_ / (1.0 - std::pow(BETA1, t));
   double v_hat = v_ / (1.0 - std::pow(BETA2, t));

   // Step 4 — apply update
   double step = lr * m_hat / (std::sqrt(v_hat) + EPSILON);
   param -= step;
   return step;
}
