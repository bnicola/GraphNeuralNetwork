#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

#include "Neuron.h"
#include "Connection.h"
#include "Filter.h"
#include "Activation.h"
#include "StandardOp.h"
#include "ReluOp.h"
#include "CustomSquareOp.h"
#include "ScoreMulOp.h"

// =============================================================
//  Helpers
// =============================================================

// Wire one neuron to another with a trainable connection.
// Returns the Connection so caller can read its gradient.
Connection* wire(Neuron* from, Neuron* to, double weight)
{
    Connection* c = new Connection();
    c->from      = from;
    c->to        = to;
    c->weight    = weight;
    c->trainable = true;
    c->filter    = nullptr;
    c->filterSlot= 0;
    c->gradient  = 0.0;

    from->outConns.push_back(c);
    to->inConns.push_back(c);
    return c;
}

// Wire a pass-through connection (no weight, non-trainable).
// Used when the Op reads inputs directly (e.g. ScoreMulOp).
Connection* wirePassThrough(Neuron* from, Neuron* to)
{
    Connection* c = new Connection();
    c->from      = from;
    c->to        = to;
    c->weight    = 1.0;
    c->trainable = false;
    c->filter    = nullptr;
    c->filterSlot= 0;
    c->gradient  = 0.0;

    from->outConns.push_back(c);
    to->inConns.push_back(c);
    return c;
}

void printSeparator(const char* title)
{
    printf("\n");
    printf("=============================================================\n");
    printf("  %s\n", title);
    printf("=============================================================\n");
}

// =============================================================
//  Demo 1 — StandardOp
//
//  Two input neurons feed a hidden neuron using StandardOp.
//  Shows the standard weighted sum + activation as an explicit Op.
//
//  Graph:
//    x0 --w0--> h (StandardOp, Tanh)
//    x1 --w1--> h
// =============================================================

void demo_standard_op()
{
    printSeparator("Demo 1: StandardOp — standard weighted sum");

    Neuron x0("x0", Activation::LINEAR);
    Neuron x1("x1", Activation::LINEAR);
    Neuron h ("h",  Activation::TANH);

    x0.output = 0.5;
    x1.output = 0.3;

    Connection* c0 = wire(&x0, &h, 0.4);
    Connection* c1 = wire(&x1, &h, 0.7);

    h.setOp(new StandardOp());

    // forward
    h.forward();

    double expected_z      = 0.4 * 0.5 + 0.7 * 0.3;               // 0.41
    double expected_output = std::tanh(expected_z);

    printf("z        = %.6f  (expected %.6f)\n", h.z,      expected_z);
    printf("output   = %.6f  (expected %.6f)\n", h.output, expected_output);

    // backward — seed error at output
    h.error = 1.0;
    h.backward();

    double dtanh = 1.0 - expected_output * expected_output;
    printf("c0.grad  = %.6f  (expected %.6f)\n", c0->gradient, dtanh * x0.output);
    printf("c1.grad  = %.6f  (expected %.6f)\n", c1->gradient, dtanh * x1.output);

    delete c0;
    delete c1;
}

// =============================================================
//  Demo 2 — ReluOp
//
//  Same graph but with ReLU.
//  Shows how swapping the Op changes neuron behaviour
//  without touching wiring or any other code.
//
//  Graph:
//    x0 --w0--> h (ReluOp)
//    x1 --w1--> h
// =============================================================

void demo_relu_op()
{
    printSeparator("Demo 2: ReluOp — swap activation without rewiring");

    Neuron x0("x0", Activation::LINEAR);
    Neuron x1("x1", Activation::LINEAR);
    Neuron h ("h",  Activation::LINEAR);   // activation ignored — ReluOp overrides

    x0.output = 0.5;
    x1.output = 0.3;

    Connection* c0 = wire(&x0, &h, 0.4);
    Connection* c1 = wire(&x1, &h, 0.7);

    h.setOp(new ReluOp());

    h.forward();

    double z      = 0.4 * 0.5 + 0.7 * 0.3;   // 0.41 — positive so relu passes it through
    printf("z        = %.6f\n", h.z);
    printf("output   = %.6f  (expected %.6f — relu of z)\n", h.output, z);

    h.error = 1.0;
    h.backward();

    printf("c0.grad  = %.6f  (expected %.6f)\n", c0->gradient, 1.0 * x0.output);  // delta=1 since z>0
    printf("c1.grad  = %.6f  (expected %.6f)\n", c1->gradient, 1.0 * x1.output);

    delete c0;
    delete c1;
}

// =============================================================
//  Demo 3 — CustomSquareOp + gradient check
//
//  A neuron that computes output = (w0*x0 + w1*x1)^2
//  Backward is computed by NumericalGradOp nudging.
//
//  Then we compute the analytic gradient by hand:
//    d(z^2)/dw0 = 2 * z * x0
//    d(z^2)/dw1 = 2 * z * x1
//
//  Both should match to within 1e-6.
//
//  Graph:
//    x0 --w0--> sq (CustomSquareOp)
//    x1 --w1--> sq
// =============================================================

void demo_square_op()
{
    printSeparator("Demo 3: CustomSquareOp — arbitrary op with free numerical backward");

    Neuron x0("x0", Activation::LINEAR);
    Neuron x1("x1", Activation::LINEAR);
    Neuron sq("sq", Activation::LINEAR);

    x0.output = 0.5;
    x1.output = 0.3;

    Connection* c0 = wire(&x0, &sq, 0.4);
    Connection* c1 = wire(&x1, &sq, 0.7);

    sq.setOp(new CustomSquareOp());

    sq.forward();

    double z        = 0.4 * 0.5 + 0.7 * 0.3;   // 0.41
    double expected = z * z;                       // 0.1681

    printf("output   = %.6f  (expected %.6f)\n", sq.output, expected);

    // seed error
    sq.error = 1.0;
    sq.backward();

    // analytic gradients: d(z^2)/dw = 2*z*x
    double analytic_c0 = 2.0 * z * x0.output;
    double analytic_c1 = 2.0 * z * x1.output;

    printf("\nGradient check:\n");
    printf("c0 numerical = %.8f   analytic = %.8f   diff = %.2e\n",
           c0->gradient, analytic_c0,
           std::fabs(c0->gradient - analytic_c0));
    printf("c1 numerical = %.8f   analytic = %.8f   diff = %.2e\n",
           c1->gradient, analytic_c1,
           std::fabs(c1->gradient - analytic_c1));

    bool pass = std::fabs(c0->gradient - analytic_c0) < 1e-6 &&
                std::fabs(c1->gradient - analytic_c1) < 1e-6;
    printf("Result: %s\n", pass ? "PASS" : "FAIL");

    delete c0;
    delete c1;
}

// =============================================================
//  Demo 4 — ScoreMulOp
//
//  This is the attention score computation:
//    score = dot(Q, K) / sqrt(d)
//
//  We manually create Q and K neurons (representing projections
//  from a previous layer), wire them into a score neuron,
//  and let ScoreMulOp compute the dot product.
//
//  Then backward sends gradients back into Q and K neurons
//  automatically via numerical nudging.
//
//  Graph (d=3):
//    Q0 --pass--> score (ScoreMulOp)
//    Q1 --pass-->
//    Q2 --pass-->
//    K0 --pass-->
//    K1 --pass-->
//    K2 --pass-->
//
//  The score neuron reads Q and K outputs directly.
//  No weights on these connections — the projections
//  were already learned in the previous layer.
// =============================================================

void demo_score_op()
{
    printSeparator("Demo 4: ScoreMulOp — attention dot-product score neuron");

    const int d = 3;

    // Q neurons — represent one query vector
    Neuron q0("q0", Activation::LINEAR);
    Neuron q1("q1", Activation::LINEAR);
    Neuron q2("q2", Activation::LINEAR);

    // K neurons — represent one key vector
    Neuron k0("k0", Activation::LINEAR);
    Neuron k1("k1", Activation::LINEAR);
    Neuron k2("k2", Activation::LINEAR);

    // Set Q and K values directly (as if from a projection layer)
    q0.output = 0.6;   q1.output = 0.1;   q2.output = 0.8;
    k0.output = 0.3;   k1.output = 0.9;   k2.output = 0.2;

    // Score neuron
    Neuron score("score", Activation::LINEAR);
    score.setOp(new ScoreMulOp(d));

    // Wire: first Q, then K — ScoreMulOp expects this layout
    wirePassThrough(&q0, &score);
    wirePassThrough(&q1, &score);
    wirePassThrough(&q2, &score);
    wirePassThrough(&k0, &score);
    wirePassThrough(&k1, &score);
    wirePassThrough(&k2, &score);

    // forward
    score.forward();

    double dot      = q0.output*k0.output
                    + q1.output*k1.output
                    + q2.output*k2.output;
    double expected = dot / std::sqrt((double)d);

    printf("dot(Q,K)    = %.6f\n", dot);
    printf("score       = %.6f  (expected %.6f)\n", score.output, expected);

    // backward — seed error at score
    score.error = 1.0;
    score.backward();

    // analytic gradients for inputs:
    //   d(score)/dQ[i] = K[i] / sqrt(d)
    //   d(score)/dK[i] = Q[i] / sqrt(d)
    double sq_d = std::sqrt((double)d);

    printf("\nGradient check (analytic vs numerical):\n");

    double q_inputs[3] = {q0.output, q1.output, q2.output};
    double k_inputs[3] = {k0.output, k1.output, k2.output};
    Neuron* q_neurons[3] = {&q0, &q1, &q2};
    Neuron* k_neurons[3] = {&k0, &k1, &k2};

    bool pass = true;

    for (int i = 0; i < d; i++)
    {
        double analytic_q = k_inputs[i] / sq_d;
        double analytic_k = q_inputs[i] / sq_d;
        double numerical_q = q_neurons[i]->error;
        double numerical_k = k_neurons[i]->error;

        printf("Q[%d]: numerical=%.8f  analytic=%.8f  diff=%.2e\n",
               i, numerical_q, analytic_q,
               std::fabs(numerical_q - analytic_q));

        printf("K[%d]: numerical=%.8f  analytic=%.8f  diff=%.2e\n",
               i, numerical_k, analytic_k,
               std::fabs(numerical_k - analytic_k));

        if (std::fabs(numerical_q - analytic_q) > 1e-6) pass = false;
        if (std::fabs(numerical_k - analytic_k) > 1e-6) pass = false;
    }

    printf("Result: %s\n", pass ? "PASS" : "FAIL");

    // clean up connections
    for (auto* c : score.inConns) delete c;
}

// =============================================================
//  Demo 5 — Chain of mixed operations
//
//  Shows that arbitrary Ops chain together correctly.
//  Gradient flows through the chain automatically.
//
//  Graph:
//    x0 --w0--> sq (CustomSquareOp) --pass--> relu (ReluOp) --w--> out (StandardOp)
//
//  Forward:
//    sq.output   = (w0 * x0)^2
//    relu.output = relu(sq.output)        always positive here so passes through
//    out.output  = tanh(w * relu.output)
//
//  Backward:
//    Each Op computes its own gradient.
//    Error flows back through the chain automatically.
// =============================================================

void demo_chain()
{
    printSeparator("Demo 5: Chain of mixed Ops — gradient flows through all");

    Neuron x0  ("x0",   Activation::LINEAR);
    Neuron sq  ("sq",   Activation::LINEAR);
    Neuron relu("relu", Activation::LINEAR);
    Neuron out ("out",  Activation::TANH);

    x0.output = 0.8;

    Connection* c_sq   = wire        (&x0,   &sq,   0.5);   // trainable
    Connection* c_relu = wirePassThrough(&sq,   &relu);       // pass-through
    Connection* c_out  = wire        (&relu, &out,  0.9);    // trainable

    sq.setOp  (new CustomSquareOp());
    relu.setOp(new ReluOp());
    out.setOp (new StandardOp());

    // forward through chain
    sq.forward();
    relu.forward();
    out.forward();

    printf("x0.output   = %.6f\n", x0.output);
    printf("sq.output   = %.6f  (expected %.6f = (0.5*0.8)^2)\n",
           sq.output, std::pow(0.5 * 0.8, 2.0));
    printf("relu.output = %.6f  (expected same — positive input)\n", relu.output);
    printf("out.output  = %.6f  (expected %.6f = tanh(0.9 * relu))\n",
           out.output, std::tanh(0.9 * relu.output));

    // backward — seed error at output
    out.error = 1.0;

    // backward through chain in reverse order
    out.backward();
    relu.backward();
    sq.backward();

    printf("\nc_sq.grad   = %.6f  (gradient of (w*x)^2 w.r.t. w)\n", c_sq->gradient);
    printf("c_out.grad  = %.6f  (gradient of tanh(w*relu) w.r.t. w)\n", c_out->gradient);
    printf("x0.error    = %.6f  (error propagated all the way back to input)\n", x0.error);

    delete c_sq;
    delete c_relu;
    delete c_out;
}

// =============================================================
//  main
// =============================================================

int main()
{
    demo_standard_op();
    demo_relu_op();
    demo_square_op();
    demo_score_op();
    demo_chain();

    printf("\nAll demos complete.\n");
    return 0;
}
