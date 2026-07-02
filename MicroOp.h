#pragma once
#include <cmath>
#include <vector>

// =============================================================
//  MicroOp
//
//  A single primitive operation inside a Neuron.
//
//  Each MicroOp has:
//    val  — the computed value
//    grad — the gradient accumulated during backward
//    a, b — pointers to input MicroOps (nullptr if unused)
//    type — which operation this node performs
//
//  The graph is just pointers between MicroOps.
//  The vector in Neuron stores them in forward order.
//  Backward walks the vector in reverse.
//
//  TYPES:
//    INPUT — holds an input value (x from inConn->from->output)
//    PARAM — holds a learnable value (weight or bias)
//    ADD   — val = a->val + b->val
//    MUL   — val = a->val * b->val
//    TANH  — val = tanh(a->val)
//    RELU  — val = max(0, a->val)
//    EXP   — val = exp(a->val)
//    NEG   — val = -(a->val)
//    SQ    — val = a->val * a->val
// =============================================================

struct MicroOp
{
    enum Type { INPUT, PARAM, ADD, MUL, TANH, RELU, EXP, NEG, SQ };

    double   val  = 0.0;
    double   grad = 0.0;
    MicroOp* a    = nullptr;
    MicroOp* b    = nullptr;
    Type     type;

    MicroOp(Type t, MicroOp* a = nullptr, MicroOp* b = nullptr)
        : type(t), a(a), b(b) {}

    // forward: compute val from inputs
    void forward()
    {
        switch (type)
        {
            case INPUT: break;
            case PARAM: break;
            case ADD:  val = a->val + b->val;                        break;
            case MUL:  val = a->val * b->val;                        break;
            case TANH: val = std::tanh(a->val);                      break;
            case RELU: val = (a->val > 0.0) ? a->val : 0.0;          break;
            case EXP:  val = std::exp(a->val);                       break;
            case NEG:  val = -(a->val);                              break;
            case SQ:   val = a->val * a->val;                        break;
        }
    }

    // backward: distribute grad to input nodes
    void backward()
    {
        switch (type)
        {
            case INPUT: break;
            case PARAM: break;
            case ADD:
                a->grad += grad;
                b->grad += grad;
                break;
            case MUL:
                a->grad += b->val * grad;
                b->grad += a->val * grad;
                break;
            case TANH:
                a->grad += (1.0 - val * val) * grad;
                break;
            case RELU:
                a->grad += (a->val > 0.0 ? 1.0 : 0.0) * grad;
                break;
            case EXP:
                a->grad += val * grad;
                break;
            case NEG:
                a->grad += -grad;
                break;
            case SQ:
                a->grad += 2.0 * a->val * grad;
                break;
        }
    }
};
