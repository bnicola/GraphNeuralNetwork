#pragma once
#include "Activation.h"
#include "Adam.h"
#include "Connection.h"
#include "Filter.h"
#include <vector>
#include <string>

class Op;

// =============================================================
//  Neuron  — updated with optional Op delegation
//
//  UNCHANGED behaviour when op_ == nullptr:
//    forward()  — standard weighted sum + activation
//    backward() — standard chain rule
//
//  NEW behaviour when op_ != nullptr:
//    forward()  — delegates entirely to op_->forward()
//    backward() — delegates entirely to op_->backward()
//
//  The neuron OWNS the Op and deletes it in the destructor.
//
//  USAGE:
//    neuron->setOp(new GeluOp());       attach custom operation
//    neuron->setOp(nullptr);            back to standard mode
// =============================================================

class Neuron
{
public:
    Neuron(const std::string& name, Activation act);
    ~Neuron();

    void forward();
    void backward();
    void step(double lr, int t);
    void zero_grad();

    void setOp(Op* op);
    Op*  getOp() const { return op_; }

    std::string id;
    Activation  act;
    double      z;
    double      output;
    double      error;
    double      bias;
    double      biasGradient;
    AdamState   biasAdam;

    std::vector<Connection*> inConns;
    std::vector<Connection*> outConns;

private:
    Op* op_ = nullptr;

    void standardForward();
    void standardBackward();
};
