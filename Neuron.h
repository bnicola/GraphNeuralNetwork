#pragma once
#include "Activation.h"
#include "Adam.h"
#include "Connection.h"
#include "Filter.h"   // needed for filter->weight() and accumulateGrad()
#include <vector>
#include <string>
#include <unordered_map>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>


// =============================================================
//  Neuron
//
//  Unchanged from before except two small branches in
//  forward() and backward() for conv connections.
//
//  forward():
//    For each inConn, reads weight as:
//      normal:  c->weight
//      conv:    c->filter->weight(c->filterSlot)
//    Everything else identical.
//
//  backward():
//    For each inConn, routes gradient as:
//      normal trainable:  c->gradient += delta * x
//      conv:              c->filter->accumulateGrad(slot, delta*x)
//      skip (trainable=false, no filter): c->gradient += error * x
//                                         (accumulated, never applied)
//    Everything else identical.
//
//  step():
//    Only updates connections where filter==nullptr and trainable==true.
//    Conv connections are updated by Conv1DLayer::step() via filter->step().
//    Skip connections (trainable=false) are never updated.
// =============================================================

class Neuron
{
public:
    Neuron(const std::string& name, Activation act);

    void forward();
    void backward();
    void step(double lr, int t);
    void zero_grad();

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
};
