#pragma once
#include "Adam.h"

class Neuron;
class Filter;   // forward declaration

// =============================================================
//  Connection
//  A directed weighted edge between two neurons.
//
//  NORMAL CONNECTION (dense / skip):
//    filter     = nullptr
//    filterSlot = -1
//    trainable  = true  (dense) or false (skip)
//    weight is owned by this connection, updated by its own Adam.
//
//  CONV CONNECTION:
//    filter     = pointer to the layer's shared Filter
//    filterSlot = which slot in filter->weights[] this edge uses
//    trainable  = true  (weight lives in the filter, not here)
//    weight field is NOT used — forward() reads filter->weight(slot)
//    gradient field is NOT used — backward() routes to filter->accumulateGrad()
//
//  WHY filterSlot:
//    Conv output neuron i connects to input neurons [i, i+1, ..., i+K-1].
//    The connection from input[i+k] to output[i] uses filter weight[k].
//    So filterSlot = k tells us which shared weight to read/write.
//
//  This design means Neuron::forward() and Neuron::backward()
//  just check (filter != nullptr) and branch — one small change,
//  everything else works as before.
// =============================================================

class Connection
{
public:
    Connection();

    Neuron*   from;
    Neuron*   to;
    double    weight;      // used only if filter == nullptr
    double    gradient;    // used only if filter == nullptr
    bool      trainable;   // false = fixed weight (skip), never updated
    Filter*   filter;      // nullptr = normal, non-null = conv
    int       filterSlot;  // which slot in the filter (-1 if not conv)
    AdamState adam;        // used only if filter == nullptr and trainable
};
