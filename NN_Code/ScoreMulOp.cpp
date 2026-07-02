#include "ScoreMulOp.h"
#include "Neuron.h"
#include "Connection.h"
#include <cmath>

void ScoreMulOp::customForward(Neuron* n,
                                const std::vector<Connection*>& inConns)
{
    // inConns layout:
    //   [0  .. d-1]  = Q connections
    //   [d  .. 2d-1] = K connections

    double dot = 0.0;

    for (int i = 0; i < d; i++)
    {
        double q = inConns[i]->from->output;       // Q[i]
        double k = inConns[i + d]->from->output;   // K[i]
        dot += q * k;
    }

    n->output = dot / std::sqrt((double)d);
}
