#ifndef NETWORK_H
#define NETWORK_H

#include "Node.h"

#include <vector>

class Network
{

private:
    const static int THREADS = 4;

    const int INPUT_NODES;
    const std::vector<int> LAYER_NODES;
    const int HIDDEN_LAYERS;
    const int OUTPUT_NODES;
    const double LR_RATE;

    std::vector<std::vector<Node>> layers;

    inline void feed(const std::vector<double> &);
    inline void handleNodeHiddenError(std::vector<double>&, const std::vector<double> &, int, int, int);

public:
    Network(std::vector<int>, double);
    void train(const std::vector<double> &inputs, const std::vector<double> &targets);
    std::vector<double> query(const std::vector<double> &);

};

#endif
