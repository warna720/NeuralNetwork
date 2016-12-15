#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

class Network
{

private:
    const int INPUT_NODES;
    const int HIDDEN_NODES;
    const int OUTPUT_NODES;
    const double LR_RATE;

public:
    Network(int, int, int, double);
    bool train(std::vector<double> inputs, std::vector<double> targets);
    int query();

};

#endif
