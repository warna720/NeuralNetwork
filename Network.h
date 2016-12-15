#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

class Network
{

private:
    static const int INPUT_NODES = 784;
    static const int HIDDEN_NODES = 200;
    static const int OUTPUT_NODES = 10;
    static constexpr double LR_RATE = 0.1;

public:
    Network();
    bool train(std::vector<int> inputs, std::vector<int> targets);
    int query();
};

#endif
