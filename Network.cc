#include "Network.h"

Network::Network(int _INPUT_NODES, int _HIDDEN_NODES, int _OUTPUT_NODES, double _LR_RATE) :
    INPUT_NODES(_INPUT_NODES), HIDDEN_NODES(_HIDDEN_NODES), OUTPUT_NODES(_OUTPUT_NODES), LR_RATE(_LR_RATE)
{

};

bool Network::train(std::vector<double> inputs, std::vector<double> targets)
{
    return false;
};

int Network::query()
{

    return 19;
};
