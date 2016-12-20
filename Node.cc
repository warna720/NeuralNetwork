#include "Node.h"

#include <random>

Node::Node(std::vector<double> _weights, double _learning_rate) : weights(_weights), learning_rate(_learning_rate) {};

void Node::feed(std::vector<double> inputs)
{
    last_inputs = inputs;
    value = 0;

    for(int i = 0; i < inputs.size(); ++i)
    {
        value += inputs.at(i) * weights.at(i);
    }

    value = sigmoid(value);
}


void Node::updateWeight(double o_error)
{
    for (int i = 0; i < last_inputs.size(); ++i)
    {
        weights.at(i) += 
                        learning_rate *
                        o_error *
                        value *
                        (1.0 - value) *
                        last_inputs.at(i);
    }
};

