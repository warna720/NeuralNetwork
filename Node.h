#ifndef NODE_H
#define NODE_H

#include <vector>
#include <math.h>

class Node
{

private:
    std::vector<double> weights;
    std::vector<double> last_inputs;
    double value;
    double learning_rate;

    double sigmoid(double x) { return 1 / (1 + exp(-x)); };

public:
    Node(std::vector<double>, double);

    void feed(std::vector<double>);
    double getValue() { return value; };
    void updateWeight(double);
    std::vector<double> getWeights() { return weights; };
    double getWeight(int i) { return weights.at(i); };

};

#endif
