#include "Network.h"

#include <random>
#include <thread>

Network::Network(std::vector<int> _LAYER_NODES, double _LR_RATE) :
    INPUT_NODES(_LAYER_NODES.at(0)), LAYER_NODES(_LAYER_NODES), HIDDEN_LAYERS(_LAYER_NODES.size()-2), OUTPUT_NODES(_LAYER_NODES.at(_LAYER_NODES.size()-1)), LR_RATE(_LR_RATE)
{
    std::vector<Node> layerNodes;
    std::vector<double> weights;

    std::random_device rd;
    std::mt19937 gen(rd());

    for(int layer = 1; layer < LAYER_NODES.size(); ++layer)
    {
        int nodes = LAYER_NODES.at(layer);
        std::normal_distribution<> dist(0.0, pow(nodes, -0.5));

        for (int node = 0; node < nodes; ++node)
        {
            for (int previousNodes = 0; previousNodes < LAYER_NODES.at(std::max(layer-1, 0)); ++previousNodes)
                weights.push_back(dist(gen));

            layerNodes.push_back(Node(weights, LR_RATE));
            weights.clear();
        }

        layers.push_back(layerNodes);
        layerNodes.clear();
    }
};

inline void Network::feed(const std::vector<double> &inputs)
{
    std::vector<double> layerInputs = inputs;
    std::vector<double> layerOutputs;
    for (int layer = 0; layer < layers.size(); ++layer)
    {
        for (int node = 0; node < layers.at(layer).size(); ++node)
        {
            layers.at(layer).at(node).feed(layerInputs);
            layerOutputs.push_back(layers.at(layer).at(node).getValue());
        }
        layerInputs = layerOutputs;
        layerOutputs.clear();
    }
}

inline void Network::handleNodeHiddenError(std::vector<double> &hidden_errors, const std::vector<double> &previous_errors, int layer, int pos, int stop)
{
    double hidden_error = 0;

    for (; pos < stop; ++pos)
    {
        for (int nodeAhead = 0; nodeAhead < layers.at(layer+1).size(); ++nodeAhead)
        {
            hidden_error += layers.at(layer+1).at(nodeAhead).getWeight(pos) * previous_errors.at(nodeAhead);
        }

        hidden_errors.at(pos) = hidden_error;
        layers.at(layer).at(pos).updateWeight(hidden_error);
        hidden_error = 0;
    }
}

void Network::train(const std::vector<double> &inputs, const std::vector<double> &targets)
{
    if (inputs.size() != INPUT_NODES || targets.size() != OUTPUT_NODES)
    {
        return;
    }

    feed(inputs);


    // update weights
    std::vector<double> output_errors;
    for (int i = 0; i < layers.at(layers.size()-1).size(); ++i)
    {
        double output_error = targets.at(i) - layers.at(layers.size()-1).at(i).getValue();
        layers.at(layers.size()-1).at(i).updateWeight(output_error);
        output_errors.push_back(output_error);
    }


    std::vector<double> previous_errors = output_errors;
    double hidden_error = 0;

    for (int layer = layers.size()-2; layer >= 0; --layer)
    {
        std::vector<double> hidden_errors (layers.at(layer).size());

        if (THREADS > 1 && layers.at(layer).size() > THREADS*50)
        {
            std::vector<std::thread> active_threads;

            int node_per_thread = layers.at(layer).size() / THREADS;
            for (int i = 0; i < THREADS-1; ++i)
            {
                active_threads.push_back(std::thread(&Network::handleNodeHiddenError,
                                                    this,
                                                    std::ref(hidden_errors),
                                                    std::ref(previous_errors),
                                                    layer, i*node_per_thread,
                                                    (i+1)*node_per_thread
                                                    ));
            }
            active_threads.push_back(std::thread(&Network::handleNodeHiddenError,
                                                  this, std::ref(hidden_errors),
                                                  std::ref(previous_errors),
                                                  layer,
                                                  (THREADS-1)*node_per_thread,
                                                  layers.at(layer).size()
                                                  ));

            for (auto &t : active_threads)
                t.join();
        }
        else
        {
            handleNodeHiddenError(std::ref(hidden_errors), std::ref(previous_errors), layer, 0, layers.at(layer).size());
        }

        previous_errors = hidden_errors;
        hidden_errors.clear();
    }

};

std::vector<double> Network::query(const std::vector<double> &inputs)
{
    feed(inputs);

    std::vector<double> result;
    for (int node = 0; node < layers.at(layers.size()-1).size(); ++node)
    {
        result.push_back(layers.at(layers.size()-1).at(node).getValue());
    }

    return result;
};

