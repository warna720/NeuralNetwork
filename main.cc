#include "Network.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <vector>
#include <algorithm>


int main()
{
    // load the mnist data
    std::ifstream trainingFile("data/mnist_train_100.csv", std::ifstream::in);
    std::vector<std::vector<double>> trainingData;
    
    std::string line;
    while(trainingFile >> line)
    {
        // replace ',' for stringstream
        std::replace_if(begin(line), end(line), [](char x){ return x == ','; }, ' ');

        std::stringstream ss(line);
        std::vector<double> rowData;
        for(double i; ss >> i; )
            rowData.push_back(i);
        
        trainingData.push_back(rowData);
    }
    trainingFile.close();

    // train the neural network
    const int INPUT_NODES = 784;
    const int HIDDEN_NODES = 200;
    const int OUTPUT_NODES = 10;
    const double LR_RATE = 0.1;
    Network network(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LR_RATE);
    for (int epochs = 5; epochs > 0; --epochs)
    {
        for (auto &rowData: trainingData)
        {
            // prepare the data for the network
            std::transform(next(begin(rowData)), end(rowData), next(begin(rowData)), [](int n){ return n / 255.0 * 0.99 + 0.01; });
            std::vector<double> targets(OUTPUT_NODES, 0.01);
            targets.at(rowData.at(0)) = 0.99;

            //network.train(new std::vector<rowData, targets);
        }
    }

    

    // TODO
    // save training to file
    // load training from file
    // query
    // compare with test

    return 0;
}
