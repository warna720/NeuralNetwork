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
    std::vector<std::vector<int>> trainingData;
    
    std::string line;
    while(trainingFile >> line)
    {
        // replace ',' for stringstream
        std::replace_if(line.begin(), line.end(), [](char x){ return x == ','; }, ' ');

        std::stringstream ss(line);
        std::vector<int> rowData;
        for(int i; ss >> i; )
            rowData.push_back(i);
        trainingData.push_back(rowData);
    }

    trainingFile.close();


    // train the neural network
    Network network;

    for (int epochs = 5; epochs > 0; --epochs)
    {
        
        std::vector<int> inputs;
        std::vector<int> targets;
        network.train(inputs, targets);
    }


    return 0;
}
