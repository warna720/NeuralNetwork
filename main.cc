#include "Network.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <vector>
#include <algorithm>

#include <chrono>


int main()
{
    // load the training data
    std::ifstream trainingFile("data/mnist_train.csv", std::ifstream::in);
    std::vector<std::vector<double>> trainingData;

    // prepare the data
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

    std::vector<int> layerNodes {784, 350, 125, 10};
    int INPUT_NODES = layerNodes.at(0);
    int HIDDEN_LAYERS = layerNodes.size()-2;
    int OUTPUT_NODES = layerNodes.at(layerNodes.size()-1);
    double LR_RATE = 0.0452;
    
    Network network(layerNodes, LR_RATE);

    std::cout << "Initiated NL with " <<
                INPUT_NODES << " input nodes, " << 
                HIDDEN_LAYERS << " hidden layer(s) [ ";
                for (int i = 1; i < layerNodes.size()-1; ++i)
                {
                    std::cout << "L" << i << ":" << layerNodes.at(i) << " nodes, ";
                }
                std::cout << "], " <<

                OUTPUT_NODES << " output nodes and a learning rate of " <<
                LR_RATE <<
                std::endl;

    std::cout << "Starting training..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();

    int epochs = 10;
    for (int epoch = 1; epoch < epochs+1; ++epoch)
    {
        std::cout << "Epoch " << epoch << " of " << epochs << std::endl;
        for (std::vector<double> rowData: trainingData)
        {
            // prepare the data for the network
            std::transform(next(begin(rowData)), end(rowData), next(begin(rowData)), [](int n){ return n / 255.0 * 0.99 + 0.01; });
            std::vector<double> targets(OUTPUT_NODES, 0.01);
            targets.at(rowData.at(0)) = 0.99;

            rowData.erase(begin(rowData));
            network.train(rowData, targets);
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Training done in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()/1000.0 << " seconds" << std::endl;


    // load the test data
    std::ifstream testFile("data/mnist_test.csv", std::ifstream::in);
    std::vector<std::vector<double>> testData;
    
    while(testFile >> line)
    {
        std::replace_if(begin(line), end(line), [](char x){ return x == ','; }, ' ');

        std::stringstream ss(line);
        std::vector<double> rowData;
        for(double i; ss >> i; )
            rowData.push_back(i);
        
        testData.push_back(rowData);
    }
    testFile.close();

    std::cout << "Starting testing..." << std::endl;
    auto t3 = std::chrono::high_resolution_clock::now();

    std::vector<int> results;
    for (auto &rowData: testData)
    {
        std::transform(next(begin(rowData)), end(rowData), next(begin(rowData)), [](int n){ return n / 255.0 * 0.99 + 0.01; });

        int target = rowData.at(0);
        rowData.erase(begin(rowData));
        std::vector<double> result = network.query(rowData);

        auto biggest = std::max_element(begin(result), end(result));
        auto answer = std::distance(begin(result), biggest);

        if (target == answer)
        {
            results.push_back(1);
        }
        else
        {
            results.push_back(0);
        }
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "Testing done in " << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count()/1000.0 << " seconds" << std::endl;
    std::cout << "Program done in " << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t1).count()/1000.0 << " seconds" << std::endl;

    int amountCorrect = std::accumulate(begin(results), end(results), 0);
    double percentage = amountCorrect/((double)results.size())*100;
    std::cout << "Result: " << amountCorrect << " of " << results.size() << " = " << percentage << "%" << std::endl;

    return 0;
}
