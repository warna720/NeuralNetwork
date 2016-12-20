# NeuralNetwork
A simple threaded neural network with support for several hidden layers.

Tested on the MNIST database with a best result of 97.02 rate.

No modifications on the training data was done (for example rotating every image 10° or -10°).

The architecture used for the results was 10 epochs, 784 input nodes (obviously), 395 hidden nodes, 10 output nodes and a learning rate of 0.0452.

I got the same result with 15 epochs and 2 hidden layers, 395 nodes for the first layer and 125 nodes for the second layer.


# Compilation
Compile with the flag -pthread to enable threading.

Compile with optimization level 3 to reduce the runtime (alot).

```
g++ -std=c++11 main.cc Network.cc Node.cc -O3 -pthread
```


# Example
```
g++ -std=c++11 main.cc Network.cc Node.cc -O3 -pthread
./a.out
Initiated NL with 784 input nodes, 1 hidden layer(s) [ L1:395 nodes, ], 10 output nodes and a learning rate of 0.0452
Starting training...
Epoch 1 of 10
Epoch 2 of 10
Epoch 3 of 10
Epoch 4 of 10
Epoch 5 of 10
Epoch 6 of 10
Epoch 7 of 10
Epoch 8 of 10
Epoch 9 of 10
Epoch 10 of 10
Training done in 124.624 seconds
Starting testing...
Testing done in 4.116 seconds
Program done in 129.553 seconds
Result: 9702 of 10000 = 97.02%
```
