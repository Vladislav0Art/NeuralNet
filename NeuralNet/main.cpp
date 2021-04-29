// neural-net sample
#include <vector>
#include <iostream>

#include "classes/Neuron/Neuron.h"
#include "classes/Net/Net.h"

/*

1) Prepare training data
2) Send each set of inputs to feedForward()
3) For training, send desired outputs to backProp()
4) Get the net's actual results back with getResults()

*/


int main() {
	std::vector<unsigned int> topology = { 3, 2, 1 };
	Net neuralNet(topology);

	std::vector<double> inputVals = { 30, 20, 10 };
	neuralNet.feedForward(inputVals);

	std::vector<double> targetVals;
	neuralNet.backProp(targetVals);

	std::vector<double> resultVals;
	neuralNet.getResults(resultVals);

	for (double r : resultVals) {
		std::cout << r << " ";
	}
	std::cout << std::endl;

	return 0;
}