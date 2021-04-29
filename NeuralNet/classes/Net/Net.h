#pragma once

#include "../Neuron/Neuron.h"

class Neuron;

typedef std::vector<Neuron> Layer;

// number of layers in the net
// number of neurons in each layer

class Net {
public:
	Net(const std::vector<unsigned int>& topology);

	void feedForward(const std::vector<double>& inputVals);
	void backProp(const std::vector<double>& targetVals);
	void getResults(std::vector<double>& resultVals) const;

private:
	std::vector<Layer> m_layers; // m_layers[layerIndex][neuronIndex]
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;
};
