#include <vector>
#include <cassert>
#include "./Net.h"

// Layer == vector<Neuron>

Net::Net(const std::vector<unsigned int>& topology) {
	unsigned int numOfLayers = topology.size();

	for (unsigned int layerInd = 0; layerInd < numOfLayers; layerInd++) {
		m_layers.push_back(Layer());
		// number of outputs for neurons in current layer (equals to number of neurons of next layer)
		unsigned int numOfOutputs = (layerInd == numOfLayers - 1) ? 0 : topology[layerInd + 1];

		// We have made a new layer, now fill it its neurons, and
		// add a bias neuron to the layer
		for(unsigned int neuronInd = 0; neuronInd <= topology[layerInd]; neuronInd++) {
			m_layers.back().push_back(Neuron(numOfOutputs, neuronInd));
		}

		// force the bias node's output value to 1.0. It is the last neuron created above
		Neuron& biasNeuron = m_layers.back().back();
		biasNeuron.setOutputValue(1.0);
	}
}


void Net::feedForward(const std::vector<double>& inputVals) {
	// number of input neurons must be equal to the number input layer neurons 
	// -1 because the last neuron is a bias included by ourselves
	assert(inputVals.size() == m_layers[0].size() - 1);

	// assign the input values into input neurons
	for (unsigned int i = 0; i < inputVals.size(); i++) {
		m_layers[0][i].setOutputValue(inputVals[i]);
	}


	// forward propagate

	// skipping the input layer
	for (unsigned int layerInd = 1; layerInd < m_layers.size(); layerInd++) {
		Layer& prevLayer = m_layers[layerInd - 1];

		// going for each neuron, skipping the bias neuron (last one)
		for (unsigned int n = 0; n < m_layers[layerInd].size() - 1; n++) {
			m_layers[layerInd][n].feedForward(prevLayer);
		}
	}
}



void Net::backProp(const std::vector<double>& targetVals) {
	// calculate overall net error (RMS of output neuron errors), RMS - Root Mean Square Error
	Layer& outputLayer = m_layers.back();
	m_error = 0.0;

	// asserting targetVals size equals to outputLayer.size() - 1
	assert(targetVals.size() == outputLayer.size() - 1);

	// going for each neuron excluding bias one
	for (unsigned int n = 0; n < outputLayer.size() - 1; n++) {
		// difference between expected value and actual value
		double delta = targetVals[n] - outputLayer[n].getOutputValue();
		m_error += delta * delta;
	}

	m_error /= (outputLayer.size() - 1); // get average error squared
	m_error = sqrt(m_error); // RMS

	// implement a recent average measurement:
	// m_recentAverageError =
	//	(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
	//	/ (m_recentAverageSmoothingFactor + 1.0);

	// calculate output layer gradients
	for (unsigned int n = 0; n < outputLayer.size() - 1; n++) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// calculate gradients on hidden layers (hidden layers - all of the layers excluding input and output layers)
	for (unsigned int layerInd = m_layers.size() - 2; layerInd > 0; layerInd--) {
		Layer& hiddenLayer = m_layers[layerInd];
		Layer& nextLayer = m_layers[layerInd + 1];

		for (unsigned int n = 0; n < hiddenLayer.size(); n++) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// for all layers from outputs to first hidden layer,
	// update connection weights

	for (unsigned int layerInd = m_layers.size() - 1; layerInd > 0; layerInd--) {
		Layer& layer = m_layers[layerInd];
		Layer& prevLayer = m_layers[layerInd - 1];

		// going fr each neuron excluding bias one
		for (unsigned int n = 0; n < layer.size() - 1; n++) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}



void Net::getResults(std::vector<double>& resultVals) const {
	resultVals.clear();

	// going for each neuron in output layer (excluding the bias neuron)
	// and moves their output value to resultsVals 
	for (unsigned int n = 0; n < m_layers.back().size() - 1; n++) {
		double outputValOfNeuron = m_layers.back()[n].getOutputValue();
		resultVals.push_back(outputValOfNeuron);
	}
}