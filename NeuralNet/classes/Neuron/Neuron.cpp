#include <vector>
#include <cmath>
#include "./Neuron.h"

// Layer == vector<Neuron>


Neuron::Neuron(unsigned int numOfOutputs, unsigned int myIndex) {
	for (unsigned int c = 0; c < numOfOutputs; c++) {
		m_outputWeights.push_back(Connection());
	}

	m_myIndex = myIndex;
	m_outputVal = 0.0;
}



void Neuron::setOutputValue(const double val) {
	m_outputVal = val;
	return;
}



double Neuron::getOutputValue() const {
	return m_outputVal;
}



void Neuron::feedForward(const Layer& prevLayer) {
	double sum = 0.0;

	// sum up previous layer's outputs, including the bias node from previous layer
	for (unsigned int n = 0; n < prevLayer.size(); n++) {
		sum +=  prevLayer[n].getOutputValue() * 
				prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
	return;
}



void Neuron::calcOutputGradients(const double targetVal) {
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}



void Neuron::calcHiddenGradients(const Layer& nextLayer) {
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}



void Neuron::updateInputWeights(Layer& prevLayer) {
	// the weights to be updated are in the Connection container
	// in the neurons in previous layer

	for (unsigned int n = 0; n < prevLayer.size(); n++) {
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight = 
			// individual input, magnified by the gradient and train rate:
			// eta - net learning rate
			eta 
			* neuron.getOutputValue()
			* m_gradient
			// also add momentum = fraction of the previous delta weight
			// alpha - momentum
			+ alpha
			* oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}


// statics

double Neuron::eta = 0.15; // overall net learning rate [0.0; 1.0]
double Neuron::alpha = 0.5; // multiplier of last deltaWeight, [0.0; n]

double Neuron::transferFunction(double x) {
	// using tanh(x) function: output range [-1.0; 1.0]
	return tanh(x);
}



double Neuron::transferFunctionDerivative(double x) {
	// tanh(x) derivative
	return 1.0 - x * x;
}



double Neuron::sumDOW(const Layer& nextLayer) const {
	double sum = 0.0;

	// sum our contributions of the errors at the nodes we feed in a next layer
	// excluding the bias neuron
	for (unsigned int n = 0; n < nextLayer.size() - 1; n++) {
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}