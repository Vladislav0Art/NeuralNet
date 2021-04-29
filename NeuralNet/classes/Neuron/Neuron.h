#pragma once

#include <random>

class Neuron;
typedef std::vector<Neuron> Layer;

struct Connection {
	double weight, deltaWeight;

	Connection() {
		weight = randomWeight();
	}

	double randomWeight() {
		std::uniform_real_distribution<double> unif(0, 1.0);
		std::default_random_engine random_engine;

		double value = unif(random_engine);
		return value;
	}
};



class Neuron {
public:
	Neuron(unsigned int numOfOutputs, unsigned int myIndex);

	void setOutputValue(const double val);
	double getOutputValue() const;
	void feedForward(const Layer& prevLayer);
	void calcOutputGradients(const double targetVal);
	void calcHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);


private:
	static double eta; // [0.0; 1.0] - overall net learning rate
	static double alpha; // [0.0; n] - multiplier of last weight change (momentum)

	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	double sumDOW(const Layer& nextLayer) const;

	double m_outputVal;
	double m_gradient;
	unsigned int m_myIndex;
	std::vector<Connection> m_outputWeights;
};