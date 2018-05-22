#pragma warning(disable:4996) //For the std::copy 

#include "Network.h"
#include <iostream>
#include <memory>

Network::Network(uint* topology, uint layerCount) : topology(topology), layerCount(layerCount) {
	//Initilize weights and biases
	weights = new double**[layerCount - 1];
	for (uint L = 0; L < layerCount - 1; L++) {
		weights[L] = new double*[topology[L + 1]];
		for (uint j = 0; j < topology[L + 1]; j++) {
			weights[L][j] = new double[topology[L]];
			for (uint k = 0; k < topology[L]; k++) {
				weights[L][j][k] = RandomNumber();
			}
		}
	}


	//Initilizing biases
	biases = new double*[layerCount - 1];
	for (uint L = 0; L < layerCount - 1; L++) {
		biases[L] = new double[topology[L + 1]];
		for (uint i = 0; i < topology[L + 1]; i++) {
			biases[L][i] = RandomNumber();
		}
	}

	//Initilize nodes
	nodes = new double*[layerCount];
	for (uint L = 0; L < layerCount; L++) {
		nodes[L] = new double[topology[L]];
		memset(nodes[L], 0, sizeof(double) * topology[L]);
	}
}

Network::~Network() {
	for (uint L = 0; L < layerCount - 1; L++) {
		for (uint j = 0; j < topology[L + 1]; j++) {
			delete[] weights[L][j];
		}

		delete[] weights[L];
		delete[] nodes[L];
	}

	delete[] weights;
	delete[] nodes;
	delete[] topology;
}

void Network::FeedForawrd(double* input) {
	std::copy(input, input + topology[0], nodes[0]);
	for (uint L = 0; L < layerCount - 1; L++) {
		for (uint j = 0; j < topology[L + 1]; j++) {
			nodes[L + 1][j] = 0; // Make sure that node is zeroed.
			for (uint k = 0; k < topology[L]; k++) {
				nodes[L + 1][j] += weights[L][j][k] * nodes[L][k];
			}

			nodes[L + 1][j] = Sigmoid(nodes[L + 1][j] + biases[L][j]);
		}
	}
}

void Network::BackProp(double* expected, double*** deltaWeights, double** deltaBiases) {

}

double Network::BackPropAndTrain(double* expected) {
	double*** deltaWeights = new double**[layerCount - 1];
	for (uint L = 0; L < layerCount - 1; L++) {
		deltaWeights[L] = new double*[topology[L + 1]];
		for (uint j = 0; j < topology[L + 1]; j++) {
			deltaWeights[L][j] = new double[topology[L]];
		}
	}

	double** deltaBiases = new double*[layerCount - 1];
	for (uint L = 0; L < layerCount - 1; L++) {
		deltaBiases[L] = new double[topology[L + 1]];
	}

	double** deltaNodes = new double*[layerCount - 1];
	for (uint L = 0; L < layerCount - 1; L++) {
		deltaNodes[L] = new double[topology[L + 1]];
	}

	std::copy(expected, expected + topology[layerCount - 1], deltaNodes[layerCount - 2]);

	for (uint i = 1; i < layerCount; i++) {
		uint L = layerCount - i;
		bool notInputLayer = L != 1;
		for (uint k = 0; k < topology[L - 1]; k++) {
			double r, s;
			double t = 1, sum = 0;
			double q = nodes[L - 1][k];
			for (uint j = 0; j < topology[L]; j++) {
				double u = weights[L - 1][j][k];
				r = SigmoidDerivative(nodes[L][j]);
				s = CostDerivative(nodes[L], deltaNodes[L - 1], topology[L]);
				deltaBiases[L - 1][j] = t * r * s;
				deltaWeights[L - 1][j][k] = q * r * s;
				sum += u * r * s;
			}

			if (notInputLayer) {
				deltaNodes[L - 2][k] = sum;
			}
		}
	}

	for (uint L = 0; L < layerCount - 1; L++) {
		for (uint j = 0; j < topology[L + 1]; j++) {
			for (uint k = 0; k < topology[L]; k++) {
				weights[L][j][k] -= deltaWeights[L][j][k];
			}
		}
	}

	for (uint L = 0; L < layerCount - 1; L++) {
		for (uint i = 0; i < topology[L + 1]; i++) {
			biases[L][i] -= deltaBiases[L][i];
		}
	}

	double cost = Cost(nodes[layerCount - 1], expected, topology[layerCount - 1]);

	for (uint L = 0; L < layerCount - 1; L++) {
		for (uint j = 0; j < topology[L + 1]; j++) {
			delete[] deltaWeights[L][j];
		}

		delete[] deltaNodes[L];
		delete[] deltaBiases[L];
		delete[] deltaWeights[L];
	}

	delete deltaNodes;
	delete deltaBiases;
	delete deltaWeights;

	return cost;
}