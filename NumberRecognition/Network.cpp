#include "Network.h"
#include <iostream>

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
	activations = new double*[layerCount];
	for (uint L = 0; L < layerCount; L++) {
		nodes[L] = new double[topology[L]];
		activations[L] = new double[topology[L]];
	}
}

Network::~Network() {
	for (uint L = 0; L < layerCount - 1; L++) {
		for (uint j = 0; j < topology[L + 1]; j++) {
			delete[] weights[L][j];
		}

		delete[] weights[L];
		delete[] activations[L];
		delete[] nodes[L];
	}

	delete[] weights;
	delete[] activations;
	delete[] nodes;
	delete[] topology;
}

void Network::FeedForawrd(double* input) {
	for (uint i = 0; i < topology[0]; i++) {
		activations[0][i] = input[i];
		nodes[0][i] = input[i];
	}

	for (uint L = 0; L < layerCount - 1; L++) {
		for (uint j = 0; j < topology[L + 1]; j++) {
			nodes[L + 1][j] = Dot(weights[L][j], activations[L], topology[L]) + biases[L][j];
			activations[L + 1][j] = Sigmoid(nodes[L + 1][j]);

			//Debug("Node[" << L << ", " << j << "] " << nodes[L][j] << " "
				//<< "Activation[" << L << ", " << j << "] " << activations[L][j] << " "
				//<< "Weight[" << L << ", " << j << ", 0] " << weights[L][j][0] << std::endl);
		}
	}
}

void Network::Train(double** input, double** expected, uint trials, uint batch) {
	double*** weightGrad = new double**[layerCount - 1];
	for (uint L = 0; L < layerCount - 1; L++) {
		weightGrad[L] = new double*[topology[L + 1]];
		for (uint j = 0; j < topology[L + 1]; j++) {
			weightGrad[L][j] = new double[topology[L]];
			for (uint k = 0; k < topology[L]; k++) {
				weightGrad[L][j][k] = 0;
			}
		}
	}

	double** biasGrad = new double*[layerCount - 1];
	for (uint L = 0; L < layerCount - 1; L++) {
		biasGrad[L] = new double[topology[L + 1]];
		for (uint j = 0; j < topology[L + 1]; j++) {
			biasGrad[L][j] = 0;
		}
	}

	for (uint i = 0; i < trials / batch; i++) {
		double avgCost = 0;
		for (uint j = 0; j < batch; j++) {
			uint index = i * batch + j;
			FeedForawrd(input[index]);
			avgCost += BackProp(expected[index], weightGrad, biasGrad);
		}

		Debug("Batch " << i << " of " << trials / batch << " Average cost: " << avgCost / batch << std::endl);

		SubtractGrad(weightGrad, biasGrad, batch);
	}
}

double Network::BackProp(double* expected, double*** weightGrad, double** biasGrad) {
	double** deltaNodes = new double*[layerCount - 2];
	for (uint L = 0; L < layerCount - 2; L++) {
		deltaNodes[L] = new double[topology[L + 1]];
	}

	for (uint L = layerCount - 1; L > 1; L--) {
		double* layerDeltaNodes = L == layerCount - 1 ? expected : deltaNodes[L - 1];

		for (uint k = 0; k < topology[L - 1]; k++) {
			double sum = 0;
			double q = activations[L][k];
			for (uint j = 0; j < topology[L]; j++) {
				double u = weights[L - 1][j][k];
				double r = SigmoidDerivative(nodes[L - 1][j]);
				double s = CostDerivative(activations[L], layerDeltaNodes, topology[L]);
				double rs = r * s;
				biasGrad[L - 1][j] += rs;
				weightGrad[L - 1][j][k] += q * rs;

				sum += u * rs;
			}

			deltaNodes[L - 2][k] = sum;
		}
	}

	for (uint k = 0; k < topology[0]; k++) {
		double q = activations[0][k];
		for (uint j = 0; j < topology[1]; j++) {
			double r = SigmoidDerivative(nodes[0][j]);
			double s = CostDerivative(activations[2], deltaNodes[0], topology[1]);
			double rs = r * s;
			biasGrad[0][j] += rs;
			weightGrad[0][j][k] += q * rs;
		}
	}

	double cost = Cost(activations[layerCount - 1], expected, topology[layerCount - 1]);

	for (uint L = 0; L < layerCount - 2; L++) {
		delete[] deltaNodes[L];
	}

	delete deltaNodes;

	return cost;
}

void Network::SubtractGrad(double*** weightGrad, double** biasGrad, uint count) {
	for (uint L = 0; L < layerCount - 1; L++) {
		for (uint j = 0; j < topology[L + 1]; j++) {
			for (uint k = 0; k < topology[L]; k++) {
				std::cout << weights[L][j][k] << " ";
				weights[L][j][k] -= weightGrad[L][j][k] / count;
				weightGrad[L][j][k] = 0;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	for (uint L = 0; L < layerCount - 1; L++) {
		for (uint i = 0; i < topology[L + 1]; i++) {
			std::cout << biases[L][i] << " ";
			biases[L][i] -= biasGrad[L][i] / count;
			biasGrad[L][i] = 0;
		}
		std::cout << std::endl;
	}
}
