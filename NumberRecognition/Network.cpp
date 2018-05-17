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
				weights[L][j][k] = j + k; //RandomNumber();
			}
		}
	}


	//Initilizing biases
	biases = new double*[layerCount - 1];
	for (uint L = 0; L < layerCount - 1; L++) {
		biases[L] = new double[topology[L + 1]];
		for (uint i = 0; i < topology[L + 1]; i++) {
			biases[L][i] = i; //RandomNumber();
		}
	}

	//Initilize nodes
	nodes = new double*[layerCount];
	for (uint L = 0; L < layerCount; L++) {
		nodes[L] = new double[topology[L]];
		memset(nodes[L], 0, sizeof(double) * topology[L]);
	}

	//Debug info
	std::cout << "Weights:" << std::endl;
	for (uint L = 0; L < layerCount - 1; L++) {
		std::cout << "Layer " << L << " to " << L + 1 << std::endl;
		for (uint j = 0; j < topology[L + 1]; j++) {
			for (uint k = 0; k < topology[L]; k++) {
				std::cout << weights[L][j][k] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	std::cout << "Biases:" << std::endl;
	for (uint L = 0; L < layerCount - 1; L++) {
		for (uint i = 0; i < topology[L + 1]; i++) {
			std::cout << biases[L][i] << " ";
		}
		std::cout << std::endl;
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
	std::cout << std::endl << "Feed forward:" << std::endl;
	std::copy(input, input + topology[0], nodes[0]);
	for (uint L = 0; L < layerCount - 1; L++) {
		std::cout << "Layer " << L << " to " << L + 1 << std::endl;
		for (uint j = 0; j < topology[L + 1]; j++) {
			for (uint k = 0; k < topology[L]; k++) {
				std::cout << "Node: " << L << " " << j << ": " << nodes[L + 1][j] << " += " <<
					"Weight: " << L << " " << j << " " << k << ": " << weights[L][j][k] << " * " <<
					"Node: " << L << " " << k << ": " << nodes[L][k] << std::endl;

				nodes[L + 1][j] += weights[L][j][k] * nodes[L][k];
			}

			std::cout << "Node: " << L << " " << j << ": " << nodes[L + 1][j] << " + " << "Bias: " << j << ": " << biases[L][j] << std::endl;

			double DEBUG_NODE_VALUE = nodes[L + 1][j];

			nodes[L + 1][j] = Sigmoid(nodes[L + 1][j] + biases[L][j]);

			std::cout << "Node: " << L << " " << j << ": " << nodes[L + 1][j] << " = " << "Sig(" << DEBUG_NODE_VALUE << ")" << std::endl;

		}
	}
}

void Network::BackProp(double* expected) {
	double** expectedNodes = new double*[layerCount - 1];
	for (uint L = 0; L < layerCount - 2; L++) {
		expectedNodes[L] = new double[topology[L + 1]];
		memset(expectedNodes[L], 0, sizeof(double) * topology[L + 1]);
	}

	std::copy(expected, expected + topology[0], expectedNodes[layerCount - 2]);

	for (uint L = layerCount - 1; L > 0; L--) {
		double cost = CostDerivative(nodes[L], expectedNodes[L - 1], topology[L]);
		for (uint k = 0; k < topology[L - 1]; k++) {
			for (uint j = 0; j < topology[L]; j++) {

			}
		}
		
	}

	//double cost = CostDerivative(nodes[layerCount - 1], expected, topology[layerCount - 1]);

	//for (uint L = layerCount - 1; L > 0; L++) {
	//	double layerCost = CostDerivative(nodes);
	//}
}

//void Network::BackProp(double* expected) {
//	//Weight gradient
//	double*** deltaWeights = new double**[layerCount - 1];
//	for (uint L = 0; L < layerCount - 1; L++) {
//		uint crntCount = topology[L];
//		uint nextCount = topology[L + 1];
//		deltaWeights[L] = new double*[nextCount];
//
//		for (uint j = 0; j < nextCount; j++) {
//			deltaWeights[L][j] = new double[crntCount];
//			memset(deltaWeights[L][j], 0, sizeof(double) * topology[L]);
//		}
//	}
//
//	//Bais gradient
//	double** deltaBiases = new double*[layerCount - 1];
//	for (uint L = 0; L < layerCount - 1; L++) {
//		deltaBiases[L] = new double[topology[L + 1]];
//		memset(deltaBiases[L], 0, sizeof(double) * topology[L + 1]);
//	}
//
//	//Back prop nodes
//	double** deltaNodes = new double*[layerCount];
//	for (uint L = 0; L < layerCount; L++) {
//		deltaNodes[L] = new double[topology[L]];
//		for (uint j = 0; j < topology[L]; j++) {
//			deltaNodes[L][j] = nodes[L][j];
//		}
//	}
//
//	for (uint L = layerCount - 1; L > 0; L--) {
//		for (uint j = 0; j < topology[L]; j++) {
//			double dCost = CostDerivative(deltaNodes[L], expected, topology[L]);
//			double dAct = SigmoidDerivative(deltaNodes[L][j]);
//			deltaBiases[L - 1][j] += dCost * dAct;
//			for (uint k = 0; k < topology[L - 1]; k++) {
//				double dZdW = deltaNodes[L - 1][k];
//				deltaWeights[L][j][k] += dCost * dAct * dZdW;
//			}
//		}
//
//		for (uint k = 0; k < topology[L - 1]; k++) {
//			for (uint j = 0; j < topology[L]; j++) {
//				double dCost = CostDerivative(deltaNodes[L], expected, topology[L]);
//				double dAct = SigmoidDerivative(deltaNodes[L][j]);
//				deltaNodes[L - 1][k] += weights[L][j][k] * dCost * dAct;
//			}
//		}
//	}
//
//	for (uint L = 0; L < layerCount - 1; L++) {
//		for (uint j = 0; j < topology[L + 1]; j++) {
//			for (uint k = 0; k < topology[L]; k++) {
//				weights[L][j][k] += deltaWeights[L][j][k];
//			}
//		}
//	}
//
//	for (uint L = 0; L < layerCount - 1; L++) {
//		for (uint j = 0; j < topology[L]; j++) {
//			biases[L][j] += deltaBiases[L][j];
//		}
//	}
//
//	for (uint L = 0; L < layerCount - 1; L++) {
//		for (uint j = 0; j < topology[L]; j++) {
//			nodes[L][j] = deltaNodes[L][j];
//		}
//	}
//}
