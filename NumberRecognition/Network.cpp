#pragma warning(disable:4996) //For the std::copy 

#include "Network.h"
#include <iostream>
#include <memory>

Network::Network(uint* topology, uint layerCount) : topology(topology), layerCount(layerCount) {
	//Initilize weights
	weights = new double**[layerCount - 1];
	for (uint i = 0; i < layerCount - 1; i++) {
		uint crntCount = topology[i];
		uint nextCount = topology[i + 1];
		weights[i] = new double*[nextCount];
		for (uint j = 0; j < nextCount; j++) {
			weights[i][j] = new double[crntCount];
			for (uint k = 0; k < crntCount; k++) {
				weights[i][j][k] = RandomNumber();
			}
		}
	}

	//https://theclevermachine.wordpress.com/2014/09/06/derivation-error-backpropagation-gradient-descent-for-neural-networks/
	//https://stackoverflow.com/questions/35000215/neural-net-bias-per-layer-or-per-node-non-input-node
	//https://github.com/huangzehao/SimpleNeuralNetwork/blob/master/src/neural-net.cpp
	//Initilize nodes
	nodes = new double*[layerCount];
	for (uint i = 0; i < layerCount; i++) {
		nodes[i] = new double[topology[i]];
		memset(nodes[i], 0, sizeof(double) * topology[i]);
	}

	//Printing debug info

	/*
	std::cout << "Weights" << std::endl;
	for (size_t i = 0; i < layerCount - 1; i++) {
	std::cout << "Layer " << i << " to " << i + 1 << std::endl;
	for (size_t j = 0; j < topology[i + 1]; j++) {
	for (size_t k = 0; k < topology[i]; k++) {
	std::cout << weights[i][j][k] << " ";
	}
	std::cout << std::endl;
	}
	std::cout << std::endl;
	}

	std::cout << "Biases" << std::endl;
	for (size_t i = 0; i < layerCount - 1; i++) {
	std::cout << biases[i] << " ";
	}
	*/
}

Network::~Network() {
	for (uint i = 0; i < layerCount - 1; i++) {
		for (uint j = 0; j < topology[i + 1]; j++) {
			delete[] weights[i][j];
		}

		delete[] weights[i];
		delete[] nodes[i];
	}

	delete[] weights;
	delete[] nodes;
	delete[] biases;
	delete[] topology;
}

void Network::FeedForawrd(double* input) {
	std::copy(input, input + topology[0], nodes[0]);

	for (uint i = 0; i < layerCount - 1; i++) {
		for (uint j = 0; j < topology[i + 1]; j++) {
			for (uint k = 0; k < topology[i]; k++) {
				nodes[i + 1][j] += weights[i][j][k] * nodes[i][k] + biases[i][k];
			}

			nodes[i + 1][j] = Sigmoid(nodes[i + 1][j]);
		}
	}
}

void Network::BackProp(double* expected) {
	double*** gradient = new double**[layerCount - 1];
	for (uint i = 0; i < layerCount - 1; i++) {
		uint crntCount = topology[i];
		uint nextCount = topology[i + 1];
		gradient[i] = new double*[nextCount];
		for (uint j = 0; j < nextCount; j++) {
			gradient[i][j] = new double[crntCount];
		}
	}

	// Caculates the gradient of the cost function
	// Calculate the gradient for the output layer then the hidden layers
	// I think it can be all put together but maybe not the video make it seem like it was suppost to be

	uint L = layerCount - 1;

	for (uint j = 0; j < topology[L]; j++) {
		double error = nodes[L][j] - expected[j];
		double dSig = SigmoidDerivative(nodes[L][j]);
		double zk = nodes[L][j];
		for (uint k = 0; k < topology[L - 1]; k++) {
			double aj = nodes[L - 1][k];
			gradient[L - 1][j][k] = error * dSig * zk * aj;
			std::cout << gradient[L - 1][j][k] << std::endl;
		}
	}

	/*
	for (uint L = layerCount - 1; L > 0; L--) {
		for (uint j = 0; j < topology[L + 1]; j++) {
			for (uint k = 0; k < topology[L]; k++) {
				// Derivative of cost over weight
				double dzLj_dwLjk = nodes[L - 1][k];

				double daLj_dzLj = SigmoidDerivative(nodes[L][j]);

				double dc_daLj = CostDerivative(nodes[layerCount - 1], expected, topology[layerCount - 1]);

				double dc_dwLjk = dzLj_dwLjk * daLj_dzLj * dc_daLj;

				//// Derivative of cost over bias
				//double dzLj_dbL = 1.0;

				//double dc_dbL = dzLj_dbL * daLj_dzLj * dc_daLj;

				std::cout << dc_dwLjk << std::endl;
			}
		}
	}
	*/
}
