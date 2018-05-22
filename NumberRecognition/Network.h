#pragma once
#include "Common.h"

class Network {
private:
	double*** weights; //Array of matrixes (2d array)
	double** biases;
	double** nodes;
	const uint* topology;
	const uint layerCount;
	void BackProp(double* expected, double*** deltaWeights, double** deltaBiases);
public:
	Network(uint* topology, uint layerCount);
	~Network();
	void FeedForawrd(double* input);
	double BackPropAndTrain(double* expected);
};