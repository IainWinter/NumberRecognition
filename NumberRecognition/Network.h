#pragma once
#include "Common.h"

class Network {
private:
	double*** weights; //Array of matrixes (2d array)
	double** biases;
	double** activations;
	double** nodes;
	const uint* topology;
	const uint layerCount;

	double BackProp(double* expected, double*** weightGrad, double** biasGrad);
	void SubtractGrad(double*** weightGrad, double** biasGrad, uint count);
public:
	Network(uint* topology, uint layerCount);
	~Network();

	inline const uint* Topology() const { return topology; }
	inline uint LayerCount() const { return layerCount; }

	void FeedForawrd(double* input);
	void Train(double** input, double** expected, uint trials, uint batch);
};