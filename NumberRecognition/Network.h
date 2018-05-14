#pragma once
#include "Common.h"

class Network {
private:
	double*** weights; //Array of matrixes (2d array)
	double** nodes;
	double* biases;
	const uint* topology;
	const uint layerCount;
public:
	Network(uint* topology, uint layerCount);
	~Network();
	void FeedForawrd(double* inputs);
	void BackProp(double* expected);

	inline const uint* Topology() const {
		return topology;
	}

	inline const uint LayerCount() const {
		return layerCount;
	}
};