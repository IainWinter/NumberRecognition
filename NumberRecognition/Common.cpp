#include "Common.h"

double Sigmoid(double x) {
	return 1.0 / (1.0 + pow(IW_E, -x));
}

double SigmoidDerivative(double x) {
	return Sigmoid(x) * (1 - Sigmoid(x));
}

double RandomNumber() {
	return rand() / (RAND_MAX / 2.0) - 1.0;
}

double Cost(double* output, double* expected, uint count) {
	double cost = 0;
	for (uint i = 0; i < count; i++) {
		double delta = (output[i] - expected[i]);
		cost += delta * delta;
	}

	return cost;
}

double CostDerivative(double* output, double* expected, uint count) {
	double cost = 0;
	for (uint i = 0; i < count; i++) {
		cost += 2 * (output[i] - expected[i]);
	}

	return cost;
}
