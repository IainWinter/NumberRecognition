#pragma once
#include <cmath>

#define IW_E 2.7182818284590452353602874713527

typedef unsigned int uint;

double Sigmoid(double x);
double SigmoidDerivative(double x);
double RandomNumber();
double Cost(double* output, double* expected, uint count);
double CostDerivative(double* output, double* expected, uint count);