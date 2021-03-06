#pragma once
#include <cmath>

#define IW_E 2.7182818284590452353602874713527

#define IW_DEBUG

#ifdef IW_DEBUG
#define Debug(x) std::cout << x
#else
#define Debug(x) 
#endif

typedef unsigned int uint;

double Dot(double* x, double* y, uint count);

double Sigmoid(double x);
double SigmoidDerivative(double x);
double Cost(double* output, double* expected, uint count);
double CostDerivative(double* output, double* expected, uint count);
double RandomNumber();
int ReverseInt(int i);
void ReadMNIST(double** images, double** labels);