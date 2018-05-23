#include <iostream>
#include <ctime>
#include "Network.h"

int main() {
	srand(time(NULL));

	uint* topology = new uint[4]{
		784, 16, 16, 10
	};

	Network network(topology, 4);

	double** images = new double*[10000];
	for (size_t i = 0; i < 10000; i++) {
		images[i] = new double[784];
	}

	double** labels = new double*[10000];
	for (size_t i = 0; i < 10000; i++) {
		labels[i] = new double[10];
	}

	ReadMNIST(images, labels);

	std::cout << "Loaded symboles" << std::endl;

	network.Train(images, labels, 10000, 100);
}