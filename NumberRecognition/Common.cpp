#include "Common.h"
#include <memory>
#include <fstream>

double Dot(double* x, double* y, uint count) {
	double dot = 0;
	for (uint i = 0; i < count; i++)
		dot += x[i] * y[i];
	return dot;
}

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

int ReverseInt(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void ReadMNIST(double** images, double** labels) {
	std::ifstream images_file("Data\\images", std::ios::binary);
	if (images_file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		images_file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		images_file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		images_file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		images_file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i < number_of_images; ++i) {
			for (int r = 0; r < n_rows; ++r) {
				for (int c = 0; c < n_cols; ++c) {
					unsigned char temp = 0;
					images_file.read((char*)&temp, sizeof(temp));
					images[i][(n_rows*r) + c] = (double)temp;
				}
			}
		}
	}

	std::ifstream labels_file("Data\\labels", std::ios::binary);
	if (labels_file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		labels_file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		labels_file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		for (int i = 0; i < number_of_images; ++i) {
			unsigned char temp = 0;
			labels_file.read((char*)&temp, sizeof(temp));
			for (int j = 0; j < 10; j++) {
				if (j == temp) {
					labels[i][j] = 1;
				} else {
					labels[i][j] = 0;
				}
			}
		}
	}
}