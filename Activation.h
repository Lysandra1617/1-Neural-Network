#pragma 
#include <cmath>

struct Activation {
	//Sigmoid Function
	static void Sigmoid(double* outputs, int n) {
		for (int i = 0; i < n; i++) {
			outputs[i] = 1.0 / (1 + std::exp(-outputs[i]));
		}
	}

	//Derivative of the Sigmoid Function
	static double dSigmoid(double n) {
		return n*(1.0 - n);
	}
};