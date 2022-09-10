#include <iostream>
#include <vector>

#include "NeuralNetwork.h"

int main() {
	//Data Used to Train
	std::vector<std::vector<double>> trainData = {
		{0.0, 1.0, 1.0},
		{0.0, 0.0, 0.0},
		{1.0, 0.0, 1.0},
		{1.0, 1.0, 1.0}
	};

	//A neural network with 2 initial inputs, 1 final output, and 1 hidden layer with 3 nodes
	int neuronsPerLayer[] = { 3 };
	NeuralNetwork nn(2, 1, 0.1, 1, neuronsPerLayer);

	std::cout << "NEURAL NETWORK:\n";
	std::cout << "This neural network is currently trained for the Logical OR operator.\nMeaning if you enter 1 (TRUE) as either or both operands, a 1 (TRUE) should be returned.\n\n";

	std::cout << "TESTING THE UNTRAINED NEURAL NETWORK:\n";
	std::cout << "Enter Pairs of 0s and 1s (e.g. 0 1, 1 0, 1 1, 0 0).\n";
	std::cout << "Invalid Input Begins Training.\n";

	std::vector<double> test(2);
	while (true) {
		std::cout << "Pair: ";
		std::cin >> test.at(0) >> test.at(1);
		if (test.at(0) != 0 && test.at(0) != 1 || test.at(1) != 0 && test.at(1) != 1) break;
		nn.test(test);
	}

	double lr = 0.0;
	int epochs = 0;
	std::cout << "TRAINING:\n";
	std::cout << "Enter a Learning Rate: ";
	std::cin >> lr;
	std::cout << "Enter the Number of Epochs: ";
	std::cin >> epochs; 
	
	nn.train(trainData, epochs, lr);
	std::cout << "NEURAL NETWORK HAS FINISHED TRAINING.\n\n";

	std::cout << "TESTING NEURAL NETWORK:\n";
	std::cout << "Enter Pairs of 0s and 1s (e.g. 0 1, 1 0, 1 1, 0 0).\n";
	std::cout << "Invalid Input Ends the Program.\n";
		
	while (true) {
		std::cout << "Pair: ";
		std::cin >> test.at(0) >> test.at(1);
		if (test.at(0) != 0 && test.at(0) != 1 || test.at(1) != 0 && test.at(1) != 1) break;
		nn.test(test);
	}
}