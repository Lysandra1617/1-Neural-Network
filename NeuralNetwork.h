#pragma once
#include <vector>
#include <iterator>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "NeuralLayer.h"

struct NeuralNetwork {
	//Hidden Layers of NN
	std::vector<NeuralLayer*> hiddenLayers;
	//Output Layer of NN
	NeuralLayer* outputLayer;
	
	//# OF initial inputs
	int initialNumberInputs;
	//# OF final outputs
	int finalNumberOutputs;
	//# OF hidden layers
	int numberHiddenLayers;

	double learningRate;
	
	NeuralNetwork(int numInputs, int numOutputs, double rate, int numHiddenLayers, int hiddenLayerNeurons[]) {
		initialNumberInputs = numInputs;
		finalNumberOutputs = numOutputs;
		learningRate = rate;
		numberHiddenLayers = numHiddenLayers;
		hiddenLayers.resize(numberHiddenLayers);
		outputLayer = nullptr;
		initializeLayers(hiddenLayerNeurons);
	}

	//Given an array with elements representing the number of neurons 
	//in each hidden layer from the 0th-index to the end, we initialize
	//the hidden layers of the NN
	void initializeLayers(int hiddenLayerNeurons[]) {
		for (int layer = 0; layer < numberHiddenLayers; layer++) {
			
			//The # of inputs for a hidden layer is equal to the number of outputs/neurons in the layer before it.
			//The first layer's # of inputs will be equal to the initial number of inputs for the NN,
			//this is because the first layer is technically the input layer, and also because accessing an element of index -1 will break the program
			int layerNumberInputs = layer == 0 ? initialNumberInputs : hiddenLayerNeurons[layer - 1];

			//The number of neurons, and therefore outputs, of a hidden layer is given by the array
			int layerNumberOutputs = hiddenLayerNeurons[layer];

			//Initializing
			hiddenLayers.at(layer) = new NeuralLayer(layerNumberInputs, layerNumberOutputs);
			hiddenLayers.at(layer)->randomizeTuners(-1, 1);
			hiddenLayers.at(layer)->initializeErrors(finalNumberOutputs);

		}
		//Output Layer
		//The output layer will take the neurons of the last hidden layer as inputs, therefore the output layer's # of inputs
		//will be equal to the number of neurons/outputs in the last hidden layer. The output layer will have a number of outputs equal to the final # of outputs
		//as it is the last layer.
		outputLayer = new NeuralLayer(hiddenLayerNeurons[numberHiddenLayers - 1], finalNumberOutputs);
		outputLayer->randomizeTuners(-1, 1);
		outputLayer->initializeErrors(finalNumberOutputs);
	}

	void train(std::vector<std::vector<double>>& trainData, int epochs, double learningRate) {
		int epoch = 0; //# OF tests on a data set
		std::vector<double> X; //Input Data for Training
		std::vector<double> Y; //Output Data for Testing

		while (epoch < epochs) {

			//For each data piece in the data set (e.g. {0.0, 1.0, 1.0}
			for (int t = 0; t < trainData.size(); t++) {

				//Extracting Inputs & Outputs in the Data Piece
				X = extractVector(trainData.at(t), 0, initialNumberInputs);
				Y = extractVector(trainData.at(t), initialNumberInputs, initialNumberInputs + finalNumberOutputs);

				//FORWARD PASS:
				forward(X);

				//Errors: We must deal with them before backpropagation
				//The total error is determined by the outputs/neurons of the output layer.
				//For example, if there are 3 final outputs (O1, O2, O3), they will all have a part in the total error.
				//Total Error = 1/2(TargetO1 - O1)^2 + 1/2(TargetO2 - O2)^2 + 1/2(TargetO3 - O3)^2
				//If we were to take the derivative of the total error with respect to a component output On we would get
				//d[Total Error]/d[On] = (On - TargetOn)
				for (int i = 0; i < finalNumberOutputs; i++) {
					//Finding d[Total Error]/d[On] and Recording
					outputLayer->dENdN[i][i] = outputLayer->output[i] - Y[i];
					std::cout << "Accuracy (" << t << "): " << (1 - (std::abs(outputLayer->output[i] - Y[i]))) * 100 << "%" << std::endl;
				}

				//Given the d[Total Error]/d[On] for each of the final output neurons,
				//we can find the derivative of the total error with respect to a neuron in the hidden layer.
				//As we are doing backpropagation, we are going backwards in the hidden layers.
				for (int n = numberHiddenLayers; n >= 1; n--) {
					//The errors that we are evaulating
					double** errors = hiddenLayers.at(n - 1)->dENdN;
					if (n == numberHiddenLayers) outputLayer->calculateErrors(errors);
					else hiddenLayers.at(n)->calculateErrors(errors);
				}

				//BACKPROPAGATION:
				backpropagation();
			}
			epoch++;
			std::cout << std::endl;
		}
	}

	//Given a vector and a valid start and last index, we are just returning a subvector of the vector.
	//This is used to extract the inputs and outputs from a data piece.
	std::vector<double> extractVector(std::vector<double>& vector, int start, int last) {
		auto begin = vector.begin() + start;
		auto end = vector.begin() + last;
		std::vector<double> v(begin, end);
		return v;
	}

	//Forward Pass
	void forward(std::vector<double> X) {
		//Turning the vector X into an array, it's better that way
		double* X_ = new double[initialNumberInputs];
		std::copy(X.begin(), X.end(), X_);

		//Doing a forward pass, which is basically just plugging in the initial inputs and running it through the NN
		for (int i = 0; i < numberHiddenLayers; i++) {
			//The first hidden layer gets the initial inputs
			if (i == 0) hiddenLayers[i]->forward(X_, &Activation::Sigmoid);
			//The other hidden layers gets the outputs of the previous hidden layer as its input
			else hiddenLayers[i]->forward(hiddenLayers[i - 1]->output, &Activation::Sigmoid);
		}
		//Output layer gets the output from the last hidden layer
		outputLayer->forward(hiddenLayers[numberHiddenLayers - 1]->output, &Activation::Sigmoid);
		//Note: The sigmoid function is used for the activation function. The activation function is applied after the output has been calculated.
		delete[] X_;
	}

	void backpropagation() {
		//2. Backward Pass:
		//Now that we know the derivative of the total error with respect to each neuron, we can
		//do a backwards pass. The only thing is I am calculating the change in the weights, I don't actually commit the change (yet).
		outputLayer->backward(learningRate);
		for (int n = numberHiddenLayers - 1; n > -1; n--) {
			hiddenLayers.at(n)->backward(learningRate);
		}

		//3. Commit:
		//Now we can fine-tune the weights as needed. Must reset the errors to prepare for the next iterations.
		for (int n = 0; n < numberHiddenLayers; n++) {
			hiddenLayers.at(n)->adjustParameters();
			hiddenLayers.at(n)->resetVariables();
		}
		outputLayer->adjustParameters();
		outputLayer->resetVariables();
	}

	//Testing the NN against input X
	void test(std::vector<double> X) {
		forward(X);
		for (int i = 0; i < finalNumberOutputs; i++) {
			std::cout << "RAW OUTPUT: " << outputLayer->output[i] << ", ROUNDED OUTPUT: " << std::round(outputLayer->output[i]) << std::endl;
		}
	}

};