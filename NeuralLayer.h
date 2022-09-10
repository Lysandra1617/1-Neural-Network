#pragma once
#include "Activation.h"
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>

struct NeuralLayer {
	//# OF neurons
	int numberNeurons;
	//# OF weights (per neuron)
	int numberWeights;
	//# OF biases
	int numberBiases;
	//# OF error-parts
	int numberErrors;

	//Input from either a data set/piece or a previous layer
	double* input;
	//For each neuron in the hidden layer, it has a # of weights determined by the # of outputs
	double** weights;
	//Each neuron has a bias, although this isn't being updated in the NN, I might remove it
	double* biases;
	//Each neuron will have a final output value or activation number
	double* output;
	//Contains d[Total Error]/d[Neuron]
	double* dETdN;
	//Contains partial d[Total Error]/d[Neuron]
	double** dENdN;
	//Contains d[Total Error]/d[Weight] => Learning of NN
	double** dETdW;

	NeuralLayer(int numberInputs, int numberOutputs) {
		numberNeurons = numberOutputs;
		numberWeights = numberInputs;
		numberBiases = numberOutputs;
		numberErrors = 0;

		input = new double[numberInputs]; 
		output = new double[numberOutputs];

		//Each neuron will have a # of weights equal to the amount of inputs in the previous layer
		weights = new double*[numberNeurons];
		for (int i = 0; i < numberNeurons; i++) weights[i] = new double[numberInputs];
		//There is a bias for every neuron
		biases = new double[numberNeurons];
		//There will be a derivative for every neuron
		dETdN = new double[numberNeurons];
		dENdN = new double*[numberNeurons];
		//There will be a derivative for each of a neuron's weights
		dETdW = new double*[numberNeurons];
		for (int i = 0; i < numberNeurons; i++) dETdW[i] = new double[numberWeights];
	}

	//Initializing Weights & Biases
	void initializeTuners(double weight = 0.0, double bias = 0.0) {
		for (int n = 0; n < numberNeurons; n++) {
			for (int w = 0; w < numberWeights; w++) {
				weights[n][w] = weight;
			}
		}
		for (int b = 0; b < numberNeurons; b++) {
			biases[b] = bias;
		}
	}

	void initializeErrors(int numberErrors) {
		this->numberErrors = numberErrors;
		for (int i = 0; i < numberNeurons; i++) {
			dENdN[i] = new double[numberErrors];
			for (int j = 0; j < numberErrors; j++) {
				dENdN[i][j] = 0.0;
			}
		}
	}

	//Randomizing Weights & Biases
	void randomizeTuners(int min, int max) {
		for (int n = 0; n < numberNeurons; n++) {
			for (int w = 0; w < numberWeights; w++) {
				weights[n][w] = random(min, max);
			}
		}
		for (int b = 0; b < numberNeurons; b++) {
			biases[b] = random(min, max);
		}
	}

	//Generates Random # Between Min and Max
	double random(int min, int max) {
		if (min < 0 && max < 0)
			return 0.0;
		int adjustedMin = std::fmax(0, min);
		int adjustedMax = (min < 0) ? max + std::abs(min) : max;
		static std::random_device r;
		static std::default_random_engine e(r());
		static std::uniform_real_distribution<float> n(adjustedMin, adjustedMax);
		double rand = n(e);
		if (min >= 0)
			return rand;
		return rand - std::abs(min);
	}

	//Resetting Variables
	void resetVariables() {
		//Trying my best to prevent unexpected results
		for (int i = 0; i < numberNeurons; i++) {
			output[i] = 0.0;
			for (int j = 0; j < numberErrors; j++) {
				dENdN[i][j] = 0.0;
				dETdW[i][j] = 0.0;
			}
		}
	}

	//Prints Layer's Information
	void statistics() {
		//INPUTS
		std::cout << "Inputs for Layer: ";
		for (int i = 0; i < numberWeights; i++) {
			std::cout << input[i] << " ";
		}
		std::cout << std::endl;

		//DATA
		std::cout << "Layer Data:" << std::endl;
		for (int i = 0; i < numberNeurons; i++) {
			//WEIGHTS
			std::cout << "\tNeuron " << i << " => Weights: ";
			for (int j = 0; j < numberWeights; j++) {
				std::cout << std::fixed << std::setprecision(5) << weights[i][j] << (weights[i][j] >= 0.0 ? "  " : " ");
			}

			//BIASES
			std::cout << "| Bias: " << biases[i] << (biases[i] >= 0.0 ? "  " : " ");

			//OUTPUTS
			std::cout << "| Output: " << output[i] << std::endl;
		}

		std::cout << std::endl;
		std::cout << std::endl;

		//ERRORS
		std::cout << "Current d[Total Error]/d[Weight] for Layer:" << std::endl;
		for (int i = 0; i < numberNeurons; i++) {
			std::cout << "\tNeuron " << i << ": ";
			for (int w = 0; w < numberWeights; w++) {
				std::cout << std::fixed << std::setprecision(5) << dETdW[i][w] << (dETdW[i][w] >= 0.0 ? "  " : " ");
			}
		}

		std::cout << std::endl;
		std::cout << std::endl;
	}

	void forward(double* input, void (*activationFunction)(double* o, int n)) {
		this->input = input;
		double out = 0.0;
		for (int i = 0; i < numberNeurons; i++) {
			out = 0.0;
			for (int j = 0; j < numberWeights; j++) {
				out += input[j] * weights[i][j];
			}
			out += biases[i];
			output[i] = out;
		}
		//Running the outputs through the activation function
		activationFunction(output, numberNeurons);
	}

	void calculateErrors(double** previousLayerErrors) {
		//Note: The number of weights in the current layer is equal to the number of neurons in the previous layer
		int previousLayerNeurons = numberWeights;

		for (int n = 0; n < numberNeurons; n++) {
			for (int e = 0; e < numberErrors; e++) {
				for (int pN = 0; pN < previousLayerNeurons; pN++) {
					previousLayerErrors[pN][e] += dENdN[n][e] * Activation::dSigmoid(output[n]) * weights[n][pN];
				}
			}
		}
	}

	void backward(double learningRate) {
		//Resetting
		for (int n = 0; n < numberNeurons; n++) {
			dETdN[n] = 0.0;
		}

		//Adding parts of the d[Total Error]/d[Neuron] to find the whole
		for (int n = 0; n < numberNeurons; n++) {
			for (int e = 0; e < numberErrors; e++) {
				dETdN[n] += dENdN[n][e];
			}
		}

		//Finding d[Total Error]/d[Weight]
		for (int n = 0; n < numberNeurons; n++) {
			for (int w = 0; w < numberWeights; w++) {
				dETdW[n][w] = -learningRate * dETdN[n] * Activation::dSigmoid(output[n]) * input[w];
			}
		}
	}

	void adjustParameters() {
		for (int n = 0; n < numberNeurons; n++) {
			for (int w = 0; w < numberWeights; w++) {
				weights[n][w] += dETdW[n][w];
			}
		}
	}
};