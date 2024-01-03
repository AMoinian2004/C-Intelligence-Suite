#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <cmath>
#include <random>
#include <numeric>

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size) 
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size),
          learning_rate(0.01) {

        // Allocate memory for weights and biases
        input_hidden_weights = new double*[input_size];
        for (int i = 0; i < input_size; ++i) {
            input_hidden_weights[i] = new double[hidden_size];
        }

        hidden_output_weights = new double*[hidden_size];
        for (int i = 0; i < hidden_size; ++i) {
            hidden_output_weights[i] = new double[output_size];
        }

        hidden_biases = new double[hidden_size]{};
        output_biases = new double[output_size]{};

        // Initialize weights and biases
        initializeWeightsAndBiases();
    }

    ~NeuralNetwork() {
        // Free memory for weights and biases
        for (int i = 0; i < input_size; ++i) {
            delete[] input_hidden_weights[i];
        }
        delete[] input_hidden_weights;

        for (int i = 0; i < hidden_size; ++i) {
            delete[] hidden_output_weights[i];
        }
        delete[] hidden_output_weights;

        delete[] hidden_biases;
        delete[] output_biases;
    }

    void train(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < X.size(); ++i) {
                // Forward pass
                std::vector<double> hidden_layer_output = forwardPass(X[i], input_hidden_weights, hidden_biases, hidden_size);
                std::vector<double> output_layer_output = forwardPass(hidden_layer_output, hidden_output_weights, output_biases, output_size);

                // Backward pass
                std::vector<double> output_layer_delta = outputLayerDelta(y[i], output_layer_output);
                std::vector<double> hidden_layer_delta = hiddenLayerDelta(hidden_layer_output, output_layer_delta);

                // Update weights and biases
                updateWeightsAndBiases(X[i], hidden_layer_output, output_layer_delta, hidden_layer_delta);
            }
        }
    }

    double predict(const std::vector<double>& sample) const {
        std::vector<double> hidden_layer_output = forwardPass(sample, input_hidden_weights, hidden_biases, hidden_size);
        std::vector<double> output_layer_output = forwardPass(hidden_layer_output, hidden_output_weights, output_biases, output_size);
        return output_layer_output[0];
    }

    void setLearningRate(double lr) {
        learning_rate = lr;
    }

private:
    int input_size, hidden_size, output_size;
    double learning_rate;
    double** input_hidden_weights;
    double** hidden_output_weights;
    double* hidden_biases;
    double* output_biases;

    void initializeWeightsAndBiases() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                input_hidden_weights[i][j] = dis(gen);
            }
        }

        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                hidden_output_weights[i][j] = dis(gen);
            }
            hidden_biases[i] = dis(gen);
        }

        for (int i = 0; i < output_size; ++i) {
            output_biases[i] = dis(gen);
        }
    }

    std::vector<double> forwardPass(const std::vector<double>& input, double** weights, double* biases, int layer_size) const {
        std::vector<double> layer_output(layer_size, 0.0);
        for (int i = 0; i < layer_size; ++i) {
            for (size_t j = 0; j < input.size(); ++j) {
                layer_output[i] += input[j] * weights[j][i];
            }
            layer_output[i] += biases[i];
            layer_output[i] = sigmoid(layer_output[i]);
        }
        return layer_output;
    }

    double sigmoid(double x) const {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoidDerivative(double x) const {
        return x * (1 - x);
    }

    std::vector<double> outputLayerDelta(double target, const std::vector<double>& output) const {
        std::vector<double> delta(output_size);
        for (int i = 0; i < output_size; ++i) {
            delta[i] = (target - output[i]) * sigmoidDerivative(output[i]);
        }
        return delta;
    }

    std::vector<double> hiddenLayerDelta(const std::vector<double>& hidden_output, const std::vector<double>& output_delta) const {
        std::vector<double> delta(hidden_size, 0.0);
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                delta[i] += output_delta[j] * hidden_output_weights[i][j];
            }
            delta[i] *= sigmoidDerivative(hidden_output[i]);
        }
        return delta;
    }

    void updateWeightsAndBiases(const std::vector<double>& input, const std::vector<double>& hidden_output, const std::vector<double>& output_delta, const std::vector<double>& hidden_delta) {
        // Update weights and biases for the hidden-output layer
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                hidden_output_weights[i][j] += learning_rate * output_delta[j] * hidden_output[i];
            }
            hidden_biases[i] += learning_rate * hidden_delta[i];
        }

        // Update weights and biases for the input-hidden layer
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                input_hidden_weights[i][j] += learning_rate * hidden_delta[j] * input[i];
            }
        }
    }
};

#endif // NEURALNETWORK_H
