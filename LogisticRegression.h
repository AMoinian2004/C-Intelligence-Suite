#include <iostream>
#include <vector>
#include <cmath>

class LogisticRegression {
public:
    LogisticRegression(double lr = 0.01, int iters = 1000) : learningRate(lr), iterations(iters) {}

    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        size_t m = X.size();
        size_t n = X[0].size() + 1; // Adding 1 for the bias term

        std::vector<std::vector<double>> X_b(m, std::vector<double>(n, 1.0));
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n - 1; ++j) {
                X_b[i][j] = X[i][j];
            }
        }

        weights = std::vector<double>(n, 0.0); // Initialize weights

        // Gradient descent
        for (int i = 0; i < iterations; ++i) {
            std::vector<double> predictions = predict_prob(X_b);
            for (size_t j = 0; j < n; ++j) {
                double gradient = 0.0;
                for (size_t k = 0; k < m; ++k) {
                    gradient += (predictions[k] - y[k]) * X_b[k][j];
                }
                weights[j] -= learningRate * gradient / m;
            }
        }
    }

    std::vector<double> predict(const std::vector<std::vector<double>>& X) const {
        size_t m = X.size();
        size_t n = X[0].size() + 1;

        std::vector<std::vector<double>> X_b(m, std::vector<double>(n, 1.0));
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n - 1; ++j) {
                X_b[i][j] = X[i][j];
            }
        }

        std::vector<double> probs = predict_prob(X_b);
        std::vector<double> predictions(m);
        for (size_t i = 0; i < m; ++i) {
            predictions[i] = probs[i] >= 0.5 ? 1 : 0;
        }
        return predictions;
    }

    std::vector<double> getWeights() const {
        return weights;
    }

private:
    double learningRate;
    int iterations;
    std::vector<double> weights;

    std::vector<double> predict_prob(const std::vector<std::vector<double>>& X_b) const {
        size_t m = X_b.size();
        std::vector<double> probs(m);
        for (size_t i = 0; i < m; ++i) {
            probs[i] = sigmoid(dot_product(X_b[i], weights));
        }
        return probs;
    }

    static double sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
    }

    static double dot_product(const std::vector<double>& v1, const std::vector<double>& v2) {
        double result = 0.0;
        for (size_t i = 0; i < v1.size(); ++i) {
            result += v1[i] * v2[i];
        }
        return result;
    }
};
