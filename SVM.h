#ifndef SVM_H
#define SVM_H

#include <vector>
#include <cmath>
#include <random>
#include <numeric>

class SVM
{
public:
    SVM(double c = 1.0, double learningRate = 0.01, int epochs = 1000)
        : C(c), learning_rate(learningRate), epochs(epochs) {}

    void fit(const std::vector<std::vector<double>> &X, const std::vector<double> &y)
    {
        int n_samples = X.size();
        int n_features = X[0].size();

        weights.resize(n_features, 0);
        bias = 0;

        std::vector<double> y_mod(y.begin(), y.end());
        for (double &val : y_mod)
        {
            if (val == 0)
                val = -1; // Convert 0 labels to -1 for SVM
        }

        // Stochastic Gradient Descent
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            for (int i = 0; i < n_samples; ++i)
            {
                double condition = y_mod[i] * (dotProduct(X[i], weights) + bias);

                if (condition < 1)
                {
                    for (int j = 0; j < n_features; ++j)
                    {
                        weights[j] += learning_rate * (y_mod[i] * X[i][j] - (2 * (1 / epochs) * weights[j]));
                    }
                    bias += learning_rate * y_mod[i];
                }
                else
                {
                    for (int j = 0; j < n_features; ++j)
                    {
                        weights[j] -= learning_rate * (2 * (1 / epochs) * weights[j]);
                    }
                }
            }
        }
    }

    double predict(const std::vector<double> &sample) const
    {
        double linear_output = dotProduct(sample, weights) + bias;
        return linear_output >= 0 ? 1 : 0; // Returns 1 for positive class and 0 for negative class
    }

private:
    std::vector<double> weights;
    double bias;
    double C;
    double learning_rate;
    int epochs;

    double dotProduct(const std::vector<double> &vec1, const std::vector<double> &vec2) const
    {
        double product = 0;
        for (size_t i = 0; i < vec1.size(); ++i)
        {
            product += vec1[i] * vec2[i];
        }
        return product;
    }
};

#endif // SVM_H
