#include "MLLibrary.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

int main() {
    // --- Linear Regression ---
    {
        std::vector<std::vector<double>> X = {{1.5}, {2.3}, {4.1}, {3.7}};
        std::vector<double> y = {2.8, 3.2, 5.9, 4.4};
        LinearRegression model;
        model.fit(X, y);
        std::vector<double> predictions = model.predict(X);
        std::cout << "Linear Regression Predictions:\n";
        for (const auto& pred : predictions) {
            std::cout << pred << " ";
        }
        std::cout << std::endl << std::endl;
    }
    
    // --- Logistic Regression ---
    {
        std::vector<std::vector<double>> X = {{1, 2}, {2, 3}, {4, 5}, {5, 6}};
        std::vector<double> y = {0, 0, 1, 1};
        LogisticRegression model(0.01, 1000);
        model.fit(X, y);
        std::vector<double> predictions = model.predict(X);
        std::cout << "Logistic Regression Predictions:\n";
        for (const auto& pred : predictions) {
            std::cout << pred << " ";
        }
        std::cout << std::endl << std::endl;
    }

    // --- Decision Tree ---
    {
        std::vector<std::vector<double>> X = {
            {1.5, 3.5},
            {2.2, 4.8},
            {3.8, 5.1},
            {4.5, 6.3},
            {3.1, 4.2},
            {4.0, 5.5}
        };
        std::vector<double> y = {1, 2, 3, 1, 0, 1};
        DecisionTree tree(5); // Maximum depth
        tree.fit(X, y);
        std::vector<double> sample = {4.0, 5.0};
        double prediction = tree.predict(sample);
        std::cout << "Decision Tree Prediction: " << prediction << std::endl << std::endl;
    }

    // --- K-Means Clustering ---
    {
        std::vector<std::vector<double>> X = {
            {1.2, 2.3},
            {1.8, 1.9},
            {5.0, 8.0},
            {8.5, 9.0},
            {1.1, 0.7},
            {9.2, 11.1}
        };
        KMeans kmeans(2, 100); // Clusters, iterations
        kmeans.fit(X);
        std::cout << "K-Means Centroids:\n";
        for (const auto& centroid : kmeans.getCentroids()) {
            for (double coord : centroid) {
                std::cout << coord << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // --- Data Splitting ---
    {
        std::vector<std::vector<double>> X = {
            {1.2, 2.1},
            {2.3, 3.2},
            {3.4, 4.3},
            {4.5, 5.4},
            {5.6, 6.5}
        };
        std::vector<double> y = {1.1, 2.2, 3.3, 4.4, 5.5};
        auto [train_set, test_set] = DataSplit::trainTestSplit(X, y, 0.2);
        std::cout << "Data Splitting - Training Set:\n";
        for (const auto& pair : train_set.first) {
            std::cout << "(" << pair[0] << ", " << pair[1] << ")" << std::endl;
        }
        std::cout << "\nData Splitting - Test Set:\n";
        for (const auto& pair : test_set.first) {
            std::cout << "(" << pair[0] << ", " << pair[1] << ")" << std::endl;
        }
        std::cout << std::endl;
    }

    // --- Neural Network ---
    {
        NeuralNetwork nn(2, 4, 1); // Input, hidden, output sizes
        std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        std::vector<double> y = {0, 1, 1, 0}; // XOR problem
        nn.train(X, y, 1000); // Train
        std::cout << "Neural Network Predictions:\n";
        for (const auto& sample : X) {
            double prediction = nn.predict(sample);
            std::cout << "Input: (" << sample[0] << ", " << sample[1] << ") - Prediction: " << prediction << std::endl;
        }
        std::cout << std::endl;
    }

    // --- Data Normalization & Standardization ---
    {
        std::vector<std::vector<double>> X = {{1.5, 2.2}, {2.8, 3.9}, {3.2, 4.6}, {4.1, 5.2}};
        auto normalizedData = DataNormalization::minMaxScaling(X);
        auto standardizedData = DataStandardization::zScoreNormalization(X);
        std::cout << "Data Normalization (Min-Max Scaling):\n";
        for (const auto& row : normalizedData) {
            for (auto val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "\nData Standardization (Z-score Normalization):\n";
        for (const auto& row : standardizedData) {
            for (auto val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // --- Monte Carlo Simulation ---
    {
        int num_samples = 1000000;
        int num_threads = 4;
        MonteCarlo mc(num_samples, num_threads);
        auto func = []() -> double {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            double x = dis(gen);
            double y = dis(gen);
            return (x * x + y * y <= 1.0) ? 1.0 : 0.0;
        };
        double pi_estimate = 4.0 * mc.simulate(func);
        std::cout << "Monte Carlo - Estimated Pi: " << pi_estimate << std::endl << std::endl;
    }

    // --- Testing Loss Functions ---
    {
        std::vector<double> y_true = {1.0, 0.0, 1.0, 0.0};
        std::vector<double> y_pred = {0.9, 0.1, 0.8, 0.2};
        double mse = mean_squared_error(y_true.data(), y_pred.data(), y_true.size());
        double cross_entropy = cross_entropy_loss(y_true.data(), y_pred.data(), y_true.size());
        std::cout << "Mean Squared Error: " << mse << std::endl;
        std::cout << "Cross-Entropy Loss: " << cross_entropy << std::endl;
    }

    return 0;
}
