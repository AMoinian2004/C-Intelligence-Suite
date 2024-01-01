#ifndef DATASPLIT_H
#define DATASPLIT_H

#include <vector>
#include <algorithm>
#include <random>

class DataSplit {
public:
    static std::pair<std::pair<std::vector<std::vector<double>>, std::vector<double>>,
                     std::pair<std::vector<std::vector<double>>, std::vector<double>>>
    trainTestSplit(const std::vector<std::vector<double>>& X, 
                   const std::vector<double>& y, 
                   double test_size = 0.2) {

        std::vector<int> indices(X.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_shuffle(indices.begin(), indices.end());

        int test_size_count = static_cast<int>(X.size() * test_size);
        std::vector<std::vector<double>> X_train, X_test;
        std::vector<double> y_train, y_test;

        for (int i = 0; i < test_size_count; ++i) {
            X_test.push_back(X[indices[i]]);
            y_test.push_back(y[indices[i]]);
        }

        for (size_t i = test_size_count; i < X.size(); ++i) {
            X_train.push_back(X[indices[i]]);
            y_train.push_back(y[indices[i]]);
        }

        return std::make_pair(std::make_pair(X_train, y_train), std::make_pair(X_test, y_test));
    }
};

#endif // DATASPLIT_H
