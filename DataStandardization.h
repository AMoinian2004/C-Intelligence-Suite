#include <cmath>

class DataStandardization {
public:
    static std::vector<std::vector<double>> zScoreNormalization(const std::vector<std::vector<double>>& data) {
        size_t numFeatures = data[0].size();
        size_t numSamples = data.size();
        std::vector<double> means(numFeatures, 0.0);
        std::vector<double> stdDevs(numFeatures, 0.0);

        // Compute mean for each feature
        for (const auto& sample : data) {
            for (size_t i = 0; i < numFeatures; ++i) {
                means[i] += sample[i];
            }
        }
        for (double& mean : means) {
            mean /= numSamples;
        }

        // Compute standard deviation for each feature
        for (const auto& sample : data) {
            for (size_t i = 0; i < numFeatures; ++i) {
                stdDevs[i] += std::pow(sample[i] - means[i], 2);
            }
        }
        for (double& stdDev : stdDevs) {
            stdDev = std::sqrt(stdDev / numSamples);
        }

        // Apply Z-score normalization
        std::vector<std::vector<double>> normalizedData(numSamples, std::vector<double>(numFeatures));
        for (size_t i = 0; i < numSamples; ++i) {
            for (size_t j = 0; j < numFeatures; ++j) {
                normalizedData[i][j] = (data[i][j] - means[j]) / stdDevs[j];
            }
        }

        return normalizedData;
    }
};
