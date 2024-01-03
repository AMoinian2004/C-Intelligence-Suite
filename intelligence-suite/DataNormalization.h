#include <vector>
#include <algorithm>
#include <numeric>

class DataNormalization {
public:
    static std::vector<std::vector<double>> minMaxScaling(const std::vector<std::vector<double>>& data) {
        size_t numFeatures = data[0].size();
        size_t numSamples = data.size();
        std::vector<double> mins(numFeatures, std::numeric_limits<double>::max());
        std::vector<double> maxs(numFeatures, std::numeric_limits<double>::lowest());

        // Compute min and max for each feature
        for (const auto& sample : data) {
            for (size_t i = 0; i < numFeatures; ++i) {
                mins[i] = std::min(mins[i], sample[i]);
                maxs[i] = std::max(maxs[i], sample[i]);
            }
        }

        // Apply Min-Max Scaling
        std::vector<std::vector<double>> scaledData(numSamples, std::vector<double>(numFeatures));
        for (size_t i = 0; i < numSamples; ++i) {
            for (size_t j = 0; j < numFeatures; ++j) {
                scaledData[i][j] = (data[i][j] - mins[j]) / (maxs[j] - mins[j]);
            }
        }

        return scaledData;
    }
};
