#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>

class KMeans {
public:
    KMeans(int k, int max_iters) : k(k), max_iters(max_iters) {}

    void fit(const std::vector<std::vector<double>>& X) {
        int n = X.size();
        int m = X[0].size();

        // Randomly initialize centroids
        initializeCentroids(X);

        for (int iter = 0; iter < max_iters; ++iter) {
            // Assign clusters
            std::vector<int> labels(n);
            for (int i = 0; i < n; ++i) {
                labels[i] = closestCentroid(X[i]);
            }

            // Update centroids
            std::vector<std::vector<double>> new_centroids(k, std::vector<double>(m, 0));
            std::vector<int> counts(k, 0);
            for (int i = 0; i < n; ++i) {
                int cluster = labels[i];
                for (int j = 0; j < m; ++j) {
                    new_centroids[cluster][j] += X[i][j];
                }
                counts[cluster]++;
            }

            for (int i = 0; i < k; ++i) {
                if (counts[i] != 0) {
                    for (int j = 0; j < m; ++j) {
                        new_centroids[i][j] /= counts[i];
                    }
                }
            }

            centroids = new_centroids;
        }
    }

    const std::vector<std::vector<double>>& getCentroids() const {
        return centroids;
    }

private:
    int k;
    int max_iters;
    std::vector<std::vector<double>> centroids;

    void initializeCentroids(const std::vector<std::vector<double>>& X) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, X.size() - 1);

        centroids.clear();
        for (int i = 0; i < k; ++i) {
            centroids.push_back(X[dis(gen)]);
        }
    }

    int closestCentroid(const std::vector<double>& point) {
        double min_dist = std::numeric_limits<double>::max();
        int closest = -1;

        for (int i = 0; i < k; ++i) {
            double dist = euclideanDistance(point, centroids[i]);
            if (dist < min_dist) {
                min_dist = dist;
                closest = i;
            }
        }

        return closest;
    }

    double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(sum);
    }
};

#endif // KMEANS_H
