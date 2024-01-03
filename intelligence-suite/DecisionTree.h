#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include "TreeNode.h"
#include <vector>
#include <limits>
#include <algorithm>
#include <map>

class DecisionTree {
public:
    DecisionTree(int depth = 10) : maxDepth(depth), root(nullptr) {}

    ~DecisionTree() {
        clearTree(root);
    }

    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        root = buildTree(X, y, 0);
    }

    double predict(const std::vector<double>& sample) const {
        return predictSample(root, sample);
    }

private:
    TreeNode* root;
    int maxDepth;

    TreeNode* buildTree(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int depth) {
        if (X.empty() || depth == maxDepth) {
            return nullptr;
        }

        int bestFeature;
        double bestSplit;
        double bestGini = std::numeric_limits<double>::max();
        std::vector<std::vector<double>> leftX, rightX;
        std::vector<double> leftY, rightY;

        for (size_t i = 0; i < X[0].size(); ++i) {
            for (const auto& row : X) {
                double splitVal = row[i];
                std::vector<std::vector<double>> currentLeftX, currentRightX;
                std::vector<double> currentLeftY, currentRightY;

                for (size_t j = 0; j < X.size(); ++j) {
                    if (X[j][i] < splitVal) {
                        currentLeftX.push_back(X[j]);
                        currentLeftY.push_back(y[j]);
                    } else {
                        currentRightX.push_back(X[j]);
                        currentRightY.push_back(y[j]);
                    }
                }

                double gini = calculateGini(currentLeftY, currentRightY);
                if (gini < bestGini) {
                    bestGini = gini;
                    bestFeature = i;
                    bestSplit = splitVal;
                    leftX = currentLeftX;
                    rightX = currentRightX;
                    leftY = currentLeftY;
                    rightY = currentRightY;
                }
            }
        }

        if (leftX.empty() || rightX.empty()) {
            return new TreeNode{-1, 0, mostCommonClass(y), nullptr, nullptr};
        }

        TreeNode* node = new TreeNode{bestFeature, bestSplit, 0, nullptr, nullptr};
        node->left = buildTree(leftX, leftY, depth + 1);
        node->right = buildTree(rightX, rightY, depth + 1);

        return node;
    }

    double predictSample(TreeNode* node, const std::vector<double>& sample) const {
        if (node == nullptr) return 0;
        if (node->featureIndex == -1) return node->predictedClass;
        if (sample[node->featureIndex] < node->splitValue)
            return predictSample(node->left, sample);
        else
            return predictSample(node->right, sample);
    }

    void clearTree(TreeNode* node) {
        if (node) {
            clearTree(node->left);
            clearTree(node->right);
            delete node;
        }
    }

    double calculateGini(const std::vector<double>& leftY, const std::vector<double>& rightY) {
        auto leftScore = score(leftY);
        auto rightScore = score(rightY);
        double total = leftY.size() + rightY.size();

        return (leftScore * leftY.size() + rightScore * rightY.size()) / total;
    }

    double score(const std::vector<double>& y) {
        double sum = 0.0;
        for (auto label : y) {
            sum += label * label;
        }
        return 1.0 - sum / (y.size() * y.size());
    }

    double mostCommonClass(const std::vector<double>& y) {
        std::map<double, int> counts;
        for (double label : y) {
            counts[label]++;
        }
        return std::max_element(counts.begin(), counts.end(),
                                [](const auto& a, const auto& b) {
                                    return a.second < b.second;
                                })->first;
    }
};

#endif // DECISIONTREE_H
