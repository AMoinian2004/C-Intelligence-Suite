#include <iostream>
#include <vector>

class LinearRegression {
public:
    LinearRegression() {}

    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        size_t m = X.size();
        size_t n = X[0].size() + 1; // Adding 1 for the bias term

        std::vector<std::vector<double>> X_b(m, std::vector<double>(n, 1.0));
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n - 1; ++j) {
                X_b[i][j] = X[i][j];
            }
        }

        // Calculate (X^T * X)
        auto XtX = multiply(transpose(X_b), X_b);

        // Calculate (X^T * y)
        std::vector<double> Xty = multiply(transpose(X_b), y);

        // Calculate the coefficients (theta) using Normal Equation
        coefficients = solve(XtX, Xty);
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

        return multiply(X_b, coefficients);
    }

    std::vector<double> getCoefficients() const {
        return coefficients;
    }

private:
    std::vector<double> coefficients;

    std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix) {
        size_t m = matrix.size();
        size_t n = matrix[0].size();
        std::vector<std::vector<double>> transposed(n, std::vector<double>(m, 0.0));
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    std::vector<std::vector<double>> multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) const {
        size_t m = A.size();
        size_t n = B[0].size();
        size_t p = A[0].size();

        std::vector<std::vector<double>> product(m, std::vector<double>(n, 0.0));
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t k = 0; k < p; ++k) {
                    product[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return product;
    }

    std::vector<double> multiply(const std::vector<std::vector<double>>& A, const std::vector<double>& B) const {
        size_t m = A.size();
        size_t n = A[0].size();

        std::vector<double> product(m, 0.0);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                product[i] += A[i][j] * B[j];
            }
        }
        return product;
    }

    std::vector<double> solve(const std::vector<std::vector<double>>& A, const std::vector<double>& B) {
    int n = A.size();
    std::vector<std::vector<double>> aug(A); // Augmented matrix

    // Augmenting matrix A with B
    for (int i = 0; i < n; ++i) {
        aug[i].push_back(B[i]);
    }

    // Performing Gaussian elimination
    for (int i = 0; i < n; ++i) {
        // Search for maximum in this column
        double maxEl = std::abs(aug[i][i]);
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (std::abs(aug[k][i]) > maxEl) {
                maxEl = std::abs(aug[k][i]);
                maxRow = k;
            }
        }

        // Swap maximum row with current row (column by column)
        for (int k = i; k < n + 1; k++) {
            std::swap(aug[maxRow][k], aug[i][k]);
        }

        // Make all rows below this one 0 in current column
        for (int k = i + 1; k < n; k++) {
            double c = -aug[k][i] / aug[i][i];
            for (int j = i; j < n + 1; j++) {
                if (i == j) {
                    aug[k][j] = 0;
                } else {
                    aug[k][j] += c * aug[i][j];
                }
            }
        }
    }

    // Solve equation Ax=b for an upper triangular matrix A
    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = aug[i][n] / aug[i][i];
        for (int k = i - 1; k >= 0; k--) {
            aug[k][n] -= aug[k][i] * x[i];
        }
    }
    return x;
}

};
