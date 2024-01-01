#include <math.h>

// Mean Squared Error Loss
double mean_squared_error(const double* y_true, const double* y_pred, int length) {
    double mse = 0.0;
    for (int i = 0; i < length; ++i) {
        double diff = y_true[i] - y_pred[i];
        mse += diff * diff;
    }
    return mse / length;
}

// Cross-Entropy Loss
double cross_entropy_loss(const double* y_true, const double* y_pred, int length) {
    double ce = 0.0;
    for (int i = 0; i < length; ++i) {
        ce -= y_true[i] * log(y_pred[i]) + (1 - y_true[i]) * log(1 - y_pred[i]);
    }
    return ce / length;
}
