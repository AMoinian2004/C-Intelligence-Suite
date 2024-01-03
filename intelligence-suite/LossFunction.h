#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#ifdef __cplusplus
extern "C" {
#endif

double mean_squared_error(const double* y_true, const double* y_pred, int length);
double cross_entropy_loss(const double* y_true, const double* y_pred, int length);

#ifdef __cplusplus
}
#endif

#endif // LOSSFUNCTION_H
