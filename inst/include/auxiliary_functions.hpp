#ifndef auxiliary_functions
#define auxiliary_functions

#include <RcppArmadillo.h>

arma::mat matrix_sign(const arma::mat & M);

arma::mat matrix_nonzero(const arma::mat & M);

bool matrix_condition(const arma::mat & M, const double & x);

double max_window(const arma::mat & x, const int & w, const int & i);

// GETS EXPORTED TO R
arma::colvec loocv(const arma::mat & O, const arma::colvec & y, const arma::colvec & b, const double & lambda);
arma::mat importance_score(const arma::mat &X, const arma::mat &W);
std::vector<std::string> classify_cpp(const arma::mat &y, const std::vector<std::string> &C, const double &t, const double &b);

#endif //auxiliary_functions