#ifndef activation_functions
#define activation_functions

#include <RcppArmadillo.h>

arma::mat sigmoid(const arma::mat & x);

arma::mat tanh(const arma::mat & x);

arma::mat sin(const arma::mat & x);

arma::mat relu(const arma::mat & x);

arma::mat silu(const arma::mat & x);

arma::mat softplus(const arma::mat & x);

arma::mat softsign(const arma::mat & x);

arma::mat sqnl(const arma::mat & x);

arma::mat gaussian(const arma::mat & x);

arma::mat sqrbf(const arma::mat & x);

arma::mat bentidentity(const arma::mat & x);

arma::mat identity(const arma::mat & x);

#endif //activation_functions