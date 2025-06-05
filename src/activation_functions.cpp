#include "RcppArmadillo.h"

//[[Rcpp::depends(RcppArmadillo)]]

arma::mat sigmoid(const arma::mat & x) {
    return 1.0 / (1.0 + arma::exp(-1.0 * x));
}

arma::mat tanh(const arma::mat & x) {
    return arma::tanh(x);
}

arma::mat relu(const arma::mat & x) {
    int N = x.n_rows;
    int M = x.n_cols;
    
    arma::mat y = x;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            if (y(n, m) < 0.0) {
                y(n, m) = 0.0;
            }
        }
    }
    
    return y;
}

arma::mat sin(const arma::mat & x) {
    int N = x.n_rows;
    int M = x.n_cols;
    
    arma::mat y = x;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            y(n, m) = std::sin(y(n, m));
        }
    }
    
    return y;
}

arma::mat silu(const arma::mat & x) {
    return x / (1.0 + arma::exp(-1.0 * x));
}

arma::mat softplus(const arma::mat & x) {
    return arma::log(1.0 + arma::exp(x));
}

arma::mat softsign(const arma::mat & x) {
    return x / (1.0 + arma::abs(x));
}

arma::mat sqnl(const arma::mat & x) {
    int N = x.n_rows;
    int M = x.n_cols;
    
    arma::mat y = x;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            if (y(n, m) < -2.0) {
                y(n, m) = -1.0;
            }
            else if (y(n, m) < 0.0) {
                y(n, m) = y(n, m) + 0.25 * y(n, m) * y(n, m);
            }
            else if (y(n, m) < 2.0) {
                y(n, m) = y(n, m) - 0.25 * y(n, m) * y(n, m);
            }
            else {
                y(n, m) = 1.0;
            }
        }
    }
    
    return y;
}

arma::mat gaussian(const arma::mat & x) {
    return arma::exp(-1.0 * x * x);
}

arma::mat sqrbf(const arma::mat & x) {
    int N = x.n_rows;
    int M = x.n_cols;
    
    arma::mat y = x;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            if (std::abs(y(n, m)) < 1.0) {
                y(n, m) = 1.0 - 0.50 * y(n, m) * y(n, m);
            }
            else if (std::abs(y(n, m)) < 2.0) {
                y(n, m) = y(n, m) + 0.25 * y(n, m) * y(n, m);
            }
            else {
                y(n, m) = 0.0;
            }
        }
    }
    
    return y;
}

arma::mat bentidentity(const arma::mat & x) {
    return 0.5 * (arma::exp(0.5 * arma::log(x % x + 1.0)) - 1.0) + x;
}

arma::mat identity(const arma::mat & x) {
    return x;
}

