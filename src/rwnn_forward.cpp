#include "RcppArmadillo.h"

//[[Rcpp::depends(RcppArmadillo)]]

#include "auxiliary_functions.hpp"
#include "activation_functions.hpp"

typedef arma::mat (*FnPtr)(const arma::mat &);

std::map<std::string, FnPtr> activation_map = {
    {"sigmoid", sigmoid},
    {"tanh", tanh},
    {"sin", sin},
    {"relu", relu},
    {"silu", silu},
    {"softplus", softplus},
    {"softsign", softsign},
    {"sqnl", sqnl},
    {"gaussian", gaussian},
    {"sqrbf", sqrbf},
    {"bentidentity", bentidentity},
    {"identity", identity}
};

//[[Rcpp::export]]
std::vector<arma::mat> rwnn_forward(arma::mat X, 
                                    const std::vector<arma::mat> & W, 
                                    const std::vector<std::string> & activation,
                                    const std::vector<bool> & bias) {
    const unsigned int & N = X.n_rows;
    const unsigned int & M = W.size();
    
    arma::colvec b(N, arma::fill::ones);
    if (bias[0]) {
        X = arma::join_horiz(b, X);
    }
    
    std::vector<arma::mat> H(M); 
    H[0] = activation_map[activation[0]](X * W[0]);
    for (unsigned int m = 1; m < M; m++) {
        arma::mat H_m = H[m - 1];
        
        if (bias[m]) {
            H_m = arma::join_horiz(b, H_m);
        }
        
        H_m = H_m * W[m];
        H_m = activation_map[activation[m]](H_m);
        
        H[m] = H_m;
    } 
    
    return H;
}


