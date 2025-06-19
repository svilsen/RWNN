#####################################################################################
####################### Reduce RWNN and ERWNN neural networks #######################
#####################################################################################

#### Reducing output-layer
##
reduce_network_output <- function(object, tolerance) {
    ##
    if (is.null(tolerance)) {
        tolerance <- 1e-8
    }
    
    ## Identifying zeroes
    zero_index <- which(abs(object$weights$beta) < tolerance)
    if (length(zero_index) > 0) {
        # Removing output bias
        if (object$bias$beta) {
            if (zero_index[1] == 1) {
                object$bias$beta <- FALSE
                object$weights$beta <- object$weights$beta[-1, , drop = FALSE]
                zero_index <- zero_index[-1] - 1    
            }
        }
        
        # Silencing input features 
        k <- as.numeric(object$bias$beta) + object$combined$X * (nrow(object$weights$W[[1]]) - object$bias$W[1])
        zero_index <- zero_index[!(zero_index <= k)]
        
        # Removing weights from hidden layers
        removal_index <- zero_index
        if (length(removal_index) > 0) {
            W <- length(object$weights$W)
            if (object$combined$W) {
                for (w in seq_len(W)) {
                    k <- k + object$n_hidden[w]
                    removal_index_w <- zero_index[zero_index <= k] - (k - object$n_hidden[w])
                    
                    if (length(removal_index_w) > 0) {
                        if (ncol(object$weights$W[[w]]) == 1) {
                            removal_index <- removal_index[-which(zero_index <= k)]
                            zero_index <- zero_index[!(zero_index <= k)]
                        }
                        else {
                            object$weights$W[[w]] <- object$weights$W[[w]][, -removal_index_w, drop = FALSE]
                            
                            if (w < W) {
                                object$weights$W[[w + 1]] <- object$weights$W[[w + 1]][-(removal_index_w + object$bias$W[w]), , drop = FALSE]
                            }
                            
                            object$n_hidden[w] <- ncol(object$weights$W[[w]])
                            zero_index <- zero_index[!(zero_index <= k)]
                        }
                    }
                }
            } 
            else {
                object$weights$W[[W]] <- object$weights$W[[W]][, -(removal_index - k), drop = FALSE]
                object$n_hidden[W] <- ncol(object$weights$W[[W]])
            }
            
            object$weights$beta <- object$weights$beta[-removal_index, , drop = FALSE]
        }
    }
    
    return(object)
}

##
reduce_network_hidden <- function(object, tolerance) {
    ##
    if (is.null(tolerance)) {
        tolerance <- 1e-8
    }
    
    ##
    if (!object$combined$W) {
        w_total <- length(object$weights$W)
        for (w in seq_len(w_total)) {
            ## 
            if (object$bias$W[w]) {
                if (sum(abs(object$weights$W[[w]][1, ])) < tolerance) {
                    object$weights$W[[w]] <- object$weights$W[[w]][-1, , drop = FALSE]
                    object$bias$W[w] <- FALSE
                }
            }
            
            ##
            zero_rows <- which(apply(abs(object$weights$W[[w]]), 1, sum) < tolerance)
            if (length(zero_rows) > 0) {
                if (w > 1) {
                    ##
                    object$weights$W[[w]] <- object$weights$W[[w]][-zero_rows, , drop = FALSE]
                    
                    ##
                    object$weights$W[[w - 1]] <- object$weights$W[[w - 1]][, -(zero_rows - object$bias$W[w]), drop = FALSE]
                    object$n_hidden[w - 1] <- object$n_hidden[w - 1] - length(zero_rows)
                }
                else {
                    ## IMPLEMENT REMOVAL FROM TRAINING DATA AND FORMULA?
                }
            }
            
            ##
            zero_cols <- which(apply(abs(object$weights$W[[w]]), 2, sum) < tolerance)
            if (length(zero_cols) > 0) {
                ##
                object$weights$W[[w]] <- object$weights$W[[w]][, -zero_cols, drop = FALSE]
                object$n_hidden[w] <- object$n_hidden[w] - length(zero_cols)
                
                if (w < w_total) {
                    ##
                    object$weights$W[[w + 1]] <- object$weights$W[[w + 1]][-(zero_cols + object$bias$W[w + 1]), , drop = FALSE]
                }
                else {
                    ##
                    object$weights$beta <- object$weights$beta[-(zero_cols + object$bias$beta + object$combined$X * (nrow(object$weights$W[[1]]) - object$bias$W[1])), , drop = FALSE]
                }
            }
        }
    }
    
    return(object)
}


#### Reducing the number of weights
##
reduce_network_global <- function(object, p) {
    if (is.null(p) | !is.numeric(p)) {
        warning("'p' is set to '0.1' as it was either 'NULL', or not 'numeric'.")
        p <- 0.1
    }
    else if (p < 0) {
        warning("'p' is set to '0.01', because it was found to be smaller than '0'.")
        p <- 0.01
    }
    else if (p > 1) {
        warning("'p' is set to '0.99', because it was found to be larger than '1'.")
        p <- 0.99
    }
    
    weights <- quantile(abs(unlist(object$weights)), probs = p)
    for(i in seq_along(object$weights$W)) {
        object$weights$W[[i]][abs(object$weights$W[[i]]) <= weights] <- 0
    }
    
    object$weights$beta[abs(object$weights$beta) <= weights] <- 0
    
    return(object)
}

##
reduce_network_uniform <- function(object, p) {
    #
    if (is.null(p) | !is.numeric(p)) {
        warning("'p' is set to '0.1' as it was either 'NULL', or not 'numeric'.")
        p <- 0.1
    }
    else if (p < 0) {
        warning("'p' is set to '0.01', because it was found to be smaller than '0'.")
        p <- 0.01
    }
    else if (p > 1) {
        warning("'p' is set to '0.99', because it was found to be larger than '1'.")
        p <- 0.99
    }
    
    #  
    for(i in seq_along(object$weights$W)) {
        weights_i <- quantile(abs(unlist(object$weights$W[[i]])), probs = p)
        object$weights$W[[i]][abs(object$weights$W[[i]]) <= weights_i] <- 0
    }
    
    weights_o <- quantile(abs(unlist(object$weights$beta)), probs = p)
    object$weights$beta[abs(object$weights$beta) <= weights_o] <- 0
    
    #
    return(object)
}

##
reduce_network_lamp <- function(object, p) {
    #
    if (is.null(p) | !is.numeric(p)) {
        warning("'p' is set to '0.1' as it was either 'NULL', or not 'numeric'.")
        p <- 0.1
    }
    else if (p < 0) {
        warning("'p' is set to '0.01', because it was found to be smaller than '0'.")
        p <- 0.01
    }
    else if (p > 1) {
        warning("'p' is set to '0.99', because it was found to be larger than '1'.")
        p <- 0.99
    }
    
    # 
    lamp <- vector("list", length(object$weights$W) + 1)
    for(i in seq_along(lamp)[-length(lamp)]) {
        w_i <- object$weights$W[[i]]
        w_i_sq <- w_i^2
        o_i <- order(w_i_sq)
        w_i_sq <- w_i_sq[o_i] / rev(cumsum(rev(w_i_sq[o_i])))
        w_i[o_i] <- w_i_sq
        lamp[[i]] <- w_i
    }
    
    w_o <- object$weights$beta
    w_o_sq <- w_o^2
    o_o <- order(w_o_sq)
    w_o_sq <- w_o_sq[o_o] / rev(cumsum(rev(w_o_sq[o_o])))
    w_o[o_o] <- w_o_sq
    lamp[[length(lamp)]] <- w_o
    
    # 
    weights <- quantile(abs(unlist(lamp)), probs = p)
    for(i in seq_along(object$weights$W)) {
        object$weights$W[[i]][abs(lamp[[i]]) <= weights] <- 0
    }
    
    object$weights$beta[abs(lamp[[length(lamp)]]) <= weights] <- 0
    
    #
    return(object)
}

#### Reducing the number of neurons
##
reduce_network_apoz <- function(object, p, tolerance, X, type) {
    #
    if (is.null(type)) {
        type <- "uniform"
    }
    else {
        type <- tolower(type)
    }
    
    if (!(type %in% c("glbl", "global", "unif", "uniform"))) {
        stop("'type' should be either 'global' or 'uniform'.")
    }
    
    #
    if (is.null(p) | !is.numeric(p)) {
        warning("'p' is set to '0.1' as it was either 'NULL', or not 'numeric'.")
        p <- 0.1
    }
    else if (p < 0) {
        warning("'p' is set to '0.01', because it was found to be smaller than '0'.")
        p <- 0.01
    }
    else if (p > 1) {
        warning("'p' is set to '0.99', because it was found to be larger than '1'.")
        p <- 0.99
    }
    
    #
    if (is.null(tolerance) | !is.numeric(tolerance)) {
        warning("'tolerance' is set to '1e-8' as it was either 'NULL', or not 'numeric'.")
        tolerance <- 1e-8
    }
    else if (tolerance < 0) {
        warning("'tolerance' is set to '1e-8', because it was found to be smaller than '0'.")
        tolerance <- 1e-8
    }
    
    #
    H <- rwnn_forward(X, object$weights$W, object$activation, object$bias$W)
    APOZ <- vector("list", length(H))
    for (i in seq_along(APOZ)) {
        H[[i]] <- matrix(H[[i]], ncol = object$n_hidden[i])
        APOZ[[i]] <- apply(abs((H[[i]] - mean(H[[i]])) / sd(H[[i]])) < tolerance, 2, mean)
    }
    
    #
    if (type %in% c("glbl", "global")) {
        K <- sapply(APOZ, length)
        K_s <- cumsum(c(0, K)) + 1
        
        gobal_removal <- order(unlist(APOZ))[-seq_len(round((1 - p) * (K_s[length(K_s)] - 1)))]
        remove_cols <- lapply(seq_along(K_s[-length(K_s)]), function(i) gobal_removal[which((gobal_removal >= K_s[i]) & (gobal_removal < K_s[i + 1]))] - (K_s[i] - 1))
    }
    else if (type %in% c("unif", "uniform")) {
        remove_cols <- lapply(APOZ, function(x) order(x)[-seq_len(round((1 - p) * length(x)))])
    }
    
    #
    W <- length(object$weights$W)
    for (w in seq_len(W)) {
        if (ncol(object$weights$W[[w]]) == 1) {
            next
        }
        
        ##
        remove_cols_w <- remove_cols[[w]]
        object$weights$W[[w]] <- object$weights$W[[w]][, -remove_cols_w, drop = FALSE]
        object$n_hidden[w] <- ncol(object$weights$W[[w]])
        
        ##
        if (w < W) {    
            remove_rows_w <- remove_cols_w + as.numeric(object$bias$W[w + 1])
            object$weights$W[[w + 1]] <- object$weights$W[[w + 1]][-remove_rows_w, , drop = FALSE]
        }
        
        if (object$combined$W | (w == W)) {
            index_offset <- object$bias$beta + p * object$combined$X
            if (w > 1) {
                previous_w <- sapply(object$weights$W[seq_len(w - 1)], function(x) dim(x)[2])
                index_offset <- index_offset + sum(previous_w)
            } 
            
            remove_rows_out_w <- remove_cols_w + index_offset
            object$weights$beta <- object$weights$beta[-remove_rows_out_w, , drop = FALSE]
        }
    }
    
    #
    return(object)
}

##
reduce_network_l2  <- function(object, p, X, type) {
    #
    if (is.null(type)) {
        type <- "uniform"
    }
    else {
        type <- tolower(type)
    }
    
    if (!(type %in% c("glbl", "global", "unif", "uniform"))) {
        stop("'type' should be either 'global' or 'uniform'.")
    }
    
    #
    if (is.null(p) | !is.numeric(p)) {
        warning("'p' is set to '0.1' as it was either 'NULL', or not 'numeric'.")
        p <- 0.1
    }
    else if (p < 0) {
        warning("'p' is set to '0.01', because it was found to be smaller than '0'.")
        p <- 0.01
    }
    else if (p > 1) {
        warning("'p' is set to '0.99', because it was found to be larger than '1'.")
        p <- 0.99
    }
    
    #
    H <- rwnn_forward(X, object$weights$W, object$activation, object$bias$W)
    H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = object$n_hidden[i]))
    
    Z <- lapply(seq_along(H), function(i) (H[[i]] - mean(H[[i]])) / sd(H[[i]]))
    L <- lapply(seq_along(Z), function(i) apply(Z[[i]], 2, function(x) sqrt(sum(x^2))))
    
    #
    if (type %in% c("glbl", "global")) {
        K <- sapply(L, length)
        K_s <- cumsum(c(0, K)) + 1
        
        gobal_removal <- order(unlist(L))[-seq_len(round((1 - p) * (K_s[length(K_s)] - 1)))]
        remove_cols <- lapply(seq_along(K_s[-length(K_s)]), function(i) gobal_removal[which((gobal_removal >= K_s[i]) & (gobal_removal < K_s[i + 1]))] - (K_s[i] - 1))
    }
    else if (type %in% c("unif", "uniform")) {
        remove_cols <- lapply(L, function(x) order(x)[-seq_len(round((1 - p) * length(x)))])
    }
    
    #
    W <- length(object$weights$W)
    for (w in seq_len(W)) {
        if (ncol(object$weights$W[[w]]) == 1) {
            next
        }
        
        ##
        remove_cols_w <- remove_cols[[w]]
        object$weights$W[[w]] <- object$weights$W[[w]][, -remove_cols_w, drop = FALSE]
        object$n_hidden[w] <- ncol(object$weights$W[[w]])
        
        ##
        if (w < W) {    
            remove_rows_w <- remove_cols_w + as.numeric(object$bias$W[w + 1])
            object$weights$W[[w + 1]] <- object$weights$W[[w + 1]][-remove_rows_w, , drop = FALSE]
        }
        
        if (object$combined$W | (w == W)) {
            index_offset <- object$bias$beta + p * object$combined$X
            if (w > 1) {
                previous_w <- sapply(object$weights$W[seq_len(w - 1)], function(x) dim(x)[2])
                index_offset <- index_offset + sum(previous_w)
            } 
            
            remove_rows_out_w <- remove_cols_w + index_offset
            object$weights$beta <- object$weights$beta[-remove_rows_out_w, , drop = FALSE]
        }
    }
    
    #
    return(object)
}

##
reduce_network_correlation <- function(object, type, rho, X) {
    if (is.null(type)) {
        type <- "pearson"
    } else {
        type <- tolower(type)
    }
    
    if (is.null(rho) | !is.numeric(rho)) {
        warning("'rho' is set to '0.99' as it was either 'NULL', or not 'numeric'.")
        rho <- 0.99
    } else if (rho < 0) {
        warning("'rho' is set to '0.01', because it was found to be smaller than '0'.")
        rho <- 0.01
    } else if (rho > 1) {
        warning("'rho' is set to '0.99', because it was found to be larger than '1'.")
        rho <- 0.99
    }
    
    p <- dim(object$weights$W[[1]])[1] - object$bias$W[1]
    W <- length(object$weights$W)
    for (w in seq_len(W)) {
        if (ncol(object$weights$W[[w]]) == 1) {
            next
        }
        
        ##
        H_w <- rwnn_forward(X, object$weights$W[seq_len(w)], object$activation[seq_len(w)], object$bias$W[seq_len(w)])
        H_w <- lapply(seq_along(H_w), function(i) matrix(H_w[[i]], ncol = object$n_hidden[i]))
        H_w <- H_w[[w]]
        
        ##
        C_w <- cor(H_w, method = type)
        C_w <- upper.tri(C_w) * C_w
        C_w <- abs(C_w)
        
        remove_cols_w <- which(apply(C_w >= rho, 2, any))
        
        ##
        if (length(remove_cols_w) > 0) {
            ##
            object$weights$W[[w]] <- object$weights$W[[w]][, -remove_cols_w, drop = FALSE]
            object$n_hidden[w] <- ncol(object$weights$W[[w]])
            
            ##
            if (w < W) {    
                remove_rows_w <- remove_cols_w + as.numeric(object$bias$W[w + 1])
                object$weights$W[[w + 1]] <- object$weights$W[[w + 1]][-remove_rows_w, , drop = FALSE]
            }
            
            if (object$combined$W | (w == W)) {
                index_offset <- object$bias$beta + p * object$combined$X
                if (w > 1) {
                    previous_w <- sapply(object$weights$W[seq_len(w - 1)], function(x) dim(x)[2])
                    index_offset <- index_offset + sum(previous_w)
                } 
                
                remove_rows_out_w <- remove_cols_w + index_offset
                object$weights$beta <- object$weights$beta[-remove_rows_out_w, , drop = FALSE]
            }
        }
    }
    
    return(object)
}

##
reduce_network_correlation_ft <- function(object, type, rho, alpha, X) { 
    if (is.null(type)) {
        type <- "pearson"
    }
    else {
        type <- tolower(type)
    }
    
    if (is.null(rho) | !is.numeric(rho)) {
        warning("'rho' is set to '0.99' as it was either 'NULL', or not 'numeric'.")
        rho <- 0.99
    } else if (rho < 0) {
        warning("'rho' is set to '0', because it was found to be smaller than '0'.")
        rho <- 0.0
    } else if (rho > 1) {
        warning("'rho' is set to '1', because it was found to be larger than '1'.")
        rho <- 1.0
    }
    
    if (is.null(alpha) | !is.numeric(alpha)) {
        warning("'alpha' is set to '0.05' as it was either 'NULL', or not 'numeric'.")
        alpha <- 0.05
    } else if (alpha < 0) {
        warning("'alpha' is set to '0.01', because it was found to be smaller than '0'.")
        alpha <- 0.01
    } else if (alpha > 1) {
        warning("'alpha' is set to '0.99', because it was found to be larger than '1'.")
        alpha <- 0.99
    }
    
    p <- dim(object$weights$W[[1]])[1] - object$bias$W[1]
    N <- dim(X)[1]
    
    W <- length(object$weights$W)
    for (w in seq_len(W)) {
        if (ncol(object$weights$W[[w]]) == 1) {
            next
        }
        
        ##
        H_w <- rwnn_forward(X, object$weights$W[seq_len(w)], object$activation[seq_len(w)], object$bias$W[seq_len(w)])
        H_w <- lapply(seq_along(H_w), function(i) matrix(H_w[[i]], ncol = object$n_hidden[i]))
        H_w <- H_w[[w]]
        
        ##
        C_w <- cor(H_w, method = type)
        C_w <- upper.tri(C_w) * C_w
        
        Z_w <- 0.5 * (log(1 + C_w) - log(1 - C_w))
        R_w <- 0.5 * (log(1 + rho) - log(1 - rho))
        
        ## 
        T_w <- (abs(Z_w) - R_w) * sqrt(N - 3)
        P_w <- upper.tri(T_w) * pnorm(T_w, 0, 1, lower.tail = FALSE)
        
        ## 
        N_w <- 0.5 * ncol(P_w) * (ncol(P_w) - 1)
        O_w <- P_w[order(P_w)][-seq_len(length(P_w) - N_w)]
        B_w <- alpha * seq_len(N_w) / N_w
        
        A_w <- B_w[which(O_w > B_w)[1]]
        if (length(A_w) < 1) {
            A_w <- 1.0
        }
        
        remove_cols_w <- which(apply(upper.tri(P_w) * (P_w < A_w), 2, any))
        
        ##
        if (length(remove_cols_w) > 0) {
            ##
            object$weights$W[[w]] <- object$weights$W[[w]][, -remove_cols_w, drop = FALSE]
            object$n_hidden[w] <- ncol(object$weights$W[[w]])
            
            ##
            if (w < W) {    
                remove_rows_w <- remove_cols_w + as.numeric(object$bias$W[w + 1])
                object$weights$W[[w + 1]] <- object$weights$W[[w + 1]][-remove_rows_w, , drop = FALSE]
            }
            
            if (object$combined$W | (w == W)) {
                index_offset <- object$bias$beta + p * object$combined$X
                if (w > 1) {
                    previous_w <- sapply(object$weights$W[seq_len(w - 1)], function(x) dim(x)[2])
                    index_offset <- index_offset + sum(previous_w)
                } 
                
                remove_rows_out_w <- remove_cols_w + index_offset
                object$weights$beta <- object$weights$beta[-remove_rows_out_w, , drop = FALSE]
            }
        }
    }
    
    return(object)
}

#### Reducing the number of weights and/or the number of neurons
##
reduce_network_relief <- function(object, p, X, type) {
    # 
    if (is.null(type)) {
        type <- "neuron"
    } else {
        type <- tolower(type)
    }
    
    if (!(type %in% c("w", "weight", "n", "neuron"))) {
        stop("'type' should be either 'weight' or 'neuron'.")
    }
    
    #
    if (is.null(p) | !is.numeric(p)) {
        warning("'p' is set to '0.1' as it was either 'NULL', or not 'numeric'.")
        p <- 0.1
    } else if (p < 0) {
        warning("'p' is set to '0.01', because it was found to be smaller than '0'.")
        p <- 0.01
    } else if (p > 1) {
        warning("'p' is set to '0.99', because it was found to be larger than '1'.")
        p <- 0.99
    }
    
    #
    k <- ncol(X)
    
    #
    H <- rwnn_forward(X, object$weights$W, object$activation, object$bias$W)
    H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = object$n_hidden[i]))
    
    #
    C <- append(list(X), H)
    W <- append(object$weights$W, list(object$weights$beta))
    B <- c(object$bias$W, object$bias$beta)
    
    #
    for (w in seq_along(C)) {
        # 
        if (w < length(C)) {
            C_w <- C[[w]]
        } else {
            if (object$combined$W){
                C_w <- do.call("cbind", H)
            }
            else {
                C_w <- H[[length(H)]]
            }
            
            if (object$combined$X) {
                C_w <- cbind(X, C_w)
            }
        }
        
        #
        W_w <- W[[w]]
        
        #
        if (B[w]) {
            b_w <- W_w[1, , drop = FALSE]
            W_w <- W_w[-1, , drop = FALSE]
        } else {
            b_w <- matrix(0, nrow = 1, ncol = ncol(W_w))
        }
        
        #
        I_w <- importance_score(C_w, W_w)
        N_w <- matrix(apply(I_w, 2, sum) + abs(b_w), nrow = 1)
        
        S_w <- I_w / matrix(N_w, nrow = nrow(I_w), ncol = ncol(I_w), byrow = TRUE)
        
        if (B[w]) {
            B_w <- abs(b_w) / N_w
            S_w <- rbind(B_w, S_w)
        }
        
        # 
        if (type %in% c("w", "weight")) {
            R_w <- quantile(S_w, probs = p)
            if (w < length(C)) {
                object$weights$W[[w]][S_w <= R_w] <- 0
            } 
            else {
                object$weights$beta[S_w <= R_w] <- 0
            }
        }
        else if (type %in% c("n", "neuron")) {
            #
            if (w == length(C)) {
                next
            }
            
            if (ncol(object$weights$W[[w]]) == 1) {
                next
            }
            
            N_w <- N_w / sum(N_w) 
            R_w <- quantile(N_w, probs = p)
            
            remove_cols_w <- which(N_w < R_w)
            object$weights$W[[w]] <- object$weights$W[[w]][, -remove_cols_w, drop = FALSE]
            object$n_hidden[w] <- ncol(object$weights$W[[w]])
            
            #
            if (w < (length(C) - 1)) {    
                remove_rows_w <- remove_cols_w + as.numeric(object$bias$W[w + 1])
                object$weights$W[[w + 1]] <- object$weights$W[[w + 1]][-remove_rows_w, , drop = FALSE]
            }
            
            if (object$combined$W | (w == (length(C) - 1))) {
                index_offset <- object$bias$beta + k * object$combined$X
                if (w > 1) {
                    previous_w <- sapply(object$weights$W[seq_len(w - 1)], function(x) dim(x)[2])
                    index_offset <- index_offset + sum(previous_w)
                } 
                
                remove_rows_out_w <- remove_cols_w + index_offset
                object$weights$beta <- object$weights$beta[-remove_rows_out_w, , drop = FALSE]
            }
        }
    }
    
    #
    return(object)
}

#### Reduce the number of models in an ensemble stack
##
reduce_network_stack <- function(object, tolerance) {
    #
    if (is.null(tolerance) | !is.numeric(tolerance)) {
        warning("'tolerance' is set to '1e-8' as it was either 'NULL', or not 'numeric'.")
        tolerance <- 1e-8
    }
    else if (tolerance < 0) {
        warning("'tolerance' is set to '1e-8', because it was found to be smaller than '0'.")
        tolerance <- 1e-8
    }
    
    #
    remove_models <- which(object$weights < tolerance)
    if (length(remove_models) == length(object$weights)) {
        stop("Because of the chosen tolerance all models were removed; the tolerance should be lowered to a more appropriate level.")
    }
    
    #
    object$models <- object$models[-remove_models]
    
    #
    object$weights <- object$weights[-remove_models]
    object$weights <- object$weights / sum(object$weights)
    
    #
    return(object)
}

####
#' @title Reduce the weights of a random weight neural network.
#' 
#' @description Methods for weight and neuron pruning in random weight neural networks.
#' 
#' @param object An \link{RWNN-object} or \link{ERWNN-object}.
#' @param method A string, or a function, setting the method used to reduce the network (see details).
#' @param retrain TRUE/FALSE: Should the output weights be retrained after reduction?
#' @param ... Additional arguments passed to the reduction method (see details).
#' 
#' @details The '\code{method}' and additional arguments required by the method are:
#' \describe{
#'   \item{\code{"global"} (or \code{"glbl"})}{\describe{
#'      \item{\code{p}: The proportion of weights to remove globally based on magnitude.}{}
#'   }}
#'   \item{\code{"uniform"} (or \code{"unif"})}{\describe{
#'      \item{\code{p}: The proportion of weights to remove uniformly layer-by-layer based on magnitude.}{}
#'   }}
#'   \item{\code{"lamp"}}{\describe{
#'      \item{\code{p}: The proportion of weights to remove based on LAMP scores.}{}
#'   }}
#'   \item{\code{"apoz"}}{\describe{
#'      \item{\code{p}: The proportion of neurons to remove based on proportion of zeroes produced.}{}
#'      \item{\code{tolerance}: The tolerance used when identifying zeroes.}{}
#'      \item{\code{type}: A string indicating whether weights should be removed globally (\code{'global'}) or uniformly  (\code{'uniform'}).}{}
#'   }}
#'   \item{\code{"correlation"} (or \code{"cor"})}{\describe{
#'      \item{\code{type}: The type of correlation (argument passed to \link{cor} function).}{}
#'      \item{\code{rho}: The correlation threshold used to remove neurons.}{}
#'   }}
#'   \item{\code{"correlationtest"} (or \code{"cortest"})}{\describe{
#'      \item{\code{type}: The type of correlation (argument passed to \link{cor} function).}{}
#'      \item{\code{rho}: The correlation threshold used to remove neurons.}{}
#'      \item{\code{alpha}: The significance levels used to test whether the observed correlation between two neurons is small than \code{rho}.}{}
#'   }}
#'   \item{\code{"relief"}}{\describe{
#'      \item{\code{p}: The proportion of neurons or weights to remove based on relief scores.}{}
#'      \item{\code{type}: A string indicating whether neurons (\code{'neuron'}) or weights (\code{'weight'}) should be removed.}{}
#'   }}
#'   \item{\code{"hidden"}}{\describe{
#'      \item{\code{tolerance}: The tolerance used when removing zeroes from the hidden layer(s).}{}
#'   }}
#'   \item{\code{"output"}}{\describe{
#'      \item{\code{tolerance}: The tolerance used when removing zeroes from the output layer.}{}
#'   }}
#' } 
#' 
#' The argument \code{retrain} should primarily be used when \code{method = "output"}, in this case it defaults to \code{TRUE}, but will default to \code{FALSE} otherwise.
#' 
#' If the object is an \link{ERWNN-object}, the reduction is applied to all \link{RWNN-object}'s in the \link{ERWNN-object}. Furthermore, when
#' the \link{ERWNN-object} is created as a stack and the weights of the stack is trained, then '\code{method}' can be set to:
#' \describe{
#' \item{\code{"stack"}}{\describe{
#'      \item{\code{tolerance}: The tolerance used when removing elements from the stack.}{}
#'   }}
#' }
#' 
#' Lastly, '\code{method}' can also be passed as a function, with additional arguments passed through the \code{...} argument. 
#' NB: features and target are passed using the names \code{X} and \code{y}, respectively.
#' 
#' @return A reduced \link{RWNN-object} or \link{ERWNN-object}.
#' 
#' @references Han S., Mao H., Dally W.J. (2016) "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding." arXiv: 1510.00149.
#' 
#' Hu H., Peng R., Tai Y.W., Tang C.K. (2016) "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures." arXiv: 1607.03250.
#' 
#' Morcos A.S., Yu H., Paganini M., Tian Y. (2019) "One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers." arXiv: 1906.02773.
#' 
#' Lee J., Park S., Mo S., Ahn S., Shin J. (2021) "Layer-adaptive sparsity for the Magnitude-based Pruning." arXiv: 2010.07611.
#' 
#' Dekhovich A., Tax D.M., Sluiter M.H., Bessa M.A. (2024) "Neural network relief: a pruning algorithm based on neural activity." \emph{Machine Learning}, 113, 2597-2618.
#' 
#' @export
reduce_network <- function(object, method, retrain = NULL, ...) {
    UseMethod("reduce_network")
}

#' @rdname reduce_network
#' @method reduce_network RWNN
#'
#' @example inst/examples/reduction_example.R
#'
#' @export
reduce_network.RWNN <- function(object, method, retrain = NULL, ...) {
    ##
    dots <- list(...)
    
    ##
    if (is.null(retrain) | !is.logical(retrain)) {
        if (method %in% c("output")) {
            retrain <- TRUE
        }
        else {
            retrain <- FALSE
        }
    }
    
    if (is.null(dots[["tolerance"]])) {
        dots[["tolerance"]] <- 1e-8
    }
    
    ##
    if ((!is.null(dots[["X"]])) & (!is.null(dots[["y"]]))) {
        X <- dots[["X"]]
        y <- dots[["y"]]
    } else if (!is.null(object$data$X)) {
        X <- object$data$X
        y <- object$data$y
    } else {
        stop("Data has to be present in the model object, or supplied through '...' argument as 'X = ' and 'y = '.")
    }
    
    ##
    if (method %in% c("mag", "magnitide", "glbl", "global")) {
        ## Weight pruning method: Reducing the number of hidden-weights based on magnitude (globally).
        object <- reduce_network_global(object = object, p = dots[["p"]])
    }
    else if (method %in% c("unif", "uniform")) {
        ## Weight pruning method: Reducing the number of hidden-weights based on magnitude (uniformly layer-by-layer).
        object <- reduce_network_uniform(object = object, p = dots[["p"]])
    }
    else if (method %in% c("lamp")) {
        ## Weight pruning method: Reducing the number of hidden-weights using the LAMP scores.
        object <- reduce_network_lamp(object = object, p = dots[["p"]])
    }
    else if (method %in% c("apoz")) {
        ## Neuron pruning method: Average percentage of "zeros".
        if (!all(object$activation == "relu")) {
            warning("APOZ was designed for 'relu' activation functions, but no 'relu' activation was found.")
        }
        
        object <- reduce_network_apoz(object = object, p = dots[["p"]], tolerance = dots[["tolerance"]], X = X, type = dots[["type"]])
    }
    else if (method %in% c("cor", "correlation")) {
        ## Neuron pruning method: Correlation between activated neurons.
        object <- reduce_network_correlation(object = object, type = dots[["type"]], rho = dots[["rho"]], X = X)
    }
    else if (method %in% c("ct", "cortest", "correlationtest")) {
        ## Neuron pruning method: Correlation between activated neurons.
        object <- reduce_network_correlation_ft(object = object, type = dots[["type"]], rho = dots[["rho"]], alpha = dots[["alpha"]], X = X)
    }
    else if (method %in% c("relief")) {
        ## Neuron and weight pruning method: Reduction based on relief scores.
        object <- reduce_network_relief(object = object, p = dots[["p"]], X = X, type = dots[["type"]])
    }
    else if (method %in% c("hidden")) {
        ## Removing '0' weight rows/columns from the hidden-layers.
        object <- reduce_network_hidden(object = object, tolerance = dots[["tolerance"]])
    }
    else if (method %in% c("output")) {
        ## Removing '0' weights from the output-layer.
        object <- reduce_network_output(object = object, tolerance = dots[["tolerance"]])
    } 
    else if (is.function(method)) {
        object_list <- list(object = object, X = X, y = y) |> append(dots)
        object <- do.call(method, object_list)
    }
    else {
        stop("'method' is either not implemented, or not a function.")
    }
    
    ## 
    w <- 1
    w_total <- length(object$weights$W)
    while (w <= w_total) {
        ## Removing biases from hidden layers
        if (object$bias$W[w]) {
            if (all(abs(object$weights$W[[w]][1, ]) < dots[["tolerance"]])) {
                object$weights$W[[w]] <- object$weights$W[[w]][-1, , drop = FALSE]
                object$bias$W[w] <- FALSE
            }
        }
        
        ## Removing layers in case of complete 'zeroing' of weights
        idx <- seq_len(nrow(object$weights$W[[w]])) + object$bias$W[w]
        idx <- idx[idx <= nrow(object$weights$W[[w]])]
        if (all(abs(object$weights$W[[w]][idx,]) < dots[["tolerance"]])) {
            if (w == 1) {
                stop("The applied 'method' have zeroed all weights in the first hidden layer.")
            }
            
            object$weights$W <- object$weights$W[seq_len(w - 1)]
            object$bias$W <- object$bias$W[seq_len(w - 1)]
            object$n_hidden <- object$n_hidden[seq_len(w - 1)]
            object$activation <- object$activation[seq_len(w - 1)]
            
            w_total <- length(object$weights$W)
            retrain <- TRUE
        }
        
        w <- w + 1
    }
    
    ## Removing bias from output layer 
    if (object$bias$beta) {
        if (abs(object$weights$beta[1]) < dots[["tolerance"]]) {
            object$weights$beta <- object$weights$beta[-1, , drop = FALSE]
            object$bias$beta <- FALSE
        }
    }
    
    ## 
    if (retrain) {
        H <- rwnn_forward(X, object$weights$W, object$activation, object$bias$W)
        H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = object$n_hidden[i]))
        
        if (object$combined$W){
            H <- do.call("cbind", H)
        } else {
            H <- H[[length(H)]]
        }
        
        O <- H
        if (object$combined$X) {
            O <- cbind(X, H)
        }
        
        if (object$bias$beta) {
            O <- cbind(1, O)
        }
        
        W <- estimate_output_weights(O, y, object$lnorm[length(object$lnorm)], object$lambda[length(object$lambda)])
        object$weights$beta <- W$beta
        object$sigma <- W$sigma
    }
    
    ##
    return(object)
}

#' @rdname reduce_network
#' @method reduce_network ERWNN
#' 
#' @export
reduce_network.ERWNN <- function(object, method, retrain = NULL, ...) {
    dots <- list(...)
    
    if (is.null(dots[["tolerance"]])) {
        dots[["tolerance"]] <- 1e-8
    }
    
    if (method %in% c("stack", "stacking")) {
        if (object$method != "stacking") {
            stop("Setting 'method' to 'stacking' is only meant for stacking ensemble models.")
        }
        
        object <- reduce_network_stack(object = object, tolerance = dots[["tolerance"]])
    }
    else {
        if ((!is.null(dots[["X"]])) & (!is.null(dots[["y"]]))) {
            X <- dots[["X"]]
            y <- dots[["y"]]
        } else if (!is.null(object$data$X)) {
            X <- object$data$X
            y <- object$data$y
        } else {
            stop("Data has to be present in the model object, or supplied through '...' argument as 'X = ' and 'y = '.")
        }
        
        B <- length(object$models)
        for (b in seq_len(B)) {
            list_b <- list(object = object$models[[b]], method = method, retrain = retrain, X = X, y = y) |> append(dots)
            object_b <- do.call(reduce_network, list_b)
            
            object$models[[b]] <- object_b
        }
    }
    
    return(object)
}

