##################################################################################
####################### AE pre-trained RWNN neural network #######################
##################################################################################

#' @title Auto-encoder pre-trained random weight neural networks
#' 
#' @description Set-up and estimate weights of a random weight neural network using an auto-encoder for unsupervised pre-training of the hidden weights.
#' 
#' @param formula A \link{formula} specifying features and targets used to estimate the parameters of the output-layer. 
#' @param data A data-set (either a \link{data.frame} or a \link[tibble]{tibble}) used to estimate the parameters of the output-layer.
#' @param n_hidden A vector of integers designating the number of neurons in each of the hidden-layers (the length of the list is taken as the number of hidden-layers).
#' @param lambda A vector of two penalisation constants used when encoding the hidden-weights and training the output-weights, respectively.
#' @param method The penalisation type used for the auto-encoder (either \code{"l1"} or \code{"l2"}).
#' @param type A string indicating whether this is a regression or classification problem. 
#' @param control A list of additional arguments passed to the \link{control_rwnn} function.
#' 
#' @return An \link{RWNN-object}.
#' 
#' @references Zhang Y., Wu J., Cai Z., Du B., Yu P.S. (2019) "An unsupervised parameter learning model for RVFL neural network." \emph{Neural Networks}, 112, 85-97.
#' 
#' @export
ae_rwnn <- function(formula, data = NULL, n_hidden = c(), lambda = NULL, method = "l1", type = NULL, control = list()) {
    UseMethod("ae_rwnn")
}

ae_rwnn_matrix <- function(X, y, n_hidden = c(), lambda = NULL, method = "l1", type = NULL, control = list()) {
    ## Creating control object 
    control$n_hidden <- n_hidden
    control <- do.call(control_rwnn, control)
    
    #
    lnorm <- control$lnorm
    bias_hidden <- control$bias_hidden
    activation <- control$activation
    n_features <- control$n_features
    rng_function <- control$rng
    rng_pars <- control$rng_pars
    
    ## Checks
    dc <- data_checks(y, X)
    
    # Regularisation
    if (is.null(lambda)) {
        lambda <- 0
        warning("Note: 'lambda' is set to '0', as it was not supplied.")
    } else if (any(lambda < 0)) {
        lambda <- 0
        warning("'lambda' has to be a real number larger than or equal to '0'.")
    }
    
    if (length(lambda) == 1) {
        lambda <- c(1, lambda)
    } else if (length(lambda) > 2) {
        lambda <- lambda[seq_len(2)]
        warning("The length of 'lambda' was larger than 2; only the first two elements will be used.")
    }
    
    if (is.null(n_features)) {
        n_features <- ncol(X)
    }
    
    if (length(n_features) > 1) {
        n_features <- n_features[1]
        warning("The length of 'n_features' was larger than 1; only the first element will be used.")
    }
    
    if ((n_features < 1) || (n_features > dim(X)[2])) {
        stop("'n_features' have to be between '1' and the total number of features.")
    }
    
    ## Creating random weights
    X_dim <- dim(X)
    W_hidden <- vector("list", length = length(n_hidden))
    for (w in seq_along(W_hidden)) {
        ## Generating random weights
        if (w == 1) {
            nr_rows <- (X_dim[2] + as.numeric(bias_hidden[w]))
        } else {
            nr_rows <- (n_hidden[w - 1] + as.numeric(bias_hidden[w]))
        }
        
        if (is.character(rng_function)) {
            if (rng_function %in% c("o", "orto", "orthogonal")) {
                random_weights <- (rng_pars$max - rng_pars$min) * random_orthonormal(w, nr_rows, X, W_hidden, n_hidden, activation, bias_hidden) + rng_pars$min
            }
            else if (rng_function %in% c("h", "halt", "halton")) {
                random_weights <- (rng_pars$max - rng_pars$min) * halton(nr_rows, n_hidden[w], init = w == 1) + rng_pars$min
            }
            else if (rng_function %in% c("s", "sobo", "sobol")) {
                random_weights <- (rng_pars$max - rng_pars$min) * sobol(nr_rows, n_hidden[w], init = w == 1) + rng_pars$min
            }
            else if (rng_function %in% c("tor", "torus")) {
                random_weights <- (rng_pars$max - rng_pars$min) * torus(nr_rows, n_hidden[w], init = w == 1, start = 0) + rng_pars$min
            }
            else {
                rng_pars$n <- n_hidden[w] * nr_rows
                random_weights <- matrix(do.call(rng_function, rng_pars), ncol = n_hidden[w]) 
            }
        } else {
            rng_pars$n <- n_hidden[w] * nr_rows
            random_weights <- matrix(do.call(rng_function, rng_pars), ncol = n_hidden[w]) 
        }
        
        W_hidden[[w]] <- random_weights
        
        if ((w == 1) && (n_features < dim(X)[2])) {
            indices_f <- sample(ncol(X), n_features, replace = FALSE) + as.numeric(bias_hidden[w])
            W_hidden[[w]][-indices_f, ] <- 0
        }
        
        ## Auto-encoder pre-training
        # Value of hidden-layer before pre-training
        H_tilde <- rwnn_forward(X = X, W = W_hidden[seq_len(w)], activation = activation, bias = bias_hidden[seq_len(w)])
        H_tilde <- lapply(seq_along(H_tilde), function(i) matrix(H_tilde[[i]], ncol = n_hidden[i]))
        
        if (w == 1) {
            P_tilde <- unname(X)
        } else {
            P_tilde <- H_tilde[[w - 1]]
        }
        
        if (bias_hidden[w]) {
            P_tilde <- cbind(1, P_tilde)
        }
        
        H_tilde <- H_tilde[[w]]
        
        # Pre-training of weights in hidden-layer
        if (method == "l1") {
            W_tilde <- estimate_output_weights(H_tilde, P_tilde, "l1", lambda[1])
            W_hidden[[w]] <- t(W_tilde$beta)
        } else if (method == "l2") {
            W_tilde <- estimate_output_weights(H_tilde, P_tilde, "l2", lambda[1])
            W_hidden[[w]] <- t(W_tilde$beta)
        } else {
            stop("Method not implemented; set method to either \"l1\" or \"l2\".")
        }
    }
    
    ## Values of last hidden layer
    if (control$include_estimate) {
        H <- rwnn_forward(X, W_hidden, activation, bias_hidden)
        H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = n_hidden[i]))
        
        if (control$combine_hidden){
            H <- do.call("cbind", H)
        } 
        else {
            H <- H[[length(H)]]
        }
        
        O <- H
        if (control$combine_input) {
            O <- cbind(X, H)
        }
        
        if (control$bias_output) {
            O <- cbind(1, O)
        }
        
        W_output <- estimate_output_weights(O, y, lnorm, lambda[2])
    } else {
        W_output <- list()
    }
    
    ## Return object
    object <- list(
        formula = NULL,
        data = if(control$include_data) list(X = X, y = y, C = ifelse(type == "regression", NA, colnames(y))) else NULL, 
        n_hidden = n_hidden, 
        activation = activation, 
        lnorm = lnorm,
        lambda = lambda[2],
        bias = list(W = bias_hidden, beta = control$bias_output),
        weights = list(W = W_hidden, beta = W_output$beta),
        sigma = W_output$sigma,
        type = type,
        combined = list(X = control$combine_input, W = control$combine_hidden)
    )
    
    class(object) <- "RWNN"
    return(object)
}


#' @rdname ae_rwnn
#' @method ae_rwnn formula
#' 
#' @example inst/examples/aerwnn_example.R
#' 
#' @export
ae_rwnn.formula <- function(formula, data = NULL, n_hidden = c(), lambda = NULL, method = "l1", type = NULL, control = list()) {
    # Checks for 'n_hidden'
    if (length(n_hidden) < 1) {
        stop("When the number of hidden-layers is 0, or left 'NULL', the RWNN reduces to a linear model, see ?lm.")
    }
    
    if (!is.numeric(n_hidden)) {
        stop("Not all elements of the 'n_hidden' vector were numeric.")
    }
    
    # Checks for 'data'
    keep_formula <- TRUE
    if (is.null(data)) {
        data <- tryCatch(
            expr = {
                as.data.frame(as.matrix(model.frame(formula)))
            },
            error = function(e) {
                stop("'data' needs to be supplied when using 'formula'.")
            }
        )
        
        x_name <- paste0(attr(terms(formula), "term.labels"), ".")
        colnames(data) <- paste0("V", gsub(x_name, "", colnames(data)))
        colnames(data)[1] <- "y"
        
        formula <- paste(colnames(data)[1], "~", paste(colnames(data)[seq_along(colnames(data))[-1]], collapse = " + "))
        formula <- as.formula(formula)
        keep_formula <- FALSE
    }
    
    # Checks for 'method'
    method <- tolower(method)
    if (!(method %in% c("l1", "l2"))) {
        stop("'method' has to be set to 'l1' or 'l2'.")
    }
    
    # Re-capture feature names when '.' is used in formula interface
    formula <- terms(formula, data = data)
    formula <- strip_terms(formula)
    
    #
    X <- model.matrix(formula, data)
    keep <- which(colnames(X) != "(Intercept)")
    if (any(colnames(X) == "(Intercept)")) {
        X <- X[, keep, drop = FALSE]
    }
    
    #
    y <- model.response(model.frame(formula, data))
    y <- as.matrix(y, nrow = nrow(data))
    
    #
    if (is.null(type)) {
        if (is(y[, 1], "numeric")) {
            type <- "regression"
            
            if (all(abs(y - round(y)) < 1e-8)) {
                warning("The response consists of only integers, is this a classification problem?")
            }
        }
        else if (class(y[, 1]) %in% c("factor", "character", "logical")) {
            type <- "classification"
        }
    }
    
    # Change output based on 'type'
    if (tolower(type) %in% c("c", "class", "classification")) {
        type <- "classification"
        
        y_names <- sort(unique(y))
        y <- factor(y, levels = y_names)
        y <- model.matrix(~ 0 + y)
        
        attr(y, "assign") <- NULL
        attr(y, "contrasts") <- NULL
        
        y <- 2 * y - 1
        
        colnames(y) <- paste(y_names, sep = "")
    } else if (tolower(type) %in% c("r", "reg", "regression")) {
        type <- "regression"
    } else {
        stop("'type' has not been correctly specified, it needs to be set to either 'regression' or 'classification'.")
    }
    
    #
    mm <- ae_rwnn_matrix(X, y, n_hidden = n_hidden, lambda = lambda, method = method, type = type, control = control)
    mm$formula <- if (keep_formula) formula
    return(mm)
}

