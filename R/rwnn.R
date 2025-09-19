####################################################################
####################### RWNN neural networks #######################
####################################################################

#' @title rwnn control function
#' 
#' @description A function used to create a control-object for the \link{rwnn} function.
#' 
#' @param n_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param n_features The number of randomly chosen features in the RWNN model. Note: This is meant for use in \link{bag_rwnn}, and it is not recommended outside of that function. 
#' @param lnorm A string indicating the type of regularisation used when estimating the weights in the output layer, \code{"l1"} or \code{"l2"} (default).
#' @param bias_hidden A vector of TRUE/FALSE values. The vector should have length 1, or be equal to the number of hidden layers.
#' @param bias_output TRUE/FALSE: Should a bias be added to the output layer?
#' @param activation A vector of strings corresponding to activation functions (see details). The vector should have length 1, or be equal to the number of hidden layers.
#' @param combine_input TRUE/FALSE: Should the input be included to predict the output?
#' @param combine_hidden TRUE/FALSE: Should all hidden layers be combined to predict the output?
#' @param include_data TRUE/FALSE: Should the original data be included in the returned object? Note: this should almost always be set to '\code{TRUE}', but using '\code{FALSE}' is more memory efficient in \link{ERWNN-object}'s.
#' @param include_estimate TRUE/FALSE: Should the \code{rwnn}-function estimate the output parameters? Note: this should almost always be set to '\code{TRUE}', but using '\code{FALSE}'is more memory efficient in \link{ERWNN-object}'s.
#' @param rng A string indicating the sampling distribution used for generating the weights of the hidden layer (defaults to \code{runif}). 
#' @param rng_pars A list of parameters passed to the \code{rng} function (defaults to \code{list(min = -1, max = 1)}).   
#' 
#' @details The possible activation functions supplied to '\code{activation}' are:
#' \describe{
#'     \item{\code{"identity"}}{\deqn{f(x) = x}}
#'     \item{\code{"bentidentity"}}{\deqn{f(x) = \frac{\sqrt{x^2 + 1} - 1}{2} + x}}
#'     \item{\code{"sigmoid"}}{\deqn{f(x) = \frac{1}{1 + \exp(-x)}}}
#'     \item{\code{"tanh"}}{\deqn{f(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}}}
#'     \item{\code{"sin"}}{\deqn{f(x) = sin(x)}}
#'     \item{\code{"relu"}}{\deqn{f(x) = \max\{0, x\}}}
#'     \item{\code{"silu"} (default)}{\deqn{f(x) = \frac{x}{1 + \exp(-x)}}}
#'     \item{\code{"softplus"}}{\deqn{f(x) = \ln(1 + \exp(x))}}
#'     \item{\code{"softsign"}}{\deqn{f(x) = \frac{x}{1 + |x|}}}
#'     \item{\code{"sqnl"}}{\deqn{f(x) = -1\text{, if }x < -2\text{, }f(x) = x + \frac{x^2}{4}\text{, if }-2 \le x < 0\text{, }f(x) = x - \frac{x^2}{4}\text{, if }0 \le x \le 2\text{, and } f(x) = 2\text{, if }x > 2}}
#'     \item{\code{"gaussian"}}{\deqn{f(x) = \exp(-x^2)}}
#'     \item{\code{"sqrbf"}}{\deqn{f(x) = 1 - \frac{x^2}{2}\text{, if }|x| \le 1\text{, }f(x) = \frac{(2 - |x|)^2}{2}\text{, if }1 < |x| < 2\text{, and }f(x) = 0\text{, if }|x| \ge 2}}
#' }
#' 
#' If '\code{boost_schedule}' is set to '\code{TRUE}' (default), then the weights applied in a gradient boosting ensemble defined as: 
#' \deqn{
#'  w_b = \frac{\varepsilon}{b}, \text{ for } b = 1, ..., B
#' }  
#' where \eqn{\varepsilon} is the learning rate, and \eqn{B} is the total number of models in the ensemble.
#' 
#' The '\code{rng}' argument can also be set to \code{"orthogonal"}, \code{"torus"}, \code{"halton"}, or \code{"sobol"} for added stability. The \code{"torus"}, \code{"halton"}, and \code{"sobol"} are quasi-random number generators, relying on \link[randtoolbox]{torus}, \link[randtoolbox]{halton}, and \link[randtoolbox]{sobol} functions from the \link{randtoolbox} package. NB: this is not recommended when creating ensembles. 
#' 
#' @return A list of control variables.
#' 
#' @references Wang W., Liu X. (2017) "The selection of input weights of extreme learning machine: A sample structure preserving point of view." \emph{Neurocomputing}, 261, 28-36.
#' 
#' @export
control_rwnn <- function(n_hidden = NULL, n_features = NULL, lnorm = NULL,
                         bias_hidden = TRUE, bias_output = TRUE, activation = NULL, 
                         combine_input = FALSE, combine_hidden = FALSE, 
                         include_data = TRUE, include_estimate = TRUE,
                         rng = runif, rng_pars = list(min = -1, max = 1)) {
    #
    if (is.null(lnorm) | !is.character(lnorm)) {
        lnorm <- "l2"
    }
    
    #
    lnorm <- tolower(lnorm)
    if (!(lnorm %in% c("l1", "l2"))) {
        stop("'lnorm' has to be either 'l1' or 'l2'.")
    }
    
    #
    if (length(bias_hidden) == 1) {
        bias_hidden <- rep(bias_hidden, length(n_hidden))
    } 
    else if (length(bias_hidden) == length(n_hidden)) {
        bias_hidden <- bias_hidden
    } 
    else {
        stop("The 'bias_hidden' vector specified in the control-object should have length 1, or be the same length as the vector 'n_hidden'.")
    }
    
    #
    if (!is.logical(bias_output)) {
        stop("'bias_output' has to be 'TRUE'/'FALSE'.")
    }
    
    #
    if (!is.logical(combine_input)) {
        stop("'combine_input' has to be 'TRUE'/'FALSE'.")
    }
    
    #
    if (!is.logical(combine_hidden)) {
        stop("'combine_hidden' has to be 'TRUE'/'FALSE'.")
    }
    
    #
    if (!is.logical(include_data)) {
        stop("'include_data' has to be 'TRUE'/'FALSE'.")
    }
    
    #
    if (!is.logical(include_estimate)) {
        stop("'include_estimate' has to be 'TRUE'/'FALSE'.")
    }
    
    #
    if (!is.logical(boost_schedule)) {
        stop("'boost_schedule' has to be 'TRUE'/'FALSE'.")
    }
    
    #
    if (is.null(activation) | !is.character(activation)) {
        activation <- "silu"
    }
    
    if (all(!(activation %in% c("sigmoid", "tanh", "sin", "relu", "silu", "softplus", "softsign", "sqnl", "gaussian", "sqrbf", "bentidentity", "identity")))) {
        stop("Invalid activation function detected in 'activation' vector. The implemented activation functions are: 'sigmoid', 'tanh', 'sin', 'relu', 'silu', 'softplus', 'softsign', 'sqnl', 'gaussian', 'sqrbf', 'bentidentity', and 'identity'.")
    }
    
    if (length(activation) == 1) {
        activation <- rep(tolower(activation), length(n_hidden))
    } 
    else if (length(activation) == length(n_hidden)) {
        activation <- tolower(activation)
    } 
    else {
        stop("The 'activation' vector specified in the control-object should have length 1, or be the same length as the vector 'n_hidden'.")
    }
    
    if (length(activation) < 1) {
        activation <- NULL
    }
    
    #
    if (is.character(rng)) {
        rng <- tolower(rng)
        if (rng %in% c("o", "orto", "orthogonal", "h", "halt", "halton", "s", "sobo", "sobol", "tor", "torus")) {
            rng_arg <- c("min", "max")
        }
        else {
            rng_arg <- formalArgs(rng)[-which(formalArgs(rng) == "n")]
        }
    }
    else {
        rng_arg <- formalArgs(rng)[-which(formalArgs(rng) == "n")]
    }
    
    if (!all(rng_arg %in% names(rng_pars))) {
        stop(paste("The following arguments were not found in 'rng_pars' list:", paste(rng_arg[!(rng_arg %in% names(rng_pars))], collapse = ", ")))
    }
    
    #
    return(
        list(
            n_hidden = n_hidden, n_features = n_features, lnorm = lnorm, 
            bias_hidden = bias_hidden, bias_output = bias_output, activation = activation, 
            combine_input = combine_input, combine_hidden = combine_hidden, 
            include_data = include_data, include_estimate = include_estimate, 
            rng = rng, rng_pars = rng_pars
        )
    )
}

#' @title Random weight neural networks
#' 
#' @description Set-up and estimate weights of a random weight neural network.
#' 
#' @param formula A \link{formula} specifying features and targets used to estimate the parameters of the output layer. 
#' @param data A data-set (either a \link{data.frame} or a \link[tibble]{tibble}) used to estimate the parameters of the output layer.
#' @param n_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant used when training the output layer.
#' @param type A string indicating whether this is a regression or classification problem. 
#' @param control A list of additional arguments passed to the \link{control_rwnn} function.
#' 
#' @details A deep RWNN is constructed by increasing the number of elements in the vector \code{n_hidden}. Furthermore, if \code{type} is null, then the function tries to deduce it from class of target.
#' 
#' @return An \link{RWNN-object}.
#' 
#' @references Schmidt W., Kraaijveld M., Duin R. (1992) "Feedforward neural networks with random weights." \emph{In Proceedings., 11th IAPR International Conference on Pattern Recognition. Vol.II. Conference B: Pattern Recognition Methodology and Systems}, 1–4.
#' 
#' Pao Y., Park G., Sobajic D. (1992) "Learning and generalization characteristics of random vector Functional-link net." \emph{Neurocomputing}, 6, 163–180.
#' 
#' Huang G.B., Zhu Q.Y., Siew C.K. (2006) "Extreme learning machine: Theory and applications." \emph{Neurocomputing}, 70(1), 489–501.
#' 
#' Henríquez P.A., Ruz G.A. (2018) "Twitter Sentiment Classification Based on Deep Random Vector Functional Link." \emph{In 2018 International Joint Conference on Neural Networks (IJCNN)}, 1–6.
#' 
#' @export
rwnn <- function(formula, data = NULL, n_hidden = c(), lambda = 0, type = NULL, control = list()) {
    UseMethod("rwnn")
}

#
rwnn_matrix <- function(X, y, n_hidden = c(), lambda = 0, type = NULL, control = list()) {
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
    if (is.null(lambda) | !is.numeric(lambda)) {
        lambda <- 0
        warning("Note: 'lambda' was not supplied, or not numeric, and is therefore set to 0.")
    } else if (length(lambda) > 1) {
        lambda <- lambda[1]
        warning("The length of 'lambda' was larger than 1, only the first element will be used.")
    }
    
    if (lambda < 0) {
        lambda <- 0
        warning("'lambda' has to be a real number larger than or equal to 0.")
    }
    
    # Feature restriction
    if (is.null(n_features)) {
        n_features <- ncol(X)
    }
    
    if (length(n_features) > 1) {
        n_features <- n_features[1]
        warning("The length of 'n_features' was larger than 1, only the first element will be used.")
    }
    
    if ((n_features < 1) || (n_features > dim(X)[2])) {
        stop("'n_features' has to be between 1 and the total number of features.")
    }
    
    ## Creating random weights
    X_dim <- dim(X)
    W_hidden <- vector("list", length = length(n_hidden))
    for (w in seq_along(W_hidden)) {
        if (w == 1) {
            nr_rows <- (X_dim[2] + as.numeric(bias_hidden[w]))
        }
        else {
            nr_rows <- (n_hidden[w - 1] + as.numeric(bias_hidden[w]))
        }
        
        if (is.character(rng_function)) {
            if (rng_function %in% c("o", "orto", "orthogonal")) {
                random_weights <- (rng_pars$max - rng_pars$min) * random_orthonormal(w, nr_rows, X, W_hidden, n_hidden, activation, bias_hidden) + rng_pars$min
            }
            else if (rng_function %in% c("h", "halt", "halton")) {
                random_weights <- (rng_pars$max - rng_pars$min) * halton(nr_rows, n_hidden[w], init = w == 1, start = 0) + rng_pars$min
            }
            else if (rng_function %in% c("s", "sobo", "sobol")) {
                random_weights <- (rng_pars$max - rng_pars$min) * sobol(nr_rows, n_hidden[w], init = w == 1, start = 0) + rng_pars$min
            }
            else if (rng_function %in% c("tor", "torus")) {
                random_weights <- (rng_pars$max - rng_pars$min) * torus(nr_rows, n_hidden[w], init = w == 1, start = 0) + rng_pars$min
            }
            else {
                rng_pars$n <- n_hidden[w] * nr_rows
                random_weights <- matrix(do.call(rng_function, rng_pars), ncol = n_hidden[w]) 
            }
        }
        else {
            rng_pars$n <- n_hidden[w] * nr_rows
            random_weights <- matrix(do.call(rng_function, rng_pars), ncol = n_hidden[w]) 
        }
        
        W_hidden[[w]] <- random_weights
        
        if ((w == 1) && (n_features < dim(X)[2])) {
            indices_f <- sample(ncol(X), n_features, replace = FALSE) + as.numeric(bias_hidden[w])
            W_hidden[[w]][-indices_f, ] <- 0
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
        
        W_output <- estimate_output_weights(O, y, lnorm, lambda)
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
        lambda = lambda,
        bias = list(W = bias_hidden, beta = control$bias_output),
        weights = list(W = W_hidden, beta = W_output$beta),
        sigma = W_output$sigma,
        type = type,
        combined = list(X = control$combine_input, W = control$combine_hidden)
    )
    
    class(object) <- "RWNN"
    return(object)
}

#' @rdname rwnn
#' @method rwnn formula
#' 
#' @example inst/examples/rwnn_example.R
#' 
#' @export
rwnn.formula <- function(formula, data = NULL, n_hidden = c(), lambda = 0, type = NULL, control = list()) {
    # Checks for 'n_hidden'
    if (length(n_hidden) < 1) {
        stop("When the number of hidden layers is 0, or left 'NULL', the RWNN reduces to a linear model, see ?lm.")
    }
    
    if (any(!is.numeric(n_hidden))) {
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
    } 
    else if (tolower(type) %in% c("r", "reg", "regression")) {
        type <- "regression"
    }
    else {
        stop("'type' has not been correctly specified, it needs to be set to either 'regression' or 'classification'.")
    }
    
    #
    mm <- rwnn_matrix(X, y, n_hidden = n_hidden, lambda = lambda, type = type, control = control)
    mm$formula <- if (keep_formula) formula
    return(mm)
}

