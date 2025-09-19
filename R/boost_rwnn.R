#############################################################################
####################### Boosting ERWNN neural network #######################
#############################################################################

#' @title Boosting random weight neural networks
#' 
#' @description Use gradient boosting to create ensemble random weight neural network models.
#' 
#' @param formula A \link{formula} specifying features and targets used to estimate the parameters of the output layer. 
#' @param data A data-set (either a \link{data.frame} or a \link[tibble]{tibble}) used to estimate the parameters of the output layer.
#' @param n_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant(s) passed to either \link{rwnn} or \link{ae_rwnn} (see \code{method} argument).
#' @param B The number of levels used in the boosting tree.
#' @param epsilon The learning rate.
#' @param schedule The schedule with which the weights are reduced over time. Set to \code{NULL}, \code{"linear"}, \code{"sqrt"}, or \code{"exp"}. If \code{NULL} the weights do not decay.
#' @param method The penalisation type passed to \link{ae_rwnn}. Set to \code{NULL} (default), \code{"l1"}, or \code{"l2"}. If \code{NULL}, \link{rwnn} is used as the base learner.
#' @param type A string indicating whether this is a regression or classification problem. 
#' @param control A list of additional arguments passed to the \link{control_rwnn} function.
#' 
#' @return An \link{ERWNN-object}.
#' 
#' @references Friedman J.H. (2001) "Greedy function approximation: A gradient boosting machine." \emph{The Annals of Statistics}, 29, 1189-1232.
#' 
#' @export
boost_rwnn <- function(formula, data = NULL, n_hidden = c(), lambda = NULL, B = 100, epsilon = 0.1, schedule = "sqrt", method = NULL, type = NULL, control = list()) {
    UseMethod("boost_rwnn")
}

boost_rwnn_matrix <- function(X, y, n_hidden = c(), lambda = NULL, B = 100, epsilon = 0.1, schedule = "sqrt", method = NULL, type = NULL, control = list()) {
    ## Checks
    if (is.null(control[["include_data"]])) {
        control$include_data <- FALSE
    }
    
    if (is.null(control[["boost_schedule"]])) {
        control$boost_schedule <- TRUE
    }
    
    dc <- data_checks(y, X)
    
    if (is.null(B) | !is.numeric(B)) {
        B <- 100
        warning("Note: 'B' was not supplied and is therefore set to 100.")
    }
    
    if (is.null(epsilon) | !is.numeric(epsilon)) {
        epsilon <- 0.1
        warning("Note: 'epsilon' was not supplied and is therefore set to 0.1.")
    }
    else if (epsilon > 1) {
        epsilon <- 1
        warning("'epsilon' has to be a number between '0' and '1'.")
    }
    else if (epsilon < 0) {
        epsilon <- 0
        warning("'epsilon' has to be a number between '0' and '1'.")
    }
    
    if (is.null(control$n_features)) {
        control$n_features <- ncol(X) 
    }
    
    ##
    #
    if (is.null(schedule)) {
        w <- epsilon * rep(1, B)
    }
    else {
        if (is.character(schedule)) {
            if (schedule == "linear") {
                w <- epsilon / seq(1, B)
            }
            else if (schedule == "sqrt") {
                w <- epsilon / sqrt(seq(1, B))
            }
            else if (schedule == "exp") {
                w <- epsilon * exp(-seq(0, B - 1))
            }
            else {
                w <- epsilon * rep(1, B)
            }
        }
        else {
            stop("If 'schedule' is not set to 'NULL', then it should be a string; either 'linear', 'sqrt', or 'exp'.")
        }
    }
    
    ##
    N <- nrow(X)
    y_b <- y
    objects <- vector("list", B)
    for (b in seq_len(B)) {
        # Update residual
        if (b > 1) {
            y_b <- y_b - w[b] * predict(objects[[b - 1]], newdata = X)
        }
        
        # Stochastic boosting
        indices_b <- sample(N, N, replace = TRUE)
        
        y_b_i <- y_b[indices_b, , drop = FALSE]
        X_b_i <- X[indices_b, , drop = FALSE]  
        
        # Train base-learner
        if (is.null(method)) {
            objects[[b]] <- rwnn_matrix(X = X_b_i, y = y_b_i, n_hidden = n_hidden, lambda = lambda, type = type, control = control)
        }
        else {
            objects[[b]] <- ae_rwnn_matrix(X = X_b_i, y = y_b_i, n_hidden = n_hidden, lambda = lambda, method = method, type = type, control = control)
        }
        
    }
    
    ##
    object <- list(
        formula = NULL,
        data = list(X = X, y = y, C = ifelse(type == "regression", NA, colnames(y))), 
        models = objects, 
        weights = w, 
        method = "boosting"
    )  
    
    class(object) <- "ERWNN"
    return(object)
}

#' @rdname boost_rwnn
#' @method boost_rwnn formula
#' 
#' @example inst/examples/boostrwnn_example.R
#' 
#' @export
boost_rwnn.formula <- function(formula, data = NULL, n_hidden = c(), lambda = NULL, B = 100, epsilon = 0.1, schedule = "sqrt", method = NULL, type = NULL, control = list()) {
    # Checks for 'n_hidden'
    if (length(n_hidden) < 1) {
        stop("When the number of hidden layers is 0, or left 'NULL', the RWNN reduces to a linear model, see ?lm.")
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
                message("'data' needs to be supplied when using 'formula'.")
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
    if (!is.null(method)) {
        method <- tolower(method)
        if (!(method %in% c("l1", "l2"))) {
            stop("'method' has to be set to 'NULL', 'l1', or 'l2'.")
        }
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
    mm <- boost_rwnn_matrix(X, y, n_hidden = n_hidden, lambda = lambda, B = B, epsilon = epsilon, schedule = schedule, method = method, type = type, control = control)
    mm$formula <- if (keep_formula) formula
    return(mm)
}