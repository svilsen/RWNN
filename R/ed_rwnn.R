#######################################################################################
####################### Ensemble Deep ERWNN neural network ############################
#######################################################################################

#' @title Ensemble deep random weight neural networks
#' 
#' @description Use multiple layers to create deep ensemble random weight neural network models.
#' 
#' @param formula A \link{formula} specifying features and targets used to estimate the parameters of the output layer. 
#' @param data A data-set (either a \link{data.frame} or a \link[tibble]{tibble}) used to estimate the parameters of the output layer.
#' @param n_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant(s) passed to either \link{rwnn} or \link{ae_rwnn} (see \code{method} argument).
#' @param method The penalisation type passed to \link{ae_rwnn}. Set to \code{NULL} (default), \code{"l1"}, or \code{"l2"}. If \code{NULL}, \link{rwnn} is used as the base learner.
#' @param type A string indicating whether this is a regression or classification problem. 
#' @param control A list of additional arguments passed to the \link{control_rwnn} function.
#' 
#' @return An \link{ERWNN-object}.
#' 
#' @references Shi Q., Katuwal R., Suganthan P., Tanveer M. (2021) "Random vector functional link neural network based ensemble deep learning." \emph{Pattern Recognition}, 117, 107978.
#' 
#' @export
ed_rwnn <- function(formula, data = NULL, n_hidden, lambda = 0, method = NULL, type = NULL, control = list()) {
    UseMethod("ed_rwnn")
}

ed_rwnn_matrix <- function(X, y, n_hidden, lambda = 0, method = NULL, type = NULL, control = list()) {
    ## Checks
    #
    control$n_hidden <- n_hidden
    
    #
    if (is.null(control[["include_data"]])) {
        control$include_data <- FALSE
    }
    
    #
    if (is.null(control[["include_estimate"]])) {
        control$include_estimate <- FALSE
    }
    
    #
    if (is.null(control[["combine_hidden"]])) {
        control$combine_hidden <- FALSE
    }
    
    if (control[["combine_hidden"]]) {
        control$combine_hidden <- FALSE
        warning("'combine_hidden' has to be set to 'FALSE' for the 'ed_rwnn' model to function correctly.")
    }
    
    #
    control <- do.call(control_rwnn, control)
    
    #
    dc <- data_checks(y, X)
    
    ## 
    if (is.null(method)) {
        deeprwnn <- rwnn_matrix(X = X, y = y, n_hidden = n_hidden, lambda = lambda, type = type, control = control)
    }
    else {
        deeprwnn <- ae_rwnn_matrix(X = X, y = y, n_hidden = n_hidden, lambda = lambda, method = method, type = type, control = control)
    }
    
    H <- rwnn_forward(X = X, W = deeprwnn$weights$W, activation = deeprwnn$activation, bias = deeprwnn$bias$W)
    H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = deeprwnn$n_hidden[i]))
    
    objects <- vector("list", length(n_hidden))
    for (i in seq_along(n_hidden)) {
        ## Set-up RWNN object
        rwnn_i <- deeprwnn
        rwnn_i$weights$W <- rwnn_i$weights$W[seq_len(i)]
        
        ## Estimate parameters in output layer
        H_i <- H[[i]]
        if (control$bias_output) {
            H_i <- cbind(1, H_i)
        }
        
        O_i <- H_i
        if (control$combine_input) {
            O_i <- cbind(X, O_i)
        }
        
        W_i <- estimate_output_weights(O_i, y, control$lnorm, lambda)
        
        ##
        rwnn_i$weights$beta <- W_i$beta
        rwnn_i$sigma <- W_i$sigma
        
        objects[[i]] <- rwnn_i
    }
    
    object <- list(
        formula = NULL,
        data = list(X = X, y = y, C = ifelse(type == "regression", NA, colnames(y))), 
        models = objects, 
        weights = rep(1L / length(n_hidden), length(n_hidden)), 
        method = "ed"
    ) 
    
    class(object) <- "ERWNN"
    return(object)
}


#' @rdname ed_rwnn
#' @method ed_rwnn formula
#' 
#' @example inst/examples/edrwnn_example.R
#' 
#' @export
ed_rwnn.formula <- function(formula, data = NULL, n_hidden, lambda = 0, method = NULL, type = NULL, control = list()) {
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
    } else if (tolower(type) %in% c("r", "reg", "regression")) {
        type <- "regression"
    } else {
        stop("'type' has not been correctly specified, it needs to be set to either 'regression' or 'classification'.")
    }
    
    #
    mm <- ed_rwnn_matrix(X, y, n_hidden = n_hidden, lambda = lambda, method = method, type = type, control = control)
    mm$formula <- if (keep_formula) formula
    return(mm)
}