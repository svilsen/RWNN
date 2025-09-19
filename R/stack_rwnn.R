#############################################################################
####################### Stacking ERWNN neural networks #######################
#############################################################################

create_folds <- function(X, folds) {
    N <- nrow(X)
    index <- sample(N, N, replace = FALSE)
    fold_index <- rep(seq_len(folds), each = floor(N / folds))
    
    if (length(fold_index) < length(index)) {
        fold_index <- c(fold_index, seq_len(folds)[seq_len(length(index) - length(fold_index))])
    }
    
    return(unname(split(x = index, f = fold_index)))
}

estimate_weights_stack <- function(C, b, B) {
    # Creating matricies for QP optimisation problem.
    # NB: diagonal matrix is added to ensure the matrix is invertible due to potential numeric instability.
    D <- t(C) %*% C + diag(1e-8, nrow = ncol(C), ncol = ncol(C))
    d <- t(C) %*% b
    A <- rbind(t(matrix(rep(1, B), ncol = 1)), diag(B), -diag(B))
    b <- c(1, rep(0, B), rep(-1, B))
    
    # Solution to QP optimisation problem
    w <- solve.QP(D, d, t(A), b, meq = 1)$solution
    
    # Ensure all weights are valid (some may not be due to machine precision)
    w[w < 1e-16] <- 1e-16
    w[w > (1 - 1e-16)] <- (1 - 1e-16)
    w <- w / sum(w)
    
    return(w)
}

#' @title Stacking random weight neural networks
#' 
#' @description Use stacking to create ensemble random weight neural networks.
#' 
#' @param formula A \link{formula} specifying features and targets used to estimate the parameters of the output layer. 
#' @param data A data-set (either a \link{data.frame} or a \link[tibble]{tibble}) used to estimate the parameters of the output layer.
#' @param n_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant(s) passed to either \link{rwnn} or \link{ae_rwnn} (see \code{method} argument).
#' @param B The number of models in the stack.
#' @param optimise TRUE/FALSE: Should the stacking weights be optimised (or should the stack just predict the average)? 
#' @param folds The number of folds used when optimising the stacking weights (see \code{optimise} argument). 
#' @param method The penalisation type passed to \link{ae_rwnn}. Set to \code{NULL} (default), \code{"l1"}, or \code{"l2"}. If \code{NULL}, \link{rwnn} is used as the base learner.
#' @param type A string indicating whether this is a regression or classification problem. 
#' @param control A list of additional arguments passed to the \link{control_rwnn} function.
#' 
#' @return An \link{ERWNN-object}.
#' 
#' @references Wolpert D. (1992) "Stacked generalization." \emph{Neural Networks}, 5, 241-259.
#' 
#' Breiman L. (1996) "Stacked regressions." \emph{Machine Learning}, 24, 49-64.
#' 
#' @export
stack_rwnn <- function(formula, data = NULL, n_hidden = c(), lambda = NULL, B = 100, optimise = FALSE, folds = 10, method = NULL, type = NULL, control = list()) {
    UseMethod("stack_rwnn")
}

stack_rwnn_matrix <- function(X, y, n_hidden = c(), lambda = NULL, B = 100, optimise = FALSE, folds = 10, method = NULL, type = NULL, control = list()) {
    ## Checks
    if (is.null(control[["include_data"]])) {
        control$include_data <- FALSE
    }
    
    dc <- data_checks(y, X)
    
    if (!is.logical(optimise)) {
        stop("'optimise' has to be 'TRUE'/'FALSE'.")
    }
    
    if (is.null(B) | (!is.numeric(B))) {
        B <- 100
        warning("Note: 'B' was not supplied and is therefore set to 100.")
    }
    
    if (is.null(control$n_features)) {
        control$n_features <- ceiling(dim(X)[2] / 3)
    }
    
    control$n_hidden <- n_hidden
    control <- do.call(control_rwnn, control)
    
    if (optimise) {
        if ((is.null(folds) | folds < 1) & (control[["lnorm"]] != "l1")) {
            loocv <- TRUE
        }
        else {
            loocv <- FALSE    
        }
        
        if (!loocv) {
            fold_index <- create_folds(X, folds)
        }
        
        C <- matrix(NA, nrow = nrow(X), ncol = B)
    }
    
    objects <- vector("list", B)
    for (b in seq_len(B)) {
        if (is.null(method)) {
            object_b <- rwnn_matrix(X = X, y = y, n_hidden = n_hidden, lambda = lambda, type = type, control = control)
        }
        else {
            object_b <- ae_rwnn_matrix(X = X, y = y, n_hidden = n_hidden, lambda = lambda, method = method, type = type, control = control)
        }
        
        if (optimise) {
            H <- rwnn_forward(X, object_b$weights$W, object_b$activation, object_b$bias$W)
            H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = n_hidden[i]))
            
            if (object_b$combined$W) {
                H <- do.call("cbind", H)
            } else {
                H <- H[[length(H)]]
            }
            
            if (object_b$bias$beta) {
                H <- cbind(1, H)
            }
            
            O <- H
            if (object_b$combined$X) {
                O <- cbind(X, H)
            }
            
            if (loocv) {
                C[, b] <- loocv(O, y, object_b[["weights"]][["beta"]], object_b$lambda[length(object_b$lambda)])
            }
            else {
                for (k in seq_len(folds)) {
                    Ok <- matrix(O[-fold_index[[k]], ], ncol = ncol(O))
                    yk <- matrix(y[-fold_index[[k]], ], ncol = ncol(y))
                    beta_b <- estimate_output_weights(Ok, yk, object_b$lnorm[length(object_b$lnorm)], object_b$lambda[length(object_b$lambda)])$beta
                    
                    Om <- matrix(O[fold_index[[k]], ], ncol = ncol(O))
                    C[fold_index[[k]], b] <- Om %*% beta_b
                }
            }
        }
        
        objects[[b]] <- object_b
    }
    
    ##
    if (optimise) {
        w <- estimate_weights_stack(C = C, b = y, B = B)
    } else {
        w <- rep(1 / B, B)
    }
    
    ##
    object <- list(
        formula = NULL,
        data = list(X = X, y = y, C = ifelse(type == "regression", NA, colnames(y))), 
        models = objects, 
        weights = w, 
        method = "stacking"
    )  
    
    class(object) <- "ERWNN"
    return(object)
}

#' @rdname stack_rwnn
#' @method stack_rwnn formula
#' 
#' @example inst/examples/stackrwnn_example.R
#' 
#' @export
stack_rwnn.formula <- function(formula, data = NULL, n_hidden = c(), lambda = NULL, B = 100, optimise = FALSE, folds = 10, method = NULL, type = NULL, control = list()) {
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
    mm <- stack_rwnn_matrix(X, y, n_hidden = n_hidden, lambda = lambda, B = B, optimise = optimise, folds = folds, method = method, type = type, control = control)
    mm$formula <- if (keep_formula) formula
    return(mm)
}
