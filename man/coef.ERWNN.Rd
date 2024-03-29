% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/auxiliary_functions_ERWNN.R
\name{coef.ERWNN}
\alias{coef.ERWNN}
\title{Coefficients of the ERWNN-object}
\usage{
\method{coef}{ERWNN}(object, ...)
}
\arguments{
\item{object}{An \link{ERWNN-object}.}

\item{...}{Additional arguments.}
}
\value{
Depended on '\code{method}' and '\code{type}':

If '\code{method}' was \code{"bagging"}, the '\code{type}' yields the following results: 
\describe{
    \item{\code{"mean" (default):}}{A vector containing the average value of each parameter taken across the bootstrap samples.}
    \item{\code{"sd":}}{A vector containing the standard deviation of each parameter taken across the bootstrap samples.}
    \item{\code{"all":}}{A matrix where every column contains the parameters of the output-layer of corresponding boostrap sample.}
}

If '\code{method}' was \code{"boosting"} or \code{"stacking"}, a matrix is returned corresponding to '\code{type == "all"}'.
}
\description{
Coefficients of the ERWNN-object
}
\details{
The additional argument '\code{type}' is only used if '\code{method}' was \code{"bagging"}, in which case it can be supplied with values \code{"all"}, \code{"sd"}, and \code{"mean"} (default), returning the full list of coefficients for all bootstrap samples, the standard deviation of each coefficient across bootstrap samples, and the average value of each coefficient across bootstrap samples, respectively.
}
