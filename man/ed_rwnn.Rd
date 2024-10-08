% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ed_rwnn.R
\name{ed_rwnn}
\alias{ed_rwnn}
\alias{ed_rwnn.formula}
\title{Ensemble deep random weight neural networks}
\usage{
ed_rwnn(
  formula,
  data = NULL,
  n_hidden,
  lambda = 0,
  method = NULL,
  type = NULL,
  control = list()
)

\method{ed_rwnn}{formula}(
  formula,
  data = NULL,
  n_hidden,
  lambda = 0,
  method = NULL,
  type = NULL,
  control = list()
)
}
\arguments{
\item{formula}{A \link{formula} specifying features and targets used to estimate the parameters of the output layer.}

\item{data}{A data-set (either a \link{data.frame} or a \link[tibble]{tibble}) used to estimate the parameters of the output layer.}

\item{n_hidden}{A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).}

\item{lambda}{The penalisation constant(s) passed to either \link{rwnn} or \link{ae_rwnn} (see \code{method} argument).}

\item{method}{The penalisation type passed to \link{ae_rwnn}. Set to \code{NULL} (default), \code{"l1"}, or \code{"l2"}. If \code{NULL}, \link{rwnn} is used as the base learner.}

\item{type}{A string indicating whether this is a regression or classification problem.}

\item{control}{A list of additional arguments passed to the \link{control_rwnn} function.}
}
\value{
An \link{ERWNN-object}.
}
\description{
Use multiple layers to create deep ensemble random weight neural network models.
}
\examples{
n_hidden <- c(20, 15, 10, 5)
lambda <- 0.01

#
m <- ed_rwnn(y ~ ., data = example_data, n_hidden = n_hidden, lambda = lambda)
}
\references{
Shi Q., Katuwal R., Suganthan P., Tanveer M. (2021) "Random vector functional link neural network based ensemble deep learning." \emph{Pattern Recognition}, 117, 107978.
}
