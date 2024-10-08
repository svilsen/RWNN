% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/reduce_network.R
\name{reduce_network}
\alias{reduce_network}
\alias{reduce_network.RWNN}
\alias{reduce_network.ERWNN}
\title{Reduce the weights of a random weight neural network.}
\usage{
reduce_network(object, method, retrain = TRUE, ...)

\method{reduce_network}{RWNN}(object, method, retrain = TRUE, ...)

\method{reduce_network}{ERWNN}(object, method, retrain = TRUE, ...)
}
\arguments{
\item{object}{An \link{RWNN-object} or \link{ERWNN-object}.}

\item{method}{A string, or a function, setting the method used to reduce the network (see details).}

\item{retrain}{TRUE/FALSE: Should the output weights be retrained after reduction (defaults to \code{TRUE})?}

\item{...}{Additional arguments passed to the reduction method (see details).}
}
\value{
A reduced \link{RWNN-object} or \link{ERWNN-object}.
}
\description{
Methods for weight and neuron pruning in random weight neural networks.
}
\details{
The '\code{method}' and additional arguments required by the method are:
\describe{
  \item{\code{"global"} (or \code{"glbl"})}{\describe{
     \item{\code{p}: The proportion of weights to remove globally based on magnitude.}{}
  }}
  \item{\code{"uniform"} (or \code{"unif"})}{\describe{
     \item{\code{p}: The proportion of weights to remove uniformly layer-by-layer based on magnitude.}{}
  }}
  \item{\code{"lamp"}}{\describe{
     \item{\code{p}: The proportion of weights to remove based on LAMP scores.}{}
  }}
  \item{\code{"apoz"}}{\describe{
     \item{\code{p}: The proportion of neurons to remove based on proportion of zeroes produced.}{}
     \item{\code{tolerance}: The tolerance used when identifying zeroes.}{}
     \item{\code{type}: A string indicating whether weights should be removed globally (\code{'global'}) or uniformly  (\code{'uniform'}).}{}
  }}
  \item{\code{"correlation"} (or \code{"cor"})}{\describe{
     \item{\code{type}: The type of correlation (argument passed to \link{cor} function).}{}
     \item{\code{rho}: The correlation threshold used to remove neurons.}{}
  }}
  \item{\code{"correlationtest"} (or \code{"cortest"})}{\describe{
     \item{\code{type}: The type of correlation (argument passed to \link{cor} function).}{}
     \item{\code{rho}: The correlation threshold used to remove neurons.}{}
     \item{\code{alpha}: The significance levels used to test whether the observed correlation between two neurons is small than \code{rho}.}{}
  }}
  \item{\code{"relief"}}{\describe{
     \item{\code{p}: The proportion of neurons or weights to remove based on relief scores.}{}
     \item{\code{type}: A string indicating whether neurons (\code{'neuron'}) or weights (\code{'weight'}) should be removed.}{}
  }}
  \item{\code{"output"}}{\describe{
     \item{\code{tolerance}: The tolerance used when removing zeroes from the output layer.}{}
  }}
} 

If the object is an \link{ERWNN-object}, the reduction is applied to all \link{RWNN-object}'s in the \link{ERWNN-object}. Furthermore, when
the \link{ERWNN-object} is created as a stack and the weights of the stack is trained, then '\code{method}' can be set to:
\describe{
\item{\code{"stack"}}{\describe{
     \item{\code{tolerance}: The tolerance used when removing elements from the stack.}{}
  }}
}

Lastly, '\code{method}' can also be passed as a function, with additional arguments passed through the \code{...} argument. 
NB: features and target are passed using the names \code{X} and \code{y}, respectively.
}
\examples{
## RWNN-object
n_hidden <- c(10, 15)
lambda <- 2

m <- rwnn(y ~ ., data = example_data, n_hidden = n_hidden, 
          lambda = lambda, control = list(lnorm = "l2"))

m |> 
    reduce_network(method = "relief", p = 0.2, type = "neuron") |> 
    (\(x) x$weights)()

m |> 
    reduce_network(method = "relief", p = 0.2, type = "neuron") |> 
    reduce_network(method = "correlationtest", rho = 0.995, alpha = 0.05) |> 
    (\(x) x$weights)()


m |> 
    reduce_network(method = "relief", p = 0.2, type = "neuron") |> 
    reduce_network(method = "correlationtest", rho = 0.995, alpha = 0.05) |> 
    reduce_network(method = "lamp", p = 0.2) |> 
    (\(x) x$weights)()

m |> 
    reduce_network(method = "relief", p = 0.4, type = "neuron") |> 
    reduce_network(method = "relief", p = 0.4, type = "weight") |> 
    reduce_network(method = "output") |> 
    (\(x) x$weights)()

## ERWNN-object (reduction is performed element-wise on each RWNN)
n_hidden <- c(10, 15)
lambda <- 2
B <- 100

\donttest{
m <- bag_rwnn(y ~ ., data = example_data, n_hidden = n_hidden, 
              lambda = lambda, B = B, control = list(lnorm = "l2"))

m |> 
    reduce_network(method = "relief", p = 0.2, type = "neuron") |> 
    reduce_network(method = "relief", p = 0.2, type = "weight") |> 
    reduce_network(method = "output")
}

\donttest{
m <- stack_rwnn(y ~ ., data = example_data, n_hidden = n_hidden,
                lambda = lambda, B = B, optimise = TRUE)

# Number of models in stack
length(m$weights)
# Number of models in stack with weights > .Machine$double.eps
length(m$weights[m$weights > .Machine$double.eps]) 

m |> 
    reduce_network(method = "stack", tolerance = 1e-8) |> 
    (\(x) x$weights)()
}
}
\references{
Han S., Mao H., Dally W.J. (2016) "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding." arXiv: 1510.00149.

Hu H., Peng R., Tai Y.W., Tang C.K. (2016) "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures." arXiv: 1607.03250.

Morcos A.S., Yu H., Paganini M., Tian Y. (2019) "One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers." arXiv: 1906.02773.

Lee J., Park S., Mo S., Ahn S., Shin J. (2021) "Layer-adaptive sparsity for the Magnitude-based Pruning." arXiv: 2010.07611.

Dekhovich A., Tax D.M., Sluiter M.H., Bessa M.A. (2024) "Neural network relief: a pruning algorithm based on neural activity." \emph{Machine Learning}, 113, 2597-2618.
}
