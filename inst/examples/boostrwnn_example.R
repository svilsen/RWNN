N <- 2000
p <- 5

s <- seq(0, pi, length.out = N)
X <- matrix(NA, ncol = p, nrow = N)
X[, 1] <- sin(s)
X[, 2] <- cos(s)
X[, 3] <- s
X[, 4] <- s^2
X[, 5] <- s^3

beta <- matrix(rnorm(p), ncol = 1) 
y <- X %*% beta + rnorm(N, 0, 1)

N_hidden <- 100
B <- 10
epsilon <- 0.2
lambda <- 0.2
\dontrun{
boost_rwnn(X = X, y = y, N_hidden = N_hidden, 
           lambda = lambda, B = B, epsilon = epsilon)
}