softmax <- function(x) {
	exp(x)/sum(exp(x))
}

softmaxGrad <- function(x) {
	softmax(x) * (1 - softmax(x))
}

nnetEval <- function(x, w1, w2) {
	if (is.matrix(x)) {
		p <- dim(x)[2]
		x <- t(x) # assume observations are rows
	} else {
		p <- length(x)
	}

	M <- dim(w1)[1]
	K <- dim(w2)[1]

	a1 <- w1[, 1] + w1[, 2:(p+1)] %*% x

	a2 <- w2[, 1] + w2[, 2:(M+1)] %*% plogis(a1)

	apply(a2, 2, softmax)
}

nnetGrad <- function(x, y, wm1, wm2) {
	if (is.matrix(x)) {
		p <- dim(x)[2]
		x <- t(x) # assume observations are rows
	} else {
		p <- length(x)
	}

	M <- dim(w1)[1]
	K <- dim(w2)[1]

	a1 <- w1[, 1] + w1[, 2:(p+1)] %*% x
	a2 <- w2[, 1] + w2[, 2:(M+1)] %*% plogis(a1)
	gw <- sweep(w2, 1, softmaxGrad(a2), "*")[, 2:(M+1)]

	fk <- dlogis(a1) %*% c(1, x)

	betavecw1 <- rep(0, K*M*(p+1))

	for (k in 1:K) {
		dwk <- sweep(fk, 1, gw[k ,], "*")
		dwk <- as.vector(t(dwk))

		betavecw1 <- c(betavec, dwk)
	}
	
}


p <- 313
M <- 32
K <- 15

set.seed(50)
w1 <- matrix(rnorm(M*(p+1)), M, p+1)
w2 <- matrix(rnorm(K*(M+1)), K, M+1)

x <- xMat[1, ]
y <- nneteval(x, w1, w2, activlogistic)