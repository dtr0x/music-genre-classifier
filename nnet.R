softmax <- function(x) {
	exp(x)/sum(exp(x))
}

softmaxGrad <- function(x) {
	softmax(x) * (1 - softmax(x))
}

nnetEval <- function(x, w1, w2) {
	if (is.matrix(x)) {
		p <- dim(x)[2] # assume observations are rows
		x <- t(x) 
	} else {
		p <- length(x)
	}

	M <- dim(w1)[1]
	K <- dim(w2)[1]

	a1 <- w1[, 1] + w1[, 2:(p+1)] %*% x

	a2 <- w2[, 1] + w2[, 2:(M+1)] %*% plogis(a1)

	apply(a2, 2, softmax) # return columns
}

fhatGrad <- function(x, w1, w2) {
	p <- length(x)
	M <- dim(w1)[1]
	K <- dim(w2)[1]

	a1 <- w1[, 1] + w1[, 2:(p+1)] %*% x
	a2 <- w2[, 1] + w2[, 2:(M+1)] %*% plogis(a1)
	gprimeouter <- softmaxGrad(a2)
	logisprime <- dlogis(a1) %*% c(1, x)
	
	nWeights <- M*(p+1) + K*(M+1)
	fkMat <- matrix(0, K, nWeights)

	for (k in 1:K) {
		DfkDw1 <- matrix(0, M, p+1)
		for (m in 1:M) {
			DfkDw1[m, ] <- gprimeouter[k] * w2[k, m+1] %*% logisprime[m, ]
		}
		DfkDw2 <- matrix(0, K, M+1)
		DfkDw2[k, 1] <- gprimeouter[k]
		DfkDw2[k, 2:(M+1)] <- gprimeouter[k] * plogis(a1)

		fkMat[k, ] <- c(t(DfkDw1), t(DfkDw2))
	}

	fkMat
}

nnetGradBatch <- function(Y, X, w1, w2) {
	nTerms <- dim(X)[1]
	p <- dim(X)[2]
	M <- dim(w1)[1]
	K <- dim(w2)[1]
	nWeights <- M*(p+1) + K*(M+1)
	wgrad <- matrix(0, nTerms, nWeights)
	fhat <- t(nnetEval(X, w1, w2))
	err <- Y - fhat
	for (r in 1:nTerms) {
		fhatgradx <- fhatGrad(X[r, ], w1, w2)
		wgrad[r, ] <- err[r, ] %*% fhatgradx
	}

	-2 * colSums(wgrad)
}

nnetGrad <- function(y, x, w1, w2) {
	fhat <- nnetEval(x, w1, w2)
	err <- y - fhat
	fhatgradx <- fhatGrad(x, w1, w2)

	-2 * t(err) %*% fhatgradx
}

nnetTrain <- function(Y, X, w1, w2, lRate, M, batchSize, nEpochs) {
	n <- nrow(X)
	p <- ncol(X)
	K <- ncol(Y)

	w1l <- prod(dim(w1))
	w2l <- prod(dim(w2))

	betavec <- c(t(w1), t(w2))

	for (J in 1:nEpochs) {
		bIdx <- sample(n, batchSize)
		batchX <- X[bIdx, ]
		batchY <- Y[bIdx, ]

		gradw <- nnetGradBatch(batchY, batchX, w1, w2)
		betavec <- betavec - lRate*gradw
		w1 <- matrix(betavec[1:w1l], M, p+1, byrow = TRUE)
		w2 <- matrix(betavec[(w1l+1):(w1l+w2l)], K, M+1, byrow = TRUE)
	}

	return(list(w1, w2))
}

classify <- function(X, w1, w2) {
	pred <- nnetEval(X, w1, w2)
	classes <- apply(pred, 2, function(p) {
		mIdx <- which.max(p)
		p[mIdx] <- 1
		p[-mIdx] <- 0
		p
		})

	t(classes)
}