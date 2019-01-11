source("nnet.R")

# scale and rotate data with given means, std devs and principal components
transformData <- function(X, allMeans, allStdDevs, prComps) {
	for (i in 1:ncol(X)) {
		X[, i] <- (X[, i] - allMeans[i])/allStdDevs[i]
	}

	X %*% prComps
}

# compute ROC metrics with macro averaging
rocData <- function(p, Y) {
	K <- ncol(p)
	precision <- rep(0, K)
	recall <- rep(0, K)
	fscore <- rep(0, K)

	for (i in 1:K) {
		tn <- sum(as.integer((p[,i] == Y[,i])[p[,i] == 0])) 
		fn <- sum(as.integer((p[,i] != Y[,i])[p[,i] == 0])) 
		fp <- sum(as.integer((p[,i] != Y[,i])[p[,i] == 1])) 
		tp <- sum(as.integer((p[,i] == Y[,i])[p[,i] == 1])) 

		precision[i] <- ifelse(tp+fp != 0, tp/(tp+fp), 0)
		recall[i] <- ifelse(tp+fn != 0, tp/(tp+fn), 0)
		fscore[i] <- ifelse(precision[i]+recall[i] !=0,
			(2*precision[i]*recall[i])/(precision[i]+recall[i]), 0)
	}

	pAvg <- mean(precision)
	rAvg <- mean(recall)
	fAvg <- mean(fscore)

	return(list(p=pAvg, r=rAvg, f=fAvg))
}


########## begin main script ###############

load("discogs.RData")

# remove features with 0 variance
trainData <- trainData[, (apply(trainData, 2, var) != 0)]

nAll <- nrow(trainData)
nTrain <- floor(0.6 * nAll)
nVal <- floor(0.2 * nAll)

trainSet <- trainData[1:nTrain, ]
valSet <- trainData[(nTrain+1):(nTrain+nVal), ]
testSet <- trainData[(nTrain+nVal+1):nAll, ]

trainTruths <- trueGenres[1:nTrain, ]
valTruths <- trueGenres[(nTrain+1):(nTrain+nVal), ]
testTruths <- trueGenres[(nTrain+nVal+1):nAll, ]

allMeans <- colMeans(trainSet)
allStdDevs <- apply(trainSet, 2, sd)
trainScaled <- scale(trainSet)

pca <- prcomp(trainScaled, center = FALSE, scale. = FALSE)

cpve <- summary(pca)$importance[3, ]

prComps <- pca$rotation[, cpve <= 0.95]

trainSetReduce <- trainScaled %*% prComps

valSetReduce <- transformData(valSet, allMeans, allStdDevs, prComps)
testSetReduce <- transformData(valSet, allMeans, allStdDevs, prComps)

p <- ncol(trainSetReduce)
K <- ncol(trueGenres)
lRate <- 0.01
nHiddenUnits <- c(5, 10, 25, 50, 100, 125, 150, 200, 225)
batchSize <- 32
nEpochs <- 200
set.seed(1)

bestFScore <- 0
rocTable <- matrix(0, length(nHiddenUnits), 4, dimnames = list(NULL, c("# of Hidden Units", "Precision", "Recall", "F-score")))
idx <- 1
for (M in nHiddenUnits) {
	w1 <- matrix(rnorm(M*(p+1)), M, p+1)
	w2 <- matrix(rnorm(K*(M+1)), K, M+1)

	parms <- nnetTrain(trainTruths, trainSetReduce, w1, w2, lRate, M, batchSize, nEpochs)

	valPred <- classify(valSetReduce, parms[[1]], parms[[2]])

	roc <- rocData(valPred, valTruths)

	if(roc$f > bestFScore) {
		bestFScore <- roc$f
		bestParms <- parms
	}

	rocTable[idx, ] <- c(M, roc$p, roc$r, roc$f)
	idx <- idx + 1
}


print("ROC Metrics for Validation Set:")
print(rocTable)

testPred <- classify(testSetReduce, bestParms[[1]], bestParms[[2]])
print("ROC Metrics for Test Set:")
rocTest <- rocData(testPred, testTruths)
cat("Precision: ", roc$p, " Recall: ", roc$r, " F-score: ", roc$f, "\n")
sum(as.integer(apply(testPred, 1, function(x) which(x == 1)) == apply(testTruths, 1, function(x) which(x == 1))))

successes <- sum(as.integer(apply(testPred, 1, function(x) which(x == 1)) == apply(testTruths, 1, function(x) which(x == 1))))

cat(successes, " recording are accurately classified on the test set." "\n")