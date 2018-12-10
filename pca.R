dm <- designMat[, (apply(designMat, 2, var) != 0)]
dmScaled <- scale(dm)

pca <- prcomp(dm, center = TRUE, scale. = TRUE)

cpve <- summary(pca)$importance[3, ]

loadings <- pca$rotation[, cpve <= 0.95]

scores <- dmScaled %*% loadings

allMeans <- colMeans(dm)
allStdDevs <- apply(dm, 2, sd)

# approximate reconstructed data 
uncomp <- scores %*% t(loadings)
uncomp <- sweep(uncomp, 2, allStdDevs, "*")
uncomp <- sweep(uncomp, 2, allMeans, "+")