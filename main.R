library(jsonlite)

TRAIN_DIR <- "discogsTraintest"

parseMusicJSON <- function(jsonFile) {
	path <- paste(TRAIN_DIR, jsonFile, sep="/")

	# read json file and remove unused metadata element
	musicJSON <- read_json(path)[-2]

	# remove beats_position feature, length differs between recordings and is redundant with bpm / beats_count
	musicJSON$rhythm$beats_position <- NULL 

	musicJSON <- oneHotEncodeTonal(musicJSON)

	musicJSON
}

oneHotEncodeTonal <- function(musicJSON) {
	keys <- c("A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#")

	# encode key_key
	key_key <- musicJSON$tonal$key_key
	key_key_onehot <- as.list(rep(0, 12))
	names(key_key_onehot) <- keys
	key_key_onehot[[key_key]] <- 1
	musicJSON$tonal$key_key <- key_key_onehot
	# encode chords_key
	chords_key <- musicJSON$tonal$chords_key
	chords_key_onehot <- as.list(rep(0, 12))
	names(chords_key_onehot) <- keys
	chords_key_onehot[[chords_key]] <- 1
	musicJSON$tonal$chords_key <- chords_key_onehot

	# encode key_scale
	isMajorKey <- as.integer(musicJSON$tonal$key_scale == "major")
	key_scale_onehot <- list(major = isMajorKey, minor = as.integer(!isMajorKey))
	musicJSON$tonal$key_scale <- key_scale_onehot
	# encode chords_scale
	isMajorChord <- as.integer(musicJSON$tonal$chords_scale == "major")
	chords_scale_onehot <- list(major = isMajorChord, minor = as.integer(!isMajorChord))
	musicJSON$tonal$chords_scale <- chords_scale_onehot

	musicJSON
}


discogsTruths <- read.delim("discogsTruths.tsv")
jsonFiles <- list.files(TRAIN_DIR)
recordingIDs <- sapply(jsonFiles, USE.NAMES = FALSE, function(s) strsplit(s, "[.]")[[1]][1])
trueGenres <- discogsTruths[discogsTruths$rid %in% recordingIDs, ]
musicData <- lapply(jsonFiles, parseMusicJSON)
colNames <- names(unlist(musicData[[1]]))

numRows <- length(jsonFiles)
numCols <- length(colNames)

designMat <- matrix(unlist(musicData), numRows, numCols, byrow = TRUE, dimnames = list(NULL, colNames))


