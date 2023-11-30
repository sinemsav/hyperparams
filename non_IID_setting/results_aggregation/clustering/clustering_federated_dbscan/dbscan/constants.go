package dbscan

import "math"

// DBSCAN params.
const DBSCAN_L = 0.15 // Granularity of the grid
var (
	DBSCAN_MINPOINTS = 4 // 2 if qty skew
	GRIDROWS         = int(math.Ceil(1 / DBSCAN_L))
	GRIDCOLS         = int(math.Ceil(1 / DBSCAN_L))
)

// For Encryption.
const (
	SCALEGRID = 1.0 / 150

	ENC_DENSE       = 1
	ENC_CLUSTERSIZE = 2
	ENC_ACC         = 3
	ENC_LR          = 4
	ENC_MOM         = 5

	ENC_CONCAT = 6 // TODO: use only packed version, remove acc/lr/mom
)

// For param (un)scaling.
const (
	MINLR  = 0.01
	MAXLR  = 0.5
	MINMOM = 0
	MAXMOM = 0.95 // TODO: check if 0.9, 0.95;
)
