// TODO: properly refactor to only use packed AVG [ACC+LR+MOM]
// TODO: clean up: remove timings, random comments, plaintext version

package main

import (
	"clustering_federated_dbscan/dbscan"
	"clustering_federated_dbscan/encrypted"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"log"
	"time"
)

func main() {
	log.Println("start")

	datasetName := "cifar10"
	skewType := "feature"
	skew := "0.1"
	clientNum := 10

	packed := true // Pack ACC+LR+MOM together
	//dummyBootstrapping := false

	// Import data.
	clients := dbscan.GetClientsData(datasetName, skewType, skew, clientNum)

	// [locally] Create local grid from lr/mom values.
	for _, client := range clients {
		client.CreateLocalGrid()
		client.CreateAdjCellMapping()
	}

	// Get DenseCell Mask - both plaintext + MHE
	//denseMaskHE := GetDenseCellMask2(clients, dummyBootstrapping)
	// [MHE] Get DenseCell Mask
	start := time.Now()
	denseMaskHE := GetDenseCellMask(clients)
	elapsed := time.Since(start)
	fmt.Println("dense single took: ", elapsed)

	// [locally] Use denseMask to remove noise and group points/params for clustering
	for _, client := range clients {
		client.RemoveOutliers(denseMaskHE) // remove outliers and assign non-dense points to dense cells (in map)
		client.CreateClusterParams(denseMaskHE)
	}

	// Avg accuracy calculated encrypted.
	//acc, clusterSize := getAvgAccAndClusterSize(clients)
	//lr, mom := getAvgLrAndMomEnc(clients, acc, clusterSize)

	// [MHE] Calculate average ACC, LR and MOM.
	if packed {
		ComputeAverages(clients)
	} else {
		ComputeAccEnc(clients)
		ComputeAvgLrAndMomEnc(clients)
	}

	// [locally] Reverse the scaling of hps
	for _, client := range clients {
		client.UnscaleParameters()
	}

	//lrDe := DescaleParameter(lr, dbscan.MAXLR, dbscan.MINLR)
	//momDe := DescaleParameter(mom, dbscan.MAXMOM, dbscan.MINMOM)
	//
	//fmt.Println("UNENC lr unscaled")
	//dbscan.PrintVec(lrDe)
	//fmt.Println("UNENC lr unscaled")
	//dbscan.PrintVec(momDe)

	fmt.Println("######## ENC ########")
	// Print results (LR/MOM HPs)
	clients[0].PrintFinalHPs()

	log.Println("done")
}

// GetDenseCellMask collectively calculates the dense mask under encryption.
// The clients sum up their local grids and compare it with a predefined threshold (DBSCAN minpoints param).
func GetDenseCellMask(clients []*dbscan.Client) *mat.Dense {
	maskHE := encrypted.AggregateGrid(clients, dbscan.ENC_DENSE, true, false)
	denseMaskHE := mat.NewDense(dbscan.GRIDROWS, dbscan.GRIDCOLS, maskHE)
	denseMaskHE.Apply(dbscan.ApplyDenseMaskApprox, denseMaskHE)
	dbscan.PrintMat(denseMaskHE)

	return denseMaskHE
}

func ComputeAverages(clients []*dbscan.Client) {
	totalNumClusters := clients[0].NumClusters
	numBestParams := 3

	start := time.Now()
	avgAccLrMomHE := encrypted.AvgEncrypted(clients, dbscan.ENC_CONCAT)[:(totalNumClusters * 3)]
	elapsed := time.Since(start)

	fmt.Println("packed took: ", elapsed)
	fmt.Println(avgAccLrMomHE)

	for _, client := range clients {
		client.SaveBestAcc(avgAccLrMomHE[:(totalNumClusters)], numBestParams)
		lr := mat.NewVecDense(totalNumClusters, avgAccLrMomHE[totalNumClusters:(totalNumClusters*2)])
		mom := mat.NewVecDense(totalNumClusters, avgAccLrMomHE[(totalNumClusters*2):])

		lrVec := dbscan.ExtractVecElements(lr, client.SortedIdxMapping[:client.NumBestParams])
		momVec := dbscan.ExtractVecElements(mom, client.SortedIdxMapping[:client.NumBestParams])
		client.SaveBestLrAndMom(lrVec, momVec)
	}
}

// ComputeAccEnc computes average accuracy values in grid by using MHE.
func ComputeAccEnc(clients []*dbscan.Client) {
	totalNumClusters := clients[0].NumClusters
	start := time.Now()
	accAggregatedHE := encrypted.AvgEncrypted(clients, dbscan.ENC_ACC)[:totalNumClusters]
	elapsed := time.Since(start)
	fmt.Println("acc single took: ", elapsed)
	fmt.Println(accAggregatedHE)

	avgAccHE := mat.NewVecDense(totalNumClusters, accAggregatedHE)
	numBestParams := 3 // TODO: get top X% (10%) clusters w.r.t. acc

	for _, client := range clients {
		client.SaveBestAcc(avgAccHE.RawVector().Data, numBestParams)
	}
}

// ComputeAvgLrAndMomEnc computes average LR and MOM values in grid by using MHE.
func ComputeAvgLrAndMomEnc(clients []*dbscan.Client) {
	numBestParams := clients[0].NumBestParams
	//totalNumClusters := clients[0].NumClusters

	//start := time.Now()
	//avgLRMomHE := encrypted.AvgEncrypted(clients, dbscan.ENC_CONCAT)[:(totalNumClusters * 3)]
	//elapsed := time.Since(start)
	//fmt.Println("packed took: ", elapsed)
	//fmt.Println(avgLRMomHE)

	start := time.Now()
	avgLrHE := encrypted.AvgEncrypted(clients, dbscan.ENC_LR)[:numBestParams]
	elapsed := time.Since(start)
	fmt.Println("lr only took: ", elapsed)
	fmt.Println(avgLrHE)

	start = time.Now()
	avgMomHE := encrypted.AvgEncrypted(clients, dbscan.ENC_MOM)[:numBestParams]
	elapsed = time.Since(start)
	fmt.Println("mom only took: ", elapsed)
	fmt.Println(avgMomHE)

	for _, client := range clients {
		client.SaveBestLrAndMom(mat.NewVecDense(numBestParams, avgLrHE), mat.NewVecDense(numBestParams, avgMomHE))
	}
}

// to delete
func GetDenseCellMask2(clients []*dbscan.Client, dummyBootstrapping bool) *mat.Dense {
	// Sum up local grids from clients.
	aggregatedGrid := mat.NewDense(dbscan.GRIDROWS, dbscan.GRIDCOLS, nil)
	for _, client := range clients {
		aggregatedGrid.Add(aggregatedGrid, client.Grid())
	}
	dbscan.PrintMat(aggregatedGrid)

	// Create dense mask based on DBSCAN_MINPOINTS value.
	denseMask := mat.NewDense(dbscan.GRIDROWS, dbscan.GRIDCOLS, nil)
	denseMask.Apply(ApplyThresholdGrid, aggregatedGrid)
	dbscan.PrintMat(denseMask)

	// [MHE] Sum up local grids from clients + threshold comparison.

	maskHE := encrypted.AggregateGrid(clients, dbscan.ENC_DENSE, true, dummyBootstrapping)
	denseMaskHE := mat.NewDense(dbscan.GRIDROWS, dbscan.GRIDCOLS, maskHE)
	denseMaskHE.Apply(dbscan.ApplyDenseMaskApprox, denseMaskHE)
	dbscan.PrintMat(denseMaskHE)

	return denseMaskHE
}

func ApplyThresholdGrid(i, j int, v float64) float64 {
	if v >= float64(dbscan.DBSCAN_MINPOINTS) {
		return 1
	}
	return 0
}
func getAvgLrAndMomEnc(clients []*dbscan.Client, acc *mat.VecDense, clusterSize *mat.VecDense) (*mat.VecDense, *mat.VecDense) {
	numBestParams := clients[0].NumBestParams

	// Without encryption.
	//println("no encrypt")
	//dbscan.PrintVec(acc)
	//dbscan.PrintVec(clusterSize)

	for _, client := range clients {
		client.SaveBestAcc(acc.RawVector().Data, numBestParams)
		client.SaveBestClusterSize(clusterSize, numBestParams)
	}
	clusterSizeSorted := clients[0].GetClusterSizes()
	idxs := clients[0].GetIdxMapping()
	avgLR := mat.NewVecDense(numBestParams, nil)
	avgMOM := mat.NewVecDense(numBestParams, nil)
	for _, client := range clients {
		sortedLR := dbscan.ExtractVecElements(client.LocalAggregatedParams().LrGrid(), idxs[:numBestParams])
		sortedMOM := dbscan.ExtractVecElements(client.LocalAggregatedParams().MomGrid(), idxs[:numBestParams])

		avgLR.AddVec(avgLR, sortedLR)
		avgMOM.AddVec(avgMOM, sortedMOM)
	}
	avgLR.DivElemVec(avgLR, clusterSizeSorted)
	avgMOM.DivElemVec(avgMOM, clusterSizeSorted)
	fmt.Println("Avg LR nonEnc:")
	dbscan.PrintVec(avgLR)
	fmt.Println("Avg MOM nonEnc:")
	dbscan.PrintVec(avgMOM)

	// ENC
	fmt.Println("################# ENC ###########")
	avgLrHE := encrypted.AvgEncrypted(clients, dbscan.ENC_LR)[:numBestParams]
	fmt.Println(avgLrHE)
	avgMomHE := encrypted.AvgEncrypted(clients, dbscan.ENC_MOM)[:numBestParams]
	fmt.Println(avgMomHE)

	for _, client := range clients {
		client.SaveBestLrAndMom(mat.NewVecDense(numBestParams, avgLrHE), mat.NewVecDense(numBestParams, avgMomHE))
	}

	return avgLR, avgMOM
}

func getAvgAccAndClusterSize(clients []*dbscan.Client) (*mat.VecDense, *mat.VecDense) {
	// Get average accuracy of each cluster.
	totalNumClusters := clients[0].NumClusters
	avgAcc := mat.NewVecDense(totalNumClusters, nil)
	clusterSize := mat.NewVecDense(totalNumClusters, nil)
	for _, client := range clients {
		avgAcc.AddVec(avgAcc, client.LocalAggregatedParams().AccGrid())                   // Sum of accuracies
		clusterSize.AddVec(clusterSize, client.LocalAggregatedParams().ClusterSizeGrid()) // Sum of cluster sizes
	}
	println("avg acc nonecrypt")
	dbscan.PrintVec(avgAcc)
	//printVec(clusterSize)
	avgAcc.DivElemVec(avgAcc, clusterSize) // Divide
	dbscan.PrintVec(avgAcc)

	// [MHE] Get summed accuracy and cluster size of each cluster; then divide on decrypted.
	accAggregatedHE := encrypted.AvgEncrypted(clients, dbscan.ENC_ACC)[:totalNumClusters]
	fmt.Println(accAggregatedHE)

	avgAccHE := mat.NewVecDense(totalNumClusters, accAggregatedHE)
	numBestParams := 3 // TODO: get top X% (10%) clusters w.r.t. acc

	for _, client := range clients {
		client.SaveBestAcc(avgAccHE.RawVector().Data, numBestParams)
	}

	return avgAcc, clusterSize
}

func DescaleParameter(param *mat.VecDense, maxValue, minValue float64) *mat.VecDense {
	scaleDiff := make([]float64, param.Len())
	scaleMin := make([]float64, param.Len())
	for i := range scaleDiff {
		scaleDiff[i] = maxValue - minValue
		scaleMin[i] = minValue
	}

	scaleDiffVector := mat.NewVecDense(param.Len(), scaleDiff)
	scaleMinVector := mat.NewVecDense(param.Len(), scaleMin)

	rescaled := mat.NewVecDense(param.Len(), nil)
	rescaled.MulElemVec(param, scaleDiffVector)
	rescaled.AddVec(rescaled, scaleMinVector)

	return rescaled
}
