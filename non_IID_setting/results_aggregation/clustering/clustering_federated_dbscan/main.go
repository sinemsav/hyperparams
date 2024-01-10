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

	// Import data.
	clients := dbscan.GetClientsData(datasetName, skewType, skew, clientNum)

	// [locally] Create local grid from lr/mom values.
	for _, client := range clients {
		client.CreateLocalGrid()
		client.CreateAdjCellMapping()
	}

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

	// [MHE] Calculate average ACC, LR and MOM.
	ComputeAverages(clients)

	// [locally] Reverse the scaling of hps
	for _, client := range clients {
		client.UnscaleParameters()
	}

	fmt.Println("######## From encrypted values: ########")
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

// ComputeAverages computes average accuracy, learning rate and momentum values in grid by using MHE.
func ComputeAverages(clients []*dbscan.Client) {
	totalNumClusters := clients[0].NumClusters
	numBestParams := 3

	avgAccLrMomHE := encrypted.AvgEncrypted(clients, dbscan.ENC_COMPACT_ACC_LR_MOM)[:(totalNumClusters * 3)] // extend 3 times - for: Acc, LR, Mom
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
