package main

import (
	"clustering_federated_dbscan/dbscan"
	"clustering_federated_dbscan/encrypted"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"log"
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
	denseMaskHE := GetDenseCellMask(clients)

	// [locally] Use denseMask to remove noise and group points/params for clustering
	for _, client := range clients {
		client.RemoveOutliers(denseMaskHE) // remove outliers and assign non-dense points to dense cells (in map)
		client.CreateClusterParams(denseMaskHE)
	}

	// [MHE] Calculate average ACC, LR and MOM.
	ComputeAccEnc(clients)
	ComputeAvgLrAndMomEnc(clients)

	// [locally] Reverse the scaling of hps
	for _, client := range clients {
		client.UnscaleParameters()
	}

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

// ComputeAccEnc computes average accuracy values in grid by using MHE.
func ComputeAccEnc(clients []*dbscan.Client) {
	totalNumClusters := clients[0].NumClusters
	accAggregatedHE := encrypted.AvgEncrypted(clients, dbscan.ENC_ACC)[:totalNumClusters]
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

	avgLrHE := encrypted.AvgEncrypted(clients, dbscan.ENC_LR)[:numBestParams]
	fmt.Println(avgLrHE)
	avgMomHE := encrypted.AvgEncrypted(clients, dbscan.ENC_MOM)[:numBestParams]
	fmt.Println(avgMomHE)

	for _, client := range clients {
		client.SaveBestLrAndMom(mat.NewVecDense(numBestParams, avgLrHE), mat.NewVecDense(numBestParams, avgMomHE))
	}
}
