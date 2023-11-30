package dbscan

import (
	"fmt"
	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/drlwe"
	"github.com/tuneinsight/lattigo/v4/rlwe"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type Parameters struct {
	lr, mom, bs, loss, acc float64
}

type Client struct {
	ID           int
	data         []Parameters // [[lr, mom, bs, loss, accuracy], ...]
	dataFiltered []Parameters // [[lr, mom, bs, loss, accuracy]] without noisy outliers

	grid                     *mat.Dense            // Grid (matrix) where each cell represents total number of local points in that cell.
	pointToAdjCell           map[Point]Cell        // Map each HP point (data point) to its closest adjacent cell
	cellToParametersFiltered map[Cell][]Parameters // Map each grid cell to data points that are inside of it

	MHEParams *EncryptionParams // Params used for multiparty homomorphic encryption

	// Final local aggregation
	NumClusters           int
	localAggregatedParams *GridData

	SortedIdxMapping []int // Indices that map localAggregatedParams values based on sorting criteria

	// Best params (collectively)
	NumBestParams    int
	globalBestParams *GridData

	// Rescaled params
	rescaledLRs  *mat.VecDense
	rescaledMOMs *mat.VecDense
}

func (client *Client) LocalAggregatedParams() *GridData {
	return client.localAggregatedParams
}

func (client *Client) Grid() *mat.Dense {
	return client.grid
}

type EncryptionParams struct {
	PubKey         *rlwe.PublicKey
	SecKey         *rlwe.SecretKey
	RLKEphemSecKey *rlwe.SecretKey

	CKGShare    *drlwe.CKGShare
	RKGShareOne *drlwe.RKGShare
	RKGShareTwo *drlwe.RKGShare
	PCKSShare   *drlwe.PCKSShare

	InputOne []float64
	InputTwo []float64
}

type GridData struct {
	accGrid         *mat.VecDense
	clusterSizeGrid *mat.VecDense
	lrGrid          *mat.VecDense
	momGrid         *mat.VecDense
	bsGrid          *mat.VecDense
}

func (g GridData) AccGrid() *mat.VecDense {
	return g.accGrid
}

func (g GridData) ClusterSizeGrid() *mat.VecDense {
	return g.clusterSizeGrid
}

func (g GridData) LrGrid() *mat.VecDense {
	return g.lrGrid
}

func (g GridData) MomGrid() *mat.VecDense {
	return g.momGrid
}

func (g GridData) BsGrid() *mat.VecDense {
	return g.bsGrid
}

// CreateLocalGrid creates a local grid that represents the number of data points inside each cell of the grid.
func (client *Client) CreateLocalGrid() {
	// Create an empty grid
	grid := mat.NewDense(GRIDROWS, GRIDCOLS, nil)

	for _, params := range client.data {
		cell := (&Point{params.lr, params.mom}).ToCell(DBSCAN_L)
		grid.Set(cell.row, cell.col, grid.At(cell.row, cell.col)+1)
	}

	client.grid = grid
}

// CreateAdjCellMapping maps (creates a map) each local point to the closest adjacent cell.
func (client *Client) CreateAdjCellMapping() {
	client.pointToAdjCell = make(map[Point]Cell)
	for _, params := range client.data {
		point := Point{params.lr, params.mom}
		cell := point.FindClosestAdjacentCell()
		client.pointToAdjCell[point] = cell
	}
}

// CreateClusterParams uses the collective dense mask as a reference for cluster creation
// and merges adjacent dense cells. Finally, the client computes its total number of points and
// its sum of acc/lr/mom for every cluster given by the dense mask.
func (client *Client) CreateClusterParams(denseMask *mat.Dense) {
	denseMaskCopy := mat.NewDense(GRIDROWS, GRIDCOLS, nil)
	denseMaskCopy.Copy(denseMask)

	clusterSize := make([]float64, 0)
	accGrid := make([]float64, 0)
	lrGrid := make([]float64, 0)
	momGrid := make([]float64, 0)
	bsGrid := make([]float64, 0)

	for i := 0; i < GRIDROWS; i++ {
		for j := 0; j < GRIDCOLS; j++ {
			cell := Cell{i, j}

			// Skip if not dense cell.
			if denseMaskCopy.At(cell.row, cell.col) == 0 {
				continue
			}
			denseMaskCopy.Set(cell.row, cell.col, 0)

			// For the dense cell: get all values inside + get all adjacent cells.
			params := client.cellToParametersFiltered[cell]
			nextCells := cell.GetAdjacentCells()

			for len(nextCells) > 0 {
				nextCell := nextCells[0]
				nextCells = nextCells[1:]

				if denseMaskCopy.At(nextCell.row, nextCell.col) == 0 {
					continue
				}
				denseMaskCopy.Set(nextCell.row, nextCell.col, 0)

				params = append(params, client.cellToParametersFiltered[nextCell]...)
				nextCells = append(nextCells, nextCell.GetAdjacentCells()...)
			}

			// Don't continue if empty - we need empty places to align all clients based on denseMaskCopy
			//if len(params) == 0 {
			//	continue
			//}

			clusterSize = append(clusterSize, float64(len(params)))
			accGrid = append(accGrid, 0)
			lrGrid = append(lrGrid, 0)
			momGrid = append(momGrid, 0)
			bsGrid = append(bsGrid, 0)

			for _, param := range params {
				lrGrid[len(lrGrid)-1] += param.lr
				momGrid[len(momGrid)-1] += param.mom
				bsGrid[len(bsGrid)-1] += param.bs
				accGrid[len(accGrid)-1] += param.acc
			}
		}
	}

	client.NumClusters = len(clusterSize) // clusterSize, accGrid, lrGrid, momGrid, bsGrid --> all are same size

	client.localAggregatedParams = &GridData{
		clusterSizeGrid: mat.NewVecDense(len(clusterSize), clusterSize),
		accGrid:         mat.NewVecDense(len(accGrid), accGrid),
		lrGrid:          mat.NewVecDense(len(lrGrid), lrGrid),
		momGrid:         mat.NewVecDense(len(momGrid), momGrid),
		bsGrid:          mat.NewVecDense(len(bsGrid), bsGrid),
	}
}

// RemoveOutliers creates a new array of filtered data by removing outliers.
// If the point is inside a dense cell, the point is not an outlier and remains unchanged.
// If the point is inside a non-dense cell, but its closest adjacent cell is a dense cell, then
// the point is 'moved' to this adjacent dense cell and is not considered an outlier.
// If the above two conditions are not fulfilled the point is an outlier and is removed.
func (client *Client) RemoveOutliers(denseMask *mat.Dense) {
	client.dataFiltered = make([]Parameters, 0)
	client.cellToParametersFiltered = make(map[Cell][]Parameters)

	for _, params := range client.data {
		point := Point{params.lr, params.mom}
		cell := (&point).ToCell(DBSCAN_L)
		adjCell := client.pointToAdjCell[point]

		// If cell is dense - keep point in this cell
		// If adjCell is dense - move point to adj cell
		// otherwise cell is outlier/noise
		if denseMask.At(cell.row, cell.col) == 1 {
			client.cellToParametersFiltered[cell] = append(client.cellToParametersFiltered[cell], params)
		} else if denseMask.At(adjCell.row, adjCell.col) == 1 {
			client.cellToParametersFiltered[adjCell] = append(client.cellToParametersFiltered[adjCell], params)
		} else {
			continue
		}

		client.dataFiltered = append(client.dataFiltered, params)
	}
}

// GenerateKeyPair generates the secret and public key given a set of CKKS parameters.
func (client *Client) GenerateKeyPair(params ckks.Parameters) {
	//start := time.Now()
	if client.MHEParams == nil {
		client.MHEParams = &EncryptionParams{}
	}

	kgen := ckks.NewKeyGenerator(params)
	client.MHEParams.SecKey, client.MHEParams.PubKey = kgen.GenKeyPair()
	//elapsed := time.Since(start)
	//log.Printf("GenerateKeyPair took %s for client %d", elapsed, client.ID)
}

// PrepareInput prepares the input used for collectively computing the dense mask and average (accuracy, lr, mom) values.
// The input values are scaled by SCALEGRID factor in order to perform operations on the ciphertext.
// When using LR/MOM as input, only specific values (the best ones when sorted) are used - this is defined with the IdxMapping parameter.
func (client *Client) PrepareInput(input int) {
	if client.ID == -1 {
		client.MHEParams.InputOne = nil
		client.MHEParams.InputTwo = nil
		return
	}

	var params, localClusterSizes *mat.VecDense

	// TODO: only packed version should be used!
	if input == ENC_CONCAT {
		params = mat.VecDenseCopyOf(client.localAggregatedParams.accGrid)
		localClusterSizes = mat.VecDenseCopyOf(client.localAggregatedParams.clusterSizeGrid)
		params.ScaleVec(SCALEGRID, params)
		localClusterSizes.ScaleVec(SCALEGRID, localClusterSizes)
		acc := params.RawVector().Data

		params = mat.VecDenseCopyOf(client.localAggregatedParams.lrGrid)
		params.ScaleVec(SCALEGRID, params)
		lr := params.RawVector().Data

		for _, x := range lr {
			acc = append(acc, x)
		}

		params = mat.VecDenseCopyOf(client.localAggregatedParams.momGrid)
		params.ScaleVec(SCALEGRID, params)
		mom := params.RawVector().Data

		for _, x := range mom {
			acc = append(acc, x)
		}

		duplicate := append([]float64{}, localClusterSizes.RawVector().Data...)
		duplicate = append(duplicate, localClusterSizes.RawVector().Data...)
		duplicate = append(duplicate, localClusterSizes.RawVector().Data...)

		client.MHEParams.InputOne = acc
		client.MHEParams.InputTwo = duplicate
		return
	}

	switch input {
	case ENC_DENSE:
		var scaledMatrix mat.Dense
		scaledMatrix.Scale(SCALEGRID, client.grid)
		client.MHEParams.InputOne = scaledMatrix.RawMatrix().Data
		return
	case ENC_ACC:
		params = mat.VecDenseCopyOf(client.localAggregatedParams.accGrid)
		localClusterSizes = mat.VecDenseCopyOf(client.localAggregatedParams.clusterSizeGrid)
	case ENC_LR:
		sortedLRs := ExtractVecElements(client.localAggregatedParams.lrGrid, client.SortedIdxMapping[:client.NumBestParams])
		localClusterSizes = ExtractVecElements(client.localAggregatedParams.clusterSizeGrid, client.SortedIdxMapping[:client.NumBestParams])
		params = mat.VecDenseCopyOf(sortedLRs)
	case ENC_MOM:
		sortedMOMs := ExtractVecElements(client.localAggregatedParams.momGrid, client.SortedIdxMapping[:client.NumBestParams])
		localClusterSizes = ExtractVecElements(client.localAggregatedParams.clusterSizeGrid, client.SortedIdxMapping[:client.NumBestParams])
		params = mat.VecDenseCopyOf(sortedMOMs)
	}

	params.ScaleVec(SCALEGRID, params)
	localClusterSizes.ScaleVec(SCALEGRID, localClusterSizes)

	client.MHEParams.InputOne = params.RawVector().Data
	client.MHEParams.InputTwo = localClusterSizes.RawVector().Data
}

// SaveBestAcc saves the best accuracies and creates an indices array to map the given accuracies array to the best accuracies array.
// The best accuracies array is restricted with the top numBestParams values.
func (client *Client) SaveBestAcc(accuracies []float64, numBestParams int) (*mat.VecDense, []int) {
	// Deep copy the given slice/array into a vector.
	avgAcc := mat.NewVecDense(len(accuracies), append(make([]float64, 0, len(accuracies)), accuracies...))

	// Sort the avgAcc vector in descending order
	idxs := make([]int, avgAcc.Len()) // How idxs are sorted
	floats.Argsort(avgAcc.RawVector().Data, idxs)
	floats.Reverse(avgAcc.RawVector().Data)
	ReverseIntArray(idxs)

	// Save the best accuracy values and the mapping of indices.
	if client.globalBestParams == nil {
		client.globalBestParams = &GridData{}
	}
	client.globalBestParams.accGrid = mat.NewVecDense(avgAcc.Len(), avgAcc.RawVector().Data)
	client.SortedIdxMapping = idxs

	//fmt.Println("Avg ACC final:")
	//printVec(c.bestAccs)
	//fmt.Println(c.sortedIdxMapping)

	if numBestParams > len(accuracies) {
		numBestParams = len(accuracies)
	}
	client.NumBestParams = numBestParams

	return client.globalBestParams.accGrid, client.SortedIdxMapping
}

// SaveBestClusterSize sorts the given clusterSize vector according to the given indices mapping
// and saves the top numBestParams values of the sorted vector.
func (client *Client) SaveBestClusterSize(clusterSize *mat.VecDense, numBestParams int) *mat.VecDense {
	if numBestParams > clusterSize.Len() {
		numBestParams = clusterSize.Len()
	}
	client.globalBestParams.clusterSizeGrid = ExtractVecElements(clusterSize, client.SortedIdxMapping[:numBestParams])
	client.NumBestParams = numBestParams

	return client.globalBestParams.clusterSizeGrid
}

func (client *Client) GetClusterSizes() *mat.VecDense {
	return client.globalBestParams.clusterSizeGrid
}

func (client *Client) GetIdxMapping() []int {
	return client.SortedIdxMapping
}

// SaveBestLrAndMom saves the given LR and MOM vectors as best ones.
func (client *Client) SaveBestLrAndMom(bestLr, bestMom *mat.VecDense) {
	client.globalBestParams.lrGrid = bestLr
	client.globalBestParams.momGrid = bestMom
}

// UnscaleParameters performs the rescaling of the LR/MOM parameters.
func (client *Client) UnscaleParameters() {
	client.rescaledLRs = client.unscaleParameter(client.globalBestParams.lrGrid, MAXLR, MINLR)
	client.rescaledMOMs = client.unscaleParameter(client.globalBestParams.momGrid, MAXMOM, MINMOM)
}

// unscaleParameter rescales a given parameter vector given the original max and min possible value of the parameters.
// The initial scaling was MinMax scaling and this function undoes the effects of MinMax scaler.
// MinMax scaler: (param_original - MinValue) / (MaxValue - MinValue) = param_current
// unscaleParameter performs: param_current * (MaxValue - MinValue) + MinValue = param_original
func (client *Client) unscaleParameter(param *mat.VecDense, maxValue, minValue float64) *mat.VecDense {
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

// CreateEmptyClient creates a dummy client with ID = -1 and empty fields.
func (client *Client) CreateEmptyClient() {
	client.ID = -1
	client.grid = mat.NewDense(GRIDROWS, GRIDCOLS, nil)
	client.localAggregatedParams = &GridData{}
	client.MHEParams = &EncryptionParams{}
}

// PrintFinalHPs prints the un-scaled LR/MOM values that were computed as DBSCAN final outputs.
func (client *Client) PrintFinalHPs() {
	fmt.Println("Avg LR un-scaled:")
	PrintVec(client.rescaledLRs)
	fmt.Println("Avg MOM un-scaled:")
	PrintVec(client.rescaledMOMs)
}
