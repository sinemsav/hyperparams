package dbscan

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"time"
)

const (
	PRECISION = 3
)

// ShuffleArray randomly shuffles a given array.
func ShuffleArray(arr [][]int) [][]int {
	rand.Seed(time.Now().UnixNano()) // Set seed for random number generator

	// Randomly shuffle the array
	for i := range arr {
		j := rand.Intn(i + 1)
		arr[i], arr[j] = arr[j], arr[i]
	}

	return arr
}

// PrintMat prints a matrix.
func PrintMat(matrix *mat.Dense) {
	fc := mat.Formatted(matrix, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v \n\n", fc)
}

// PrintVec print a vector.
func PrintVec(vector *mat.VecDense) {
	fc := mat.Formatted(vector, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v \n\n", fc)
}

// ApplyDenseMaskApprox rounds matrix values to 1 if they are bigger than 0.4, else rounds to 0.
func ApplyDenseMaskApprox(i, j int, v float64) float64 {
	if v >= 0.4 {
		return 1
	}
	return 0
}

// ReverseIntArray reverses the values of an integer array.
func ReverseIntArray(array []int) []int {
	for i, j := 0, len(array)-1; i < j; i, j = i+1, j-1 {
		array[i], array[j] = array[j], array[i]
	}

	return array
}

// toFixed converts a float number to a float number with a given precision.
func toFixed(num float64, precision int) float64 {
	output := math.Pow(10, float64(precision))
	return (math.Round(num * output)) / output
}

// ComplexArrayToFloatArray converts an array of complex numbers to an array of floats by ignoring the imaginary part.
func ComplexArrayToFloatArray(x []complex128, length uint) []float64 {
	array := make([]float64, length)
	for i := 0; i < len(array); i++ {
		array[i] = toFixed(real(x[i]), PRECISION)
	}
	return array
}

// ExtractVecElements uses an int array to permute the values of a given vector.
func ExtractVecElements(vector *mat.VecDense, idxs []int) *mat.VecDense {
	retVector := mat.NewVecDense(len(idxs), nil)
	for i, elem := range idxs {
		retVector.SetVec(i, vector.AtVec(elem))
	}

	return retVector
}
