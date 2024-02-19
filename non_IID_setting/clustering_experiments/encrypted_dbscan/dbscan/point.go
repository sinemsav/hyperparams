package dbscan

import (
	"math"
)

// Point is defined by 2 float numbers - the X and Y coordinate.
type Point struct {
	X, Y float64
}

// ToCell maps a point to a cell (discretization) where L is the cell size.
func (point *Point) ToCell(L float64) Cell {
	i := int(math.Floor(point.X / L))
	j := int(math.Floor(point.Y / L))
	return Cell{i, j}
}

// FindClosestAdjacentCell returns the closest (Euclidean distance) adjacent cell for a given point.
func (point *Point) FindClosestAdjacentCell() Cell {
	// Get the cell the point belongs to and all adjacent cells
	cell := point.ToCell(DBSCAN_L)
	adjCells := cell.GetAdjacentCells()

	// Initialize the minimum distance and closest Cell
	minDist := math.MaxFloat64
	closestCell := Cell{-1, -1}

	// Check the adjacent cells for the closest one
	for _, adjCell := range adjCells {
		// Calculate the distance between the point and the center of the adjacent cell
		adjCellMidPoint := Point{float64(adjCell.row)*DBSCAN_L + DBSCAN_L/2, float64(adjCell.col)*DBSCAN_L + DBSCAN_L/2}
		dist := distance(*point, adjCellMidPoint)

		// If the distance is smaller than the current minimum, update the minimum and the index of the closest cell
		if dist < minDist {
			minDist = dist
			closestCell = adjCell
		}
	}

	// Return the index of the closest adjacent cell
	return closestCell
}

// Function to calculate the Euclidean distance between two points
func distance(p1 Point, p2 Point) float64 {
	return math.Sqrt(math.Pow(p2.X-p1.X, 2) + math.Pow(p2.Y-p1.Y, 2))
}
