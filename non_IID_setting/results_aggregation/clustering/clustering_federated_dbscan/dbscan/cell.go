package dbscan

// Cell is defined by two integer values - row and column.
type Cell struct {
	row, col int
}

// GetAdjacentCells returns the adjacent cells (i.e., top + bottom + left + right) for a given cell.
func (cell Cell) GetAdjacentCells() []Cell {
	adjCells := make([]Cell, 0)

	offsetArray := [][]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	for _, offset := range ShuffleArray(offsetArray) {
		// Calculate adjacent cell
		adjCell := Cell{cell.row + offset[0], cell.col + offset[1]}

		// Check if the adjacent cell is within the bounds of the grid
		if adjCell.row < 0 || adjCell.row >= GRIDROWS || adjCell.col < 0 || adjCell.col >= GRIDCOLS {
			continue
		}

		adjCells = append(adjCells, adjCell)
	}

	return adjCells
}
