package comparator

import (
	"fmt"
	"github.com/tuneinsight/lattigo/v4/ckks"
)

// Defined by formula in 2019-1234.pdf, page 12
func GetPolynomialF(n uint) *ckks.Polynomial {

	coef_inv := 1 << uint(2*(n-1))
	coef_inv_float := float64(coef_inv)

	coefs := make([]float64, 2*(n+1))
	switch n {
	case 1:
		coefs[1] = 3.0  // Degree 1
		coefs[3] = -1.0 // Degree 3
	case 2:
		coefs[1] = 15.0  // Degree 1
		coefs[3] = -10.0 // Degree 3
		coefs[5] = 3.0   // Degree 5
	case 3:
		coefs[1] = 35.0  // Degree 1
		coefs[3] = -35.0 // Degree 3
		coefs[5] = 21.0  // Degree 5
		coefs[7] = -5.0  // Degree 7
	case 4:
		coefs[1] = 315.0  // Degree 1
		coefs[3] = -420.0 // Degree 3
		coefs[5] = 378.0  // Degree 5
		coefs[7] = -180.0 // Degree 7
		coefs[9] = 35.0   // Degree 9

	default:
		fmt.Printf("No valid n (%v) for f\n", n)
		return nil
	}

	coefs_complex := make([]complex128, 2*(n+1))
	for i := range coefs {
		coefs_complex[i] = complex(coefs[i]/float64(coef_inv_float), 0.0)
	}

	return ckks.NewPoly(coefs_complex)
}

// Defined by algo in 2019-1234.pdf, page 19
// Polynomials are defined in 2019-1234, p.21
func GetPolynomialG(n uint) *ckks.Polynomial {

	coef_inv := 1 << 10
	coef_inv_float := float64(coef_inv)

	coefs := make([]float64, 2*(n+1))
	switch n {
	case 1:
		coefs[1] = 2126.0  // Degree 1
		coefs[3] = -1359.0 // Degree 3
	case 2:
		coefs[1] = 3334.0  // Degree 1
		coefs[3] = -6108.0 // Degree 3
		coefs[5] = 3796.0  // Degree 5
	case 3:
		coefs[1] = 4589.0   // Degree 1
		coefs[3] = -16577.0 // Degree 3
		coefs[5] = 25614.0  // Degree 5
		coefs[7] = -12860.0 // Degree 7
	case 4:
		coefs[1] = 5850.0    // Degree 1
		coefs[3] = -34974.0  // Degree 3
		coefs[5] = 97015.0   // Degree 5
		coefs[7] = -113492.0 // Degree 7
		coefs[9] = 46623.0   // Degree 9
	default:
		fmt.Printf("No valid n (%v) for g\n", n)
		return nil
	}

	coefs_complex := make([]complex128, 2*(n+1))
	for i := range coefs {
		coefs_complex[i] = complex(coefs[i]/float64(coef_inv_float), 0.0)
	}

	return ckks.NewPoly(coefs_complex)
}
