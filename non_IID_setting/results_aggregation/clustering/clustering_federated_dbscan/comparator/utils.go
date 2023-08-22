package comparator

import "math"

func round(x complex128) complex128 {
	var factor float64 = 100000000
	a := math.Round(real(x)*factor) / factor
	b := math.Round(imag(x)*factor) / factor
	return complex(a, b)
}

func processBeforePrint(x complex128) complex128 {
	return round(x)
}
