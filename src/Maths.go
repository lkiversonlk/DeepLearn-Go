package src

import (
	"math"
)
func Sigmoid(x float64) float64{
	return 1.0 / (1 + math.Pow(math.E, -1 * x))
}

func SigmoidPrime(x float64) float64{
	sigmoied := Sigmoid(x)
	return (1 - sigmoied) * sigmoied
}
