package layer

import "gonum.org/v1/gonum/mat"

// Layer defines the interface that all neural network layers must implement
type Layer interface {
	Forward(input *mat.Dense) *mat.Dense
	Backward(dout *mat.Dense) *mat.Dense

	GetWeights() *mat.Dense
	GetGradients() *mat.Dense
	UpdateWeights(weightsUpdate *mat.Dense)

	GetBiases() *mat.Dense
	GetBiasGradients() *mat.Dense
	UpdateBiases(biasUpdate *mat.Dense)

	ClearCache()
}
