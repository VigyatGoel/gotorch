package layer

import "gorgonia.org/tensor"

type Layer interface {
	Forward(input *tensor.Dense) *tensor.Dense
	Backward(dout *tensor.Dense) *tensor.Dense

	GetWeights() *tensor.Dense
	GetGradients() *tensor.Dense
	UpdateWeights(weightsUpdate *tensor.Dense)

	GetBiases() *tensor.Dense
	GetBiasGradients() *tensor.Dense
	UpdateBiases(biasUpdate *tensor.Dense)
}
