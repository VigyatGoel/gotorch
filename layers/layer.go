package layer

type Layer interface {
	Forward(input [][]float64) [][]float64
	Backward(dout [][]float64) [][]float64

	GetWeights() [][]float64
	GetGradients() [][]float64
	UpdateWeights(weightsUpdate [][]float64)

	GetBiases() [][]float64
	GetBiasGradients() [][]float64
	UpdateBiases(biasUpdate [][]float64)
}
