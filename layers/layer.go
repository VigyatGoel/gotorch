package layer

type Layer interface {
	Forward(input [][]float64) [][]float64
	Backward(dout [][]float64, learningRate float64) [][]float64
}
