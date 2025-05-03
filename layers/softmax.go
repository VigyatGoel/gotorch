package layer

import "github.com/VigyatGoel/gotorch/utils"

type Softmax struct {
	Output [][]float64
}

func NewSoftmax() *Softmax {
	return &Softmax{}
}

func (s *Softmax) Forward(x [][]float64) [][]float64 {
	s.Output = utils.Softmax(x)
	return s.Output
}

func (s *Softmax) Backward(gradOutput [][]float64) [][]float64 {
	return gradOutput
}

func (s *Softmax) GetWeights() [][]float64 {
	return nil
}

func (s *Softmax) GetGradients() [][]float64 {
	return nil
}

func (s *Softmax) UpdateWeights(weightsUpdate [][]float64) {
}

func (s *Softmax) GetBiases() [][]float64 {
	return nil
}

func (s *Softmax) GetBiasGradients() [][]float64 {
	return nil
}

func (s *Softmax) UpdateBiases(biasUpdate [][]float64) {
}
