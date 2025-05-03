package layer

import "github.com/VigyatGoel/gotorch/utils"

type Sigmoid struct {
	Output [][]float64
}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (s *Sigmoid) Forward(x [][]float64) [][]float64 {
	s.Output = utils.ApplyFunc(x, utils.Sigmoid)
	return s.Output
}

func (s *Sigmoid) Backward(gradOutput [][]float64) [][]float64 {
	grad := utils.Zeros(len(s.Output), len(s.Output[0]))
	for i := range s.Output {
		for j := range s.Output[0] {
			grad[i][j] = gradOutput[i][j] * s.Output[i][j] * (1 - s.Output[i][j])
		}
	}
	return grad
}

func (s *Sigmoid) GetWeights() [][]float64 {
	return nil
}

func (s *Sigmoid) GetGradients() [][]float64 {
	return nil
}

func (s *Sigmoid) UpdateWeights(weightsUpdate [][]float64) {
}

func (s *Sigmoid) GetBiases() [][]float64 {
	return nil
}

func (s *Sigmoid) GetBiasGradients() [][]float64 {
	return nil
}

func (s *Sigmoid) UpdateBiases(biasUpdate [][]float64) {
}
