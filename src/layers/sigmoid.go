package layer

import "github.com/VigyatGoel/gotorch/src/utils"

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

func (s *Sigmoid) Backward(dout [][]float64, _ float64) [][]float64 {
	grad := utils.Zeros(len(s.Output), len(s.Output[0]))
	for i := range s.Output {
		for j := range s.Output[0] {
			grad[i][j] = dout[i][j] * s.Output[i][j] * (1 - s.Output[i][j])
		}
	}
	return grad
}
