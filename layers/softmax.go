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

func (s *Softmax) Backward(dout [][]float64, _ float64) [][]float64 {
	return dout
}
