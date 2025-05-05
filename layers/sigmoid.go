package layer

import (
	"github.com/VigyatGoel/gotorch/utils"
	"gonum.org/v1/gonum/mat"
)

type Sigmoid struct {
	output *mat.Dense
}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (s *Sigmoid) Forward(x *mat.Dense) *mat.Dense {
	s.output = utils.ApplyFuncDense(x, utils.Sigmoid)
	return s.output
}

func (s *Sigmoid) Backward(gradOutput *mat.Dense) *mat.Dense {
	rows, cols := s.output.Dims()
	grad := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := s.output.At(i, j)
			grad.Set(i, j, gradOutput.At(i, j)*val*(1-val))
		}
	}
	return grad
}

func (s *Sigmoid) GetWeights() *mat.Dense                 { return nil }
func (s *Sigmoid) GetGradients() *mat.Dense               { return nil }
func (s *Sigmoid) UpdateWeights(weightsUpdate *mat.Dense) {}
func (s *Sigmoid) GetBiases() *mat.Dense                  { return nil }
func (s *Sigmoid) GetBiasGradients() *mat.Dense           { return nil }
func (s *Sigmoid) UpdateBiases(biasUpdate *mat.Dense)     {}
