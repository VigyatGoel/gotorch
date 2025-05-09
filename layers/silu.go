package layer

import (
	"github.com/VigyatGoel/gotorch/utils"
	"gonum.org/v1/gonum/mat"
)

type SiLU struct {
	output *mat.Dense
}

func NewSiLU() *SiLU {
	return &SiLU{}
}

func (s *SiLU) Forward(x *mat.Dense) *mat.Dense {
	s.output = utils.ApplyFuncDense(x, utils.SiLU)
	return s.output
}

func (s *SiLU) Backward(gradOutput *mat.Dense) *mat.Dense {
	rows, cols := s.output.Dims()
	grad := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			deriv := utils.SiLUDerivative(s.output.At(i, j))
			grad.Set(i, j, gradOutput.At(i, j)*deriv)
		}
	}
	return grad
}

func (s *SiLU) GetWeights() *mat.Dense                 { return nil }
func (s *SiLU) GetGradients() *mat.Dense               { return nil }
func (s *SiLU) UpdateWeights(weightsUpdate *mat.Dense) {}
func (s *SiLU) GetBiases() *mat.Dense                  { return nil }
func (s *SiLU) GetBiasGradients() *mat.Dense           { return nil }
func (s *SiLU) UpdateBiases(biasUpdate *mat.Dense)     {}
