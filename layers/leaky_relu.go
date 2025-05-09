package layer

import (
	"github.com/VigyatGoel/gotorch/utils"
	"gonum.org/v1/gonum/mat"
)

type LeakyReLU struct {
	input *mat.Dense
	Alpha float64
}

func NewLeakyReLU(alpha float64) *LeakyReLU {
	return &LeakyReLU{Alpha: alpha}
}

func (r *LeakyReLU) Forward(x *mat.Dense) *mat.Dense {
	r.input = x
	if r.Alpha == 0 {
		r.Alpha = 0.01
	}
	return utils.ApplyFuncDense(x, func(v float64) float64 {
		return utils.LeakyReLU(v, r.Alpha)
	})
}

func (r *LeakyReLU) Backward(gradOutput *mat.Dense) *mat.Dense {
	rows, cols := r.input.Dims()
	grad := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			deriv := utils.LeakyReLUDerivative(r.input.At(i, j), r.Alpha)
			grad.Set(i, j, gradOutput.At(i, j)*deriv)
		}
	}
	return grad
}

func (r *LeakyReLU) GetWeights() *mat.Dense                 { return nil }
func (r *LeakyReLU) GetGradients() *mat.Dense               { return nil }
func (r *LeakyReLU) UpdateWeights(weightsUpdate *mat.Dense) {}
func (r *LeakyReLU) GetBiases() *mat.Dense                  { return nil }
func (r *LeakyReLU) GetBiasGradients() *mat.Dense           { return nil }
func (r *LeakyReLU) UpdateBiases(biasUpdate *mat.Dense)     {}
