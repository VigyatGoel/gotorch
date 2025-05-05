package layer

import (
	"github.com/VigyatGoel/gotorch/utils"
	"gonum.org/v1/gonum/mat"
)

type ReLU struct {
	input *mat.Dense
}

func NewReLU() *ReLU {
	return &ReLU{}
}

func (r *ReLU) Forward(x *mat.Dense) *mat.Dense {
	r.input = x
	return utils.ApplyFuncDense(x, utils.ReLU)
}

func (r *ReLU) Backward(gradOutput *mat.Dense) *mat.Dense {
	rows, cols := r.input.Dims()
	grad := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if r.input.At(i, j) > 0 {
				grad.Set(i, j, gradOutput.At(i, j))
			}
		}
	}
	return grad
}

func (r *ReLU) GetWeights() *mat.Dense                 { return nil }
func (r *ReLU) GetGradients() *mat.Dense               { return nil }
func (r *ReLU) UpdateWeights(weightsUpdate *mat.Dense) {}
func (r *ReLU) GetBiases() *mat.Dense                  { return nil }
func (r *ReLU) GetBiasGradients() *mat.Dense           { return nil }
func (r *ReLU) UpdateBiases(biasUpdate *mat.Dense)     {}
