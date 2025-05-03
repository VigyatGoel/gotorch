package layer

import (
	"github.com/VigyatGoel/gotorch/src/utils"
)

type ReLU struct {
	Input [][]float64
}

func NewReLU() *ReLU {
	return &ReLU{}
}

func (r *ReLU) Forward(x [][]float64) [][]float64 {
	r.Input = x
	return utils.ApplyFunc(x, utils.ReLU)
}

func (r *ReLU) Backward(dout [][]float64, _ float64) [][]float64 {
	grad := utils.Zeros(len(r.Input), len(r.Input[0]))
	for i := range r.Input {
		for j := range r.Input[0] {
			if r.Input[i][j] > 0 {
				grad[i][j] = dout[i][j]
			}
		}
	}
	return grad
}
