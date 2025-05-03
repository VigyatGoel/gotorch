package layer

import (
	"github.com/VigyatGoel/gotorch/utils"
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

func (r *ReLU) Backward(gradOutput [][]float64) [][]float64 {
	grad := utils.Zeros(len(r.Input), len(r.Input[0]))
	for i := range r.Input {
		for j := range r.Input[0] {
			if r.Input[i][j] > 0 {
				grad[i][j] = gradOutput[i][j]
			}
		}
	}
	return grad
}

func (r *ReLU) GetWeights() [][]float64 {
	return nil
}

func (r *ReLU) GetGradients() [][]float64 {
	return nil
}

func (r *ReLU) UpdateWeights(weightsUpdate [][]float64) {}

func (r *ReLU) GetBiases() [][]float64 {
	return nil
}

func (r *ReLU) GetBiasGradients() [][]float64 {
	return nil
}

func (r *ReLU) UpdateBiases(biasUpdate [][]float64) {}
