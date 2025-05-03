package layer

import (
	"github.com/VigyatGoel/gotorch/src/utils"
	"math/rand"
)

var rng = rand.New(rand.NewSource(42))

type Linear struct {
	Input  [][]float64
	Weight [][]float64
	Bias   [][]float64
}

func NewLinear(inFeautes, outFeatures int) *Linear {
	weights := make([][]float64, inFeautes)
	bias := make([][]float64, 1)
	bias[0] = make([]float64, outFeatures)

	for i := range weights {
		weights[i] = make([]float64, outFeatures)
		for j := range weights[i] {
			weights[i][j] = rng.Float64()*2 - 1
		}
	}

	return &Linear{
		Weight: weights,
		Bias:   bias,
	}
}

func (l *Linear) Forward(x [][]float64) [][]float64 {
	l.Input = x
	out := utils.Dot(x, l.Weight)
	out = utils.Add(out, l.Bias)
	return out
}

func (l *Linear) Backward(gradOutput [][]float64, lr float64) [][]float64 {
	inputT := utils.Transpose(l.Input)
	dWeight := utils.Dot(inputT, gradOutput)

	dBias := make([][]float64, 1)
	dBias[0] = make([]float64, len(gradOutput[0]))
	for i := range gradOutput {
		for j := range gradOutput[0] {
			dBias[0][j] += gradOutput[i][j]
		}
	}

	weightT := utils.Transpose(l.Weight)
	gradInput := utils.Dot(gradOutput, weightT)

	for i := range l.Weight {
		for j := range l.Weight[0] {
			l.Weight[i][j] -= lr * dWeight[i][j]
		}
	}
	for j := range l.Bias[0] {
		l.Bias[0][j] -= lr * dBias[0][j]
	}

	return gradInput
}
