package layer

import (
	"math/rand"

	"github.com/VigyatGoel/gotorch/utils"
)

var rng = rand.New(rand.NewSource(42))

type Linear struct {
	Input   [][]float64
	Weight  [][]float64
	Bias    [][]float64
	dWeight [][]float64
	dBias   [][]float64
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

	dWeights := make([][]float64, inFeautes)
	dBias := make([][]float64, 1)
	dBias[0] = make([]float64, outFeatures)

	for i := range dWeights {
		dWeights[i] = make([]float64, outFeatures)
	}

	return &Linear{
		Weight:  weights,
		Bias:    bias,
		dWeight: dWeights,
		dBias:   dBias,
	}
}

func (l *Linear) Forward(x [][]float64) [][]float64 {
	l.Input = x
	out := utils.Dot(x, l.Weight)
	out = utils.Add(out, l.Bias)
	return out
}

func (l *Linear) Backward(gradOutput [][]float64) [][]float64 { // Removed lr
	inputT := utils.Transpose(l.Input)
	l.dWeight = utils.Dot(inputT, gradOutput)

	l.dBias = make([][]float64, 1)
	l.dBias[0] = make([]float64, len(gradOutput[0]))
	for i := range gradOutput {
		for j := range gradOutput[0] {
			l.dBias[0][j] += gradOutput[i][j]
		}
	}

	weightT := utils.Transpose(l.Weight)
	gradInput := utils.Dot(gradOutput, weightT)

	return gradInput
}

func (l *Linear) GetWeights() [][]float64 {
	return l.Weight
}

func (l *Linear) GetGradients() [][]float64 {
	return l.dWeight
}

func (l *Linear) UpdateWeights(weightsUpdate [][]float64) {
	l.Weight = weightsUpdate
}

func (l *Linear) GetBiases() [][]float64 {
	return l.Bias
}

func (l *Linear) GetBiasGradients() [][]float64 {
	return l.dBias
}

func (l *Linear) UpdateBiases(biasUpdate [][]float64) {
	l.Bias = biasUpdate
}
