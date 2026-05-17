package optimizer

import "gonum.org/v1/gonum/mat"

type Optimizer interface {
	Step(weights *mat.Dense, gradients *mat.Dense) *mat.Dense
	StepBias(biases *mat.Dense, biasGradients *mat.Dense) *mat.Dense
	ZeroGrad()
	GetLearningRate() float64
}
