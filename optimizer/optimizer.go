package optimizer

import "gorgonia.org/tensor"

type Optimizer interface {
	Step(weights *tensor.Dense, gradients *tensor.Dense) *tensor.Dense
	StepBias(biases *tensor.Dense, biasGradients *tensor.Dense) *tensor.Dense
	ZeroGrad()
	GetLearningRate() float64
}
