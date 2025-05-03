package optimizer

type Optimizer interface {
	Step(weights [][]float64, gradients [][]float64) [][]float64
	StepBias(biases [][]float64, biasGradients [][]float64) [][]float64
	ZeroGrad()
	GetLearningRate() float64
}
