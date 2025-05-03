package optimizer

type SGD struct {
	LR float64
}

func NewSGD(lr float64) *SGD {
	return &SGD{
		LR: lr,
	}
}

func (sgd *SGD) Step(weights [][]float64, gradients [][]float64) [][]float64 {
	for i := range weights {
		for j := range weights[i] {
			weights[i][j] -= sgd.LR * gradients[i][j]
		}
	}
	return weights
}

func (sgd *SGD) StepBias(biases [][]float64, biasGradients [][]float64) [][]float64 {
	if len(biases) == 0 || len(biasGradients) == 0 || len(biases[0]) == 0 || len(biasGradients[0]) == 0 {
		return biases
	}
	for j := range biases[0] {
		biases[0][j] -= sgd.LR * biasGradients[0][j]
	}
	return biases
}

func (sgd *SGD) ZeroGrad() {
}

func (sgd *SGD) GetLearningRate() float64 {
	return sgd.LR
}
