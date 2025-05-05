package optimizer

import "gonum.org/v1/gonum/mat"

type SGD struct {
	LR float64
}

func NewSGD(lr float64) *SGD {
	return &SGD{LR: lr}
}

func (sgd *SGD) Step(weights *mat.Dense, gradients *mat.Dense) *mat.Dense {
	rows, cols := weights.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			weights.Set(i, j, weights.At(i, j)-sgd.LR*gradients.At(i, j))
		}
	}
	return weights
}

func (sgd *SGD) StepBias(biases *mat.Dense, biasGradients *mat.Dense) *mat.Dense {
	if biases == nil || biasGradients == nil {
		return biases
	}
	rows, cols := biases.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			biases.Set(i, j, biases.At(i, j)-sgd.LR*biasGradients.At(i, j))
		}
	}
	return biases
}

func (sgd *SGD) ZeroGrad() {}

func (sgd *SGD) GetLearningRate() float64 {
	return sgd.LR
}
