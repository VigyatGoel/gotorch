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
	wData := weights.RawMatrix().Data
	gData := gradients.RawMatrix().Data
	result := make([]float64, rows*cols)
	for i := range result {
		result[i] = wData[i] - sgd.LR*gData[i]
	}
	return mat.NewDense(rows, cols, result)
}

func (sgd *SGD) StepBias(biases *mat.Dense, biasGradients *mat.Dense) *mat.Dense {
	if biases == nil || biasGradients == nil {
		return biases
	}
	return sgd.Step(biases, biasGradients)
}

func (sgd *SGD) ZeroGrad() {}

func (sgd *SGD) GetLearningRate() float64 {
	return sgd.LR
}
