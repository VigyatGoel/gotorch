package loss

import (
	"math"

	"gorgonia.org/tensor"
)

type CrossEntropyLoss struct {
	Predictions *tensor.Dense
	Targets     *tensor.Dense
}

func NewCrossEntropyLoss() *CrossEntropyLoss {
	return &CrossEntropyLoss{}
}

func (l *CrossEntropyLoss) Forward(predictions, targets *tensor.Dense) float64 {
	l.Predictions = predictions
	l.Targets = targets

	// Calculate cross-entropy loss: -sum(target * log(prediction))
	predData := predictions.Data().([]float64)
	targetData := targets.Data().([]float64)

	loss := 0.0
	for i := range predData {
		loss -= targetData[i] * math.Log(predData[i]+1e-9)
	}

	shape := predictions.Shape()
	rows := shape[0]
	return loss / float64(rows)
}

func (l *CrossEntropyLoss) Backward() *tensor.Dense {
	// Gradient of cross-entropy loss with softmax: predictions - targets
	// This assumes that the last layer is a softmax activation
	predData := l.Predictions.Data().([]float64)
	targetData := l.Targets.Data().([]float64)

	gradData := make([]float64, len(predData))
	for i := range predData {
		gradData[i] = (predData[i] - targetData[i])
	}

	shape := l.Predictions.Shape()
	rows := shape[0]
	for i := range gradData {
		gradData[i] /= float64(rows)
	}

	grad := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(gradData))
	return grad
}
