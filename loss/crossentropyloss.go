package loss

import (
	"math"

	"gorgonia.org/tensor"
)

// CrossEntropyLoss implements cross-entropy loss for multi-class classification
type CrossEntropyLoss struct {
	Predictions *tensor.Dense // cached predictions for gradient computation
	Targets     *tensor.Dense // cached targets (one-hot encoded)
}

// NewCrossEntropyLoss creates a new cross-entropy loss function
func NewCrossEntropyLoss() *CrossEntropyLoss {
	return &CrossEntropyLoss{}
}

// Forward computes cross-entropy loss: -mean(sum(target * log(prediction)))
func (l *CrossEntropyLoss) Forward(predictions, targets *tensor.Dense) float64 {
	l.Predictions = predictions
	l.Targets = targets

	predData := predictions.Data().([]float64)
	targetData := targets.Data().([]float64)

	loss := 0.0
	for i := range predData {
		loss -= targetData[i] * math.Log(predData[i]+1e-9) // add epsilon for numerical stability
	}

	shape := predictions.Shape()
	rows := shape[0]
	return loss / float64(rows)
}

// Backward computes gradient: (predictions - targets) / batch_size
func (l *CrossEntropyLoss) Backward() *tensor.Dense {
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
