package loss

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// CrossEntropyLoss implements cross-entropy loss for multi-class classification
type CrossEntropyLoss struct {
	Predictions *mat.Dense
	Targets     *mat.Dense
}

func NewCrossEntropyLoss() *CrossEntropyLoss {
	return &CrossEntropyLoss{}
}

func (l *CrossEntropyLoss) Forward(predictions, targets *mat.Dense) float64 {
	l.Predictions = predictions
	l.Targets = targets
	predData := predictions.RawMatrix().Data
	targetData := targets.RawMatrix().Data
	loss := 0.0
	for i := range predData {
		loss -= targetData[i] * math.Log(predData[i]+1e-9)
	}
	rows, _ := predictions.Dims()
	return loss / float64(rows)
}

func (l *CrossEntropyLoss) Backward() *mat.Dense {
	predData := l.Predictions.RawMatrix().Data
	targetData := l.Targets.RawMatrix().Data
	rows, cols := l.Predictions.Dims()
	gradData := make([]float64, len(predData))
	for i := range predData {
		gradData[i] = (predData[i] - targetData[i]) / float64(rows)
	}
	return mat.NewDense(rows, cols, gradData)
}
