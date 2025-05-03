package loss

import (
	"github.com/VigyatGoel/gotorch/src/utils"
	"math"
)

type CrossEntropyLoss struct {
	Predictions [][]float64
	Targets     [][]float64
}

func NewCrossEntropyLoss() *CrossEntropyLoss {
	return &CrossEntropyLoss{}
}

func (l *CrossEntropyLoss) Forward(predictions, targets [][]float64) float64 {
	l.Predictions = predictions
	l.Targets = targets

	loss := 0.0
	for i := range predictions {
		for j := range predictions[0] {
			loss -= targets[i][j] * math.Log(predictions[i][j]+1e-9)
		}
	}
	return loss / float64(len(predictions))
}

func (l *CrossEntropyLoss) Backward() [][]float64 {
	grad := utils.Zeros(len(l.Predictions), len(l.Predictions[0]))
	for i := range grad {
		for j := range grad[0] {
			grad[i][j] = (l.Predictions[i][j] - l.Targets[i][j]) / float64(len(l.Predictions))
		}
	}
	return grad
}
