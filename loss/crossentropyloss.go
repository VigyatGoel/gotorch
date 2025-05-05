package loss

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

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
	loss := 0.0
	rows, cols := predictions.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			loss -= targets.At(i, j) * math.Log(predictions.At(i, j)+1e-9)
		}
	}
	return loss / float64(rows)
}

func (l *CrossEntropyLoss) Backward() *mat.Dense {
	rows, cols := l.Predictions.Dims()
	grad := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			grad.Set(i, j, (l.Predictions.At(i, j)-l.Targets.At(i, j))/float64(rows))
		}
	}
	return grad
}
