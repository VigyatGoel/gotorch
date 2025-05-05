package loss

import "gonum.org/v1/gonum/mat"

type MSELoss struct {
	Predictions *mat.Dense
	Targets     *mat.Dense
}

func NewMSELoss() *MSELoss {
	return &MSELoss{}
}

func (l *MSELoss) Forward(predictions, targets *mat.Dense) float64 {
	l.Predictions = predictions
	l.Targets = targets
	sum := 0.0
	rows, cols := predictions.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			diff := predictions.At(i, j) - targets.At(i, j)
			sum += diff * diff
		}
	}
	return sum / float64(rows)
}

func (l *MSELoss) Backward() *mat.Dense {
	rows, cols := l.Predictions.Dims()
	grad := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			grad.Set(i, j, 2*(l.Predictions.At(i, j)-l.Targets.At(i, j))/float64(rows))
		}
	}
	return grad
}
