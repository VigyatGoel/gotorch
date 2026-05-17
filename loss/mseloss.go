package loss

import "gonum.org/v1/gonum/mat"

// MSELoss implements Mean Squared Error loss for regression tasks
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
	predData := predictions.RawMatrix().Data
	targetData := targets.RawMatrix().Data
	sum := 0.0
	for i := range predData {
		diff := predData[i] - targetData[i]
		sum += diff * diff
	}
	return sum / float64(len(predData))
}

func (l *MSELoss) Backward() *mat.Dense {
	predData := l.Predictions.RawMatrix().Data
	targetData := l.Targets.RawMatrix().Data
	rows, cols := l.Predictions.Dims()
	n := float64(len(predData))
	gradData := make([]float64, len(predData))
	for i := range predData {
		gradData[i] = 2 * (predData[i] - targetData[i]) / n
	}
	return mat.NewDense(rows, cols, gradData)
}
