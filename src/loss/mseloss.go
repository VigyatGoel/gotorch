package loss

type MSELoss struct {
	Predictions [][]float64
	Targets     [][]float64
}

func NewMSELoss() *MSELoss {
	return &MSELoss{}
}

func (l *MSELoss) Forward(predictions, targets [][]float64) float64 {
	l.Predictions = predictions
	l.Targets = targets

	sum := 0.0
	for i := range predictions {
		for j := range predictions[0] {
			diff := predictions[i][j] - targets[i][j]
			sum += diff * diff
		}
	}
	return sum / float64(len(predictions))
}

func (l *MSELoss) Backward() [][]float64 {
	grad := make([][]float64, len(l.Predictions))
	for i := range grad {
		grad[i] = make([]float64, len(l.Predictions[0]))
		for j := range grad[0] {
			grad[i][j] = 2 * (l.Predictions[i][j] - l.Targets[i][j]) / float64(len(l.Predictions))
		}
	}
	return grad
}
