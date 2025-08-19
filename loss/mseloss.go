package loss

import "gorgonia.org/tensor"

type MSELoss struct {
	Predictions *tensor.Dense
	Targets     *tensor.Dense
}

func NewMSELoss() *MSELoss {
	return &MSELoss{}
}

func (l *MSELoss) Forward(predictions, targets *tensor.Dense) float64 {
	l.Predictions = predictions
	l.Targets = targets
	
	// Calculate MSE: mean((predictions - targets)^2)
	predData := predictions.Data().([]float64)
	targetData := targets.Data().([]float64)
	
	sum := 0.0
	for i := range predData {
		diff := predData[i] - targetData[i]
		sum += diff * diff
	}
	
	shape := predictions.Shape()
	rows := shape[0]
	return sum / float64(rows)
}

func (l *MSELoss) Backward() *tensor.Dense {
	// Gradient of MSE: 2 * (predictions - targets) / rows
	predData := l.Predictions.Data().([]float64)
	targetData := l.Targets.Data().([]float64)
	
	gradData := make([]float64, len(predData))
	for i := range predData {
		gradData[i] = 2 * (predData[i] - targetData[i])
	}
	
	shape := l.Predictions.Shape()
	rows := shape[0]
	for i := range gradData {
		gradData[i] /= float64(rows)
	}
	
	grad := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(gradData))
	return grad
}
