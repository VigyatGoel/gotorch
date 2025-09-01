package loss

import "gorgonia.org/tensor"

// MSELoss implements Mean Squared Error loss for regression tasks
type MSELoss struct {
	Predictions *tensor.Dense // cached predictions for gradient computation
	Targets     *tensor.Dense // cached target values
}

// NewMSELoss creates a new mean squared error loss function
func NewMSELoss() *MSELoss {
	return &MSELoss{}
}

// Forward computes MSE loss: mean((predictions - targets)^2)
func (l *MSELoss) Forward(predictions, targets *tensor.Dense) float64 {
	l.Predictions = predictions
	l.Targets = targets

	predData := predictions.Data().([]float64)
	targetData := targets.Data().([]float64)

	sum := 0.0
	for i := range predData {
		diff := predData[i] - targetData[i]
		sum += diff * diff
	}

	totalElements := len(predData)
	return sum / float64(totalElements)
}

// Backward computes gradient: 2 * (predictions - targets) / total_elements
func (l *MSELoss) Backward() *tensor.Dense {
	predData := l.Predictions.Data().([]float64)
	targetData := l.Targets.Data().([]float64)

	gradData := make([]float64, len(predData))
	for i := range predData {
		gradData[i] = 2 * (predData[i] - targetData[i])
	}

	shape := l.Predictions.Shape()
	totalElements := len(gradData)
	for i := range gradData {
		gradData[i] /= float64(totalElements)
	}

	grad := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(gradData))
	return grad
}
