package layer

import (
	"gorgonia.org/tensor"
)

// LeakyReLU implements Leaky ReLU activation: f(x) = x if x > 0, else alpha * x
type LeakyReLU struct {
	Alpha  float64       // negative slope coefficient
	input  *tensor.Dense // cached input for gradient computation
	output *tensor.Dense // cached output
}

// NewLeakyReLU creates a new Leaky ReLU activation layer
func NewLeakyReLU(alpha float64) *LeakyReLU {
	return &LeakyReLU{
		Alpha: alpha,
	}
}

// Forward applies Leaky ReLU activation element-wise
func (l *LeakyReLU) Forward(x *tensor.Dense) *tensor.Dense {
	l.input = x
	l.output = x.Clone().(*tensor.Dense)
	data := l.output.Data().([]float64)
	for i, v := range data {
		if v < 0 {
			data[i] = v * l.Alpha
		}
	}
	return l.output
}

// Backward computes Leaky ReLU gradient: 1 if input > 0, else alpha
func (l *LeakyReLU) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	grad := l.input.Clone().(*tensor.Dense)
	inputData := grad.Data().([]float64)
	for i, v := range inputData {
		if v > 0 {
			inputData[i] = 1.0
		} else {
			inputData[i] = l.Alpha
		}
	}

	result, _ := tensor.Mul(grad, gradOutput)
	return result.(*tensor.Dense)
}

func (l *LeakyReLU) GetWeights() *tensor.Dense                 { return nil }
func (l *LeakyReLU) GetGradients() *tensor.Dense               { return nil }
func (l *LeakyReLU) UpdateWeights(weightsUpdate *tensor.Dense) {}
func (l *LeakyReLU) GetBiases() *tensor.Dense                  { return nil }
func (l *LeakyReLU) GetBiasGradients() *tensor.Dense           { return nil }
func (l *LeakyReLU) UpdateBiases(biasUpdate *tensor.Dense)     {}

// ClearCache releases cached tensors to prevent memory leaks
func (l *LeakyReLU) ClearCache() {
	l.input = nil
	l.output = nil
}
