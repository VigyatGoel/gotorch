package layer

import (
	"gorgonia.org/tensor"
)

// ReLU implements Rectified Linear Unit activation: f(x) = max(0, x)
type ReLU struct {
	input *tensor.Dense // cached input for gradient computation
}

// NewReLU creates a new ReLU activation layer
func NewReLU() *ReLU {
	return &ReLU{}
}

// Forward applies ReLU activation element-wise
func (r *ReLU) Forward(x *tensor.Dense) *tensor.Dense {
	r.input = x
	result := x.Clone().(*tensor.Dense)
	data := result.Data().([]float64)
	for i, v := range data {
		if v < 0 {
			data[i] = 0
		}
	}
	return result
}

// Backward computes ReLU gradient: 1 if input > 0, else 0
func (r *ReLU) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	grad := r.input.Clone().(*tensor.Dense)
	inputData := grad.Data().([]float64)
	for i, v := range inputData {
		if v > 0 {
			inputData[i] = 1.0
		} else {
			inputData[i] = 0.0
		}
	}

	result, _ := tensor.Mul(grad, gradOutput)
	return result.(*tensor.Dense)
}

func (r *ReLU) GetWeights() *tensor.Dense                 { return nil }
func (r *ReLU) GetGradients() *tensor.Dense               { return nil }
func (r *ReLU) UpdateWeights(weightsUpdate *tensor.Dense) {}
func (r *ReLU) GetBiases() *tensor.Dense                  { return nil }
func (r *ReLU) GetBiasGradients() *tensor.Dense           { return nil }
func (r *ReLU) UpdateBiases(biasUpdate *tensor.Dense)     {}

// ClearCache releases cached input to prevent memory leaks
func (r *ReLU) ClearCache() {
	r.input = nil
}
