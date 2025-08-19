package layer

import (
	"gorgonia.org/tensor"
)

type ReLU struct {
	input *tensor.Dense
}

func NewReLU() *ReLU {
	return &ReLU{}
}

func (r *ReLU) Forward(x *tensor.Dense) *tensor.Dense {
	r.input = x
	// Apply ReLU: max(0, x)
	result := x.Clone().(*tensor.Dense)
	data := result.Data().([]float64)
	for i, v := range data {
		if v < 0 {
			data[i] = 0
		}
	}
	return result
}

func (r *ReLU) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	// Gradient of ReLU: 1 if x > 0, 0 otherwise
	grad := r.input.Clone().(*tensor.Dense)
	inputData := grad.Data().([]float64)
	for i, v := range inputData {
		if v > 0 {
			inputData[i] = 1.0
		} else {
			inputData[i] = 0.0
		}
	}
	
	// Element-wise multiplication with gradOutput
	result, _ := tensor.Mul(grad, gradOutput)
	return result.(*tensor.Dense)
}

func (r *ReLU) GetWeights() *tensor.Dense                    { return nil }
func (r *ReLU) GetGradients() *tensor.Dense                  { return nil }
func (r *ReLU) UpdateWeights(weightsUpdate *tensor.Dense)    {}
func (r *ReLU) GetBiases() *tensor.Dense                     { return nil }
func (r *ReLU) GetBiasGradients() *tensor.Dense              { return nil }
func (r *ReLU) UpdateBiases(biasUpdate *tensor.Dense)        {}
