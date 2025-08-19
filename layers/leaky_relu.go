package layer

import (
	"gorgonia.org/tensor"
)

type LeakyReLU struct {
	Alpha  float64
	input  *tensor.Dense
	output *tensor.Dense
}

func NewLeakyReLU(alpha float64) *LeakyReLU {
	return &LeakyReLU{
		Alpha: alpha,
	}
}

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

func (l *LeakyReLU) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	// Gradient of LeakyReLU: 1 if x > 0, alpha otherwise
	grad := l.input.Clone().(*tensor.Dense)
	inputData := grad.Data().([]float64)
	for i, v := range inputData {
		if v > 0 {
			inputData[i] = 1.0
		} else {
			inputData[i] = l.Alpha
		}
	}
	
	// Element-wise multiplication with gradOutput
	result, _ := tensor.Mul(grad, gradOutput)
	return result.(*tensor.Dense)
}

func (l *LeakyReLU) GetWeights() *tensor.Dense                    { return nil }
func (l *LeakyReLU) GetGradients() *tensor.Dense                  { return nil }
func (l *LeakyReLU) UpdateWeights(weightsUpdate *tensor.Dense)    {}
func (l *LeakyReLU) GetBiases() *tensor.Dense                     { return nil }
func (l *LeakyReLU) GetBiasGradients() *tensor.Dense              { return nil }
func (l *LeakyReLU) UpdateBiases(biasUpdate *tensor.Dense)        {}
