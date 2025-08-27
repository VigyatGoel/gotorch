package layer

import (
	"gorgonia.org/tensor"
	"math"
)

type Sigmoid struct {
	output *tensor.Dense // Store output for use in backward pass (needed for gradient computation)
}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (s *Sigmoid) Forward(x *tensor.Dense) *tensor.Dense {
	// Apply sigmoid: 1 / (1 + exp(-x))
	s.output = x.Clone().(*tensor.Dense)
	data := s.output.Data().([]float64)
	for i, v := range data {
		data[i] = 1.0 / (1.0 + math.Exp(-v))
	}
	return s.output
}

func (s *Sigmoid) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	// Gradient of sigmoid: output * (1 - output)
	deriv := s.output.Clone().(*tensor.Dense)
	derivData := deriv.Data().([]float64)
	outputData := s.output.Data().([]float64)

	// Calculate output * (1 - output) directly
	for i, v := range outputData {
		derivData[i] = v * (1.0 - v)
	}

	// Element-wise multiplication with gradOutput
	result, _ := tensor.Mul(deriv, gradOutput)
	return result.(*tensor.Dense)
}

func (s *Sigmoid) GetWeights() *tensor.Dense                 { return nil }
func (s *Sigmoid) GetGradients() *tensor.Dense               { return nil }
func (s *Sigmoid) UpdateWeights(weightsUpdate *tensor.Dense) {}
func (s *Sigmoid) GetBiases() *tensor.Dense                  { return nil }
func (s *Sigmoid) GetBiasGradients() *tensor.Dense           { return nil }
func (s *Sigmoid) UpdateBiases(biasUpdate *tensor.Dense)     {}
