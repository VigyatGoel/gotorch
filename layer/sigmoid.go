package layer

import (
	"gorgonia.org/tensor"
	"math"
)

// Sigmoid implements sigmoid activation: f(x) = 1 / (1 + exp(-x))
type Sigmoid struct {
	output *tensor.Dense // cached output for gradient computation
}

// NewSigmoid creates a new sigmoid activation layer
func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

// Forward applies sigmoid activation element-wise
func (s *Sigmoid) Forward(x *tensor.Dense) *tensor.Dense {
	s.output = x.Clone().(*tensor.Dense)
	data := s.output.Data().([]float64)
	for i, v := range data {
		data[i] = 1.0 / (1.0 + math.Exp(-v))
	}
	return s.output
}

// Backward computes sigmoid gradient: output * (1 - output)
func (s *Sigmoid) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	deriv := s.output.Clone().(*tensor.Dense)
	derivData := deriv.Data().([]float64)
	outputData := s.output.Data().([]float64)

	for i, v := range outputData {
		derivData[i] = v * (1.0 - v)
	}

	result, _ := tensor.Mul(deriv, gradOutput)
	return result.(*tensor.Dense)
}

func (s *Sigmoid) GetWeights() *tensor.Dense                 { return nil }
func (s *Sigmoid) GetGradients() *tensor.Dense               { return nil }
func (s *Sigmoid) UpdateWeights(weightsUpdate *tensor.Dense) {}
func (s *Sigmoid) GetBiases() *tensor.Dense                  { return nil }
func (s *Sigmoid) GetBiasGradients() *tensor.Dense           { return nil }
func (s *Sigmoid) UpdateBiases(biasUpdate *tensor.Dense)     {}

// ClearCache releases cached output to prevent memory leaks
func (s *Sigmoid) ClearCache() {
	s.output = nil
}
