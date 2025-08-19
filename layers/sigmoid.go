package layer

import (
	"math"
	"gorgonia.org/tensor"
)

type Sigmoid struct {
	output *tensor.Dense
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
	ones := tensor.Ones(tensor.Float64, s.output.Shape()...)
	onesData := ones.Data().([]float64)
	outputData := s.output.Data().([]float64)
	
	// Calculate 1 - output
	for i, v := range outputData {
		onesData[i] = 1.0 - v
	}
	
	// Calculate output * (1 - output)
	deriv := s.output.Clone().(*tensor.Dense)
	derivData := deriv.Data().([]float64)
	for i, v := range outputData {
		derivData[i] = v * onesData[i]
	}
	
	// Element-wise multiplication with gradOutput
	result, _ := tensor.Mul(deriv, gradOutput)
	return result.(*tensor.Dense)
}

func (s *Sigmoid) GetWeights() *tensor.Dense                    { return nil }
func (s *Sigmoid) GetGradients() *tensor.Dense                  { return nil }
func (s *Sigmoid) UpdateWeights(weightsUpdate *tensor.Dense)    {}
func (s *Sigmoid) GetBiases() *tensor.Dense                     { return nil }
func (s *Sigmoid) GetBiasGradients() *tensor.Dense              { return nil }
func (s *Sigmoid) UpdateBiases(biasUpdate *tensor.Dense)        {}
