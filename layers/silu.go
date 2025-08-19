package layer

import (
	"math"
	"gorgonia.org/tensor"
)

type SiLU struct {
	input  *tensor.Dense
	output *tensor.Dense
}

func NewSiLU() *SiLU {
	return &SiLU{}
}

func (s *SiLU) Forward(x *tensor.Dense) *tensor.Dense {
	// Store input for backward pass
	s.input = x.Clone().(*tensor.Dense)
	
	// SiLU (Swish): x * sigmoid(x)
	s.output = x.Clone().(*tensor.Dense)
	data := s.output.Data().([]float64)
	inputData := s.input.Data().([]float64)
	
	for i, v := range inputData {
		// sigmoid(x) = 1 / (1 + exp(-x))
		sigmoid := 1.0 / (1.0 + math.Exp(-v))
		// x * sigmoid(x)
		data[i] = v * sigmoid
	}
	return s.output
}

func (s *SiLU) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	// Derivative of SiLU: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
	// We need the original input values to compute this correctly
	inputData := s.input.Data().([]float64)
	gradData := gradOutput.Data().([]float64)
	
	resultData := make([]float64, len(gradData))
	
	for i, x := range inputData {
		// Compute sigmoid(x)
		sigmoid := 1.0 / (1.0 + math.Exp(-x))
		// Compute derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
		deriv := sigmoid + x*sigmoid*(1-sigmoid)
		// Multiply by upstream gradient
		resultData[i] = deriv * gradData[i]
	}
	
	result := tensor.New(tensor.WithShape(gradOutput.Shape()...), tensor.WithBacking(resultData))
	return result
}

func (s *SiLU) GetWeights() *tensor.Dense                    { return nil }
func (s *SiLU) GetGradients() *tensor.Dense                  { return nil }
func (s *SiLU) UpdateWeights(weightsUpdate *tensor.Dense)    {}
func (s *SiLU) GetBiases() *tensor.Dense                     { return nil }
func (s *SiLU) GetBiasGradients() *tensor.Dense              { return nil }
func (s *SiLU) UpdateBiases(biasUpdate *tensor.Dense)        {}
