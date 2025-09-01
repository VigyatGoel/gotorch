package layer

import (
	"gorgonia.org/tensor"
	"math"
)

// SiLU implements Sigmoid Linear Unit (Swish): f(x) = x * sigmoid(x)
type SiLU struct {
	input  *tensor.Dense // cached input for gradient computation
	output *tensor.Dense // cached output
}

// NewSiLU creates a new SiLU (Swish) activation layer
func NewSiLU() *SiLU {
	return &SiLU{}
}

// Forward applies SiLU activation: x * sigmoid(x)
func (s *SiLU) Forward(x *tensor.Dense) *tensor.Dense {
	s.input = x.Clone().(*tensor.Dense)
	s.output = x.Clone().(*tensor.Dense)
	data := s.output.Data().([]float64)
	inputData := s.input.Data().([]float64)

	for i, v := range inputData {
		sigmoid := 1.0 / (1.0 + math.Exp(-v))
		data[i] = v * sigmoid
	}
	return s.output
}

// Backward computes SiLU gradient: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
func (s *SiLU) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	inputData := s.input.Data().([]float64)
	gradData := gradOutput.Data().([]float64)

	resultData := make([]float64, len(gradData))

	for i, x := range inputData {
		sigmoid := 1.0 / (1.0 + math.Exp(-x))
		deriv := sigmoid + x*sigmoid*(1-sigmoid)
		resultData[i] = deriv * gradData[i]
	}

	result := tensor.New(tensor.WithShape(gradOutput.Shape()...), tensor.WithBacking(resultData))
	return result
}

func (s *SiLU) GetWeights() *tensor.Dense                 { return nil }
func (s *SiLU) GetGradients() *tensor.Dense               { return nil }
func (s *SiLU) UpdateWeights(weightsUpdate *tensor.Dense) {}
func (s *SiLU) GetBiases() *tensor.Dense                  { return nil }
func (s *SiLU) GetBiasGradients() *tensor.Dense           { return nil }
func (s *SiLU) UpdateBiases(biasUpdate *tensor.Dense)     {}

// ClearCache releases cached tensors to prevent memory leaks
func (s *SiLU) ClearCache() {
	s.input = nil
	s.output = nil
}
