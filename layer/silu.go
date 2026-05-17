package layer

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// SiLU implements Sigmoid Linear Unit (Swish): f(x) = x * sigmoid(x)
type SiLU struct {
	input  *mat.Dense
	output *mat.Dense
}

func NewSiLU() *SiLU {
	return &SiLU{}
}

func (s *SiLU) Forward(x *mat.Dense) *mat.Dense {
	s.input = mat.DenseCopyOf(x)
	rows, cols := x.Dims()
	inputData := x.RawMatrix().Data
	outputData := make([]float64, rows*cols)
	for i, v := range inputData {
		sigmoid := 1.0 / (1.0 + math.Exp(-v))
		outputData[i] = v * sigmoid
	}
	s.output = mat.NewDense(rows, cols, outputData)
	return s.output
}

func (s *SiLU) Backward(gradOutput *mat.Dense) *mat.Dense {
	rows, cols := s.input.Dims()
	inputData := s.input.RawMatrix().Data
	gradData := gradOutput.RawMatrix().Data
	resultData := make([]float64, rows*cols)
	for i, x := range inputData {
		sigmoid := 1.0 / (1.0 + math.Exp(-x))
		deriv := sigmoid + x*sigmoid*(1-sigmoid)
		resultData[i] = deriv * gradData[i]
	}
	return mat.NewDense(rows, cols, resultData)
}

func (s *SiLU) GetWeights() *mat.Dense       { return nil }
func (s *SiLU) GetGradients() *mat.Dense     { return nil }
func (s *SiLU) UpdateWeights(w *mat.Dense)   {}
func (s *SiLU) GetBiases() *mat.Dense        { return nil }
func (s *SiLU) GetBiasGradients() *mat.Dense { return nil }
func (s *SiLU) UpdateBiases(b *mat.Dense)    {}
func (s *SiLU) ClearCache()                  { s.input = nil; s.output = nil }
