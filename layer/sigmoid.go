package layer

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Sigmoid implements sigmoid activation: f(x) = 1 / (1 + exp(-x))
type Sigmoid struct {
	output *mat.Dense
}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (s *Sigmoid) Forward(x *mat.Dense) *mat.Dense {
	s.output = mat.DenseCopyOf(x)
	data := s.output.RawMatrix().Data
	for i, v := range data {
		data[i] = 1.0 / (1.0 + math.Exp(-v))
	}
	return s.output
}

func (s *Sigmoid) Backward(gradOutput *mat.Dense) *mat.Dense {
	rows, cols := s.output.Dims()
	outputData := s.output.RawMatrix().Data
	gradData := gradOutput.RawMatrix().Data
	resultData := make([]float64, rows*cols)
	for i, v := range outputData {
		resultData[i] = v * (1.0 - v) * gradData[i]
	}
	return mat.NewDense(rows, cols, resultData)
}

func (s *Sigmoid) GetWeights() *mat.Dense       { return nil }
func (s *Sigmoid) GetGradients() *mat.Dense     { return nil }
func (s *Sigmoid) UpdateWeights(w *mat.Dense)   {}
func (s *Sigmoid) GetBiases() *mat.Dense        { return nil }
func (s *Sigmoid) GetBiasGradients() *mat.Dense { return nil }
func (s *Sigmoid) UpdateBiases(b *mat.Dense)    {}
func (s *Sigmoid) ClearCache()                  { s.output = nil }
