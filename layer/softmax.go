package layer

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Softmax implements softmax activation for multi-class classification
type Softmax struct {
	input  *mat.Dense
	output *mat.Dense
}

func NewSoftmax() *Softmax {
	return &Softmax{}
}

func (s *Softmax) Forward(x *mat.Dense) *mat.Dense {
	s.input = mat.DenseCopyOf(x)
	s.output = mat.DenseCopyOf(x)
	data := s.output.RawMatrix().Data
	rows, cols := s.output.Dims()
	for i := 0; i < rows; i++ {
		maxVal := data[i*cols]
		for j := 1; j < cols; j++ {
			if data[i*cols+j] > maxVal {
				maxVal = data[i*cols+j]
			}
		}
		sum := 0.0
		for j := 0; j < cols; j++ {
			data[i*cols+j] = math.Exp(data[i*cols+j] - maxVal)
			sum += data[i*cols+j]
		}
		for j := 0; j < cols; j++ {
			data[i*cols+j] /= sum
		}
	}
	return s.output
}

// Backward assumes Softmax is paired with CrossEntropyLoss; gradient passes through unchanged.
func (s *Softmax) Backward(gradOutput *mat.Dense) *mat.Dense {
	return gradOutput
}

func (s *Softmax) GetWeights() *mat.Dense       { return nil }
func (s *Softmax) GetGradients() *mat.Dense     { return nil }
func (s *Softmax) UpdateWeights(w *mat.Dense)   {}
func (s *Softmax) GetBiases() *mat.Dense        { return nil }
func (s *Softmax) GetBiasGradients() *mat.Dense { return nil }
func (s *Softmax) UpdateBiases(b *mat.Dense)    {}
func (s *Softmax) ClearCache()                  { s.input = nil; s.output = nil }
