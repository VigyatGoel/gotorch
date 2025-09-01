package layer

import (
	"gorgonia.org/tensor"
	"math"
)

// Softmax implements softmax activation for multi-class classification
type Softmax struct {
	input  *tensor.Dense // cached input
	output *tensor.Dense // cached output for gradient computation
}

// NewSoftmax creates a new softmax activation layer
func NewSoftmax() *Softmax {
	return &Softmax{}
}

// Forward applies softmax with numerical stability (subtract max)
func (s *Softmax) Forward(x *tensor.Dense) *tensor.Dense {
	s.input = x.Clone().(*tensor.Dense)
	s.output = x.Clone().(*tensor.Dense)

	data := s.output.Data().([]float64)
	shape := s.output.Shape()

	if len(shape) == 2 {
		rows, cols := shape[0], shape[1]
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
	} else {
		maxVal := data[0]
		for i := 1; i < len(data); i++ {
			if data[i] > maxVal {
				maxVal = data[i]
			}
		}

		sum := 0.0
		for i := 0; i < len(data); i++ {
			data[i] = math.Exp(data[i] - maxVal)
			sum += data[i]
		}

		for i := 0; i < len(data); i++ {
			data[i] /= sum
		}
	}

	return s.output
}

// Backward passes gradient through (assumes use with CrossEntropyLoss)
func (s *Softmax) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	// IMPORTANT: This implementation assumes that the Softmax layer is used in combination
	// with Cross-Entropy Loss. In this specific case, the gradient simplifies to just
	// passing through the gradient from the loss function (which is typically predictions - targets).
	//
	// For a standalone Softmax layer, the full Jacobian computation would be required:
	// gradInput_i = sum_j (gradOutput_j * (output_i * (I_ij - output_j)))
	// where I_ij is 1 if i==j and 0 otherwise.
	//
	// This optimization is commonly used in practice when Softmax is combined with
	// Cross-Entropy Loss for classification tasks.
	return gradOutput
}

func (s *Softmax) GetWeights() *tensor.Dense {
	return nil
}

func (s *Softmax) GetGradients() *tensor.Dense {
	return nil
}

func (s *Softmax) UpdateWeights(weightsUpdate *tensor.Dense) {
}

func (s *Softmax) GetBiases() *tensor.Dense {
	return nil
}

func (s *Softmax) GetBiasGradients() *tensor.Dense {
	return nil
}

func (s *Softmax) UpdateBiases(biasUpdate *tensor.Dense) {
}

// ClearCache releases cached tensors to prevent memory leaks
func (s *Softmax) ClearCache() {
	s.input = nil
	s.output = nil
}
