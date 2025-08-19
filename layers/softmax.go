package layer

import (
	"math"
	"gorgonia.org/tensor"
)

type Softmax struct {
	input  *tensor.Dense
	output *tensor.Dense
}

func NewSoftmax() *Softmax {
	return &Softmax{}
}

func (s *Softmax) Forward(x *tensor.Dense) *tensor.Dense {
	// Store input for backward pass
	s.input = x.Clone().(*tensor.Dense)
	
	// Apply softmax: exp(x) / sum(exp(x))
	s.output = x.Clone().(*tensor.Dense)
	
	// Get data slice
	data := s.output.Data().([]float64)
	shape := s.output.Shape()
	
	// For 2D tensor, apply softmax along the last dimension (columns)
	if len(shape) == 2 {
		rows, cols := shape[0], shape[1]
		for i := 0; i < rows; i++ {
			// Find max value for numerical stability
			maxVal := data[i*cols]
			for j := 1; j < cols; j++ {
				if data[i*cols+j] > maxVal {
					maxVal = data[i*cols+j]
				}
			}
			
			// Calculate sum of exponentials
			sum := 0.0
			for j := 0; j < cols; j++ {
				data[i*cols+j] = math.Exp(data[i*cols+j] - maxVal)
				sum += data[i*cols+j]
			}
			
			// Normalize
			for j := 0; j < cols; j++ {
				data[i*cols+j] /= sum
			}
		}
	} else {
		// For 1D tensor
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

func (s *Softmax) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	// When using softmax with cross-entropy loss, the gradient simplifies to
	// just passing through the gradient from the loss function
	// This is because the derivative of cross-entropy loss with softmax activation
	// is simply (predictions - targets)
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
