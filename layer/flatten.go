package layer

import (
	"gorgonia.org/tensor"
)

// Flatten reshapes multi-dimensional input to 2D (batch_size, features)
type Flatten struct {
	inputShape []int // original input shape for backward pass
}

// NewFlatten creates a new flatten layer
func NewFlatten() *Flatten {
	return &Flatten{}
}

// Forward flattens input to [batch_size, flattened_features]
func (f *Flatten) Forward(input *tensor.Dense) *tensor.Dense {
	f.inputShape = input.Shape()
	batchSize := f.inputShape[0]
	flatSize := 1
	for i := 1; i < len(f.inputShape); i++ {
		flatSize *= f.inputShape[i]
	}

	data := input.Data().([]float64)
	return tensor.New(
		tensor.WithShape(batchSize, flatSize),
		tensor.WithBacking(data),
	)
}

// Backward reshapes gradient back to original input shape
func (f *Flatten) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	data := gradOutput.Data().([]float64)
	return tensor.New(
		tensor.WithShape(f.inputShape...),
		tensor.WithBacking(data),
	)
}

func (f *Flatten) GetWeights() *tensor.Dense                 { return nil }
func (f *Flatten) GetGradients() *tensor.Dense               { return nil }
func (f *Flatten) UpdateWeights(weightsUpdate *tensor.Dense) {}
func (f *Flatten) GetBiases() *tensor.Dense                  { return nil }
func (f *Flatten) GetBiasGradients() *tensor.Dense           { return nil }
func (f *Flatten) UpdateBiases(biasUpdate *tensor.Dense)     {}

// ClearCache releases cached shape to prevent memory leaks
func (f *Flatten) ClearCache() {
	f.inputShape = nil
}
