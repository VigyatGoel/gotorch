package layer

import (
	"gorgonia.org/tensor"
)

type Flatten struct {
	inputShape []int
}

func NewFlatten() *Flatten {
	return &Flatten{}
}

func (f *Flatten) Forward(input *tensor.Dense) *tensor.Dense {
	// Store input shape for backward pass
	f.inputShape = input.Shape()

	// Calculate flattened size
	batchSize := f.inputShape[0]
	flatSize := 1
	for i := 1; i < len(f.inputShape); i++ {
		flatSize *= f.inputShape[i]
	}

	// Reshape to [batchSize, flatSize]
	data := input.Data().([]float64)
	return tensor.New(
		tensor.WithShape(batchSize, flatSize),
		tensor.WithBacking(data),
	)
}

func (f *Flatten) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	// Reshape gradient back to original input shape
	data := gradOutput.Data().([]float64)
	return tensor.New(
		tensor.WithShape(f.inputShape...),
		tensor.WithBacking(data),
	)
}

// Implement Layer interface
func (f *Flatten) GetWeights() *tensor.Dense                 { return nil }
func (f *Flatten) GetGradients() *tensor.Dense               { return nil }
func (f *Flatten) UpdateWeights(weightsUpdate *tensor.Dense) {}
func (f *Flatten) GetBiases() *tensor.Dense                  { return nil }
func (f *Flatten) GetBiasGradients() *tensor.Dense           { return nil }
func (f *Flatten) UpdateBiases(biasUpdate *tensor.Dense)     {}
