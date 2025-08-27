package loss

import "gorgonia.org/tensor"

// Loss interface for all loss functions
type Loss interface {
	Forward(predictions, targets *tensor.Dense) float64
	Backward() *tensor.Dense
}
