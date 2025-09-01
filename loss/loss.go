package loss

import "gorgonia.org/tensor"

// Loss defines the interface for all loss functions used in training
type Loss interface {
	Forward(predictions, targets *tensor.Dense) float64 // computes loss value
	Backward() *tensor.Dense                            // computes gradients w.r.t predictions
}
