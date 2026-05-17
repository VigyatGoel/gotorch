package loss

import "gonum.org/v1/gonum/mat"

// Loss defines the interface for all loss functions used in training
type Loss interface {
	Forward(predictions, targets *mat.Dense) float64
	Backward() *mat.Dense
}
