package layer

import "gorgonia.org/tensor"

// Layer defines the interface that all neural network layers must implement
type Layer interface {
	// Core forward and backward propagation
	Forward(input *tensor.Dense) *tensor.Dense // computes layer output
	Backward(dout *tensor.Dense) *tensor.Dense // computes input gradients

	// Weight management (returns nil for layers without weights)
	GetWeights() *tensor.Dense                 // returns current weights
	GetGradients() *tensor.Dense               // returns weight gradients
	UpdateWeights(weightsUpdate *tensor.Dense) // updates weights with new values

	// Bias management (returns nil for layers without biases)
	GetBiases() *tensor.Dense              // returns current biases
	GetBiasGradients() *tensor.Dense       // returns bias gradients
	UpdateBiases(biasUpdate *tensor.Dense) // updates biases with new values

	// Memory management
	ClearCache() // releases cached data to prevent memory leaks
}
