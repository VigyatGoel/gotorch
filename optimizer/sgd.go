package optimizer

import "gorgonia.org/tensor"

// SGD implements the Stochastic Gradient Descent optimizer.
// It performs the standard PyTorch update rule: parameter = parameter - learning_rate * gradient.
//
// Note: In this implementation, gradient zeroing is handled by the layers themselves
// during the backward pass, where they overwrite their stored gradients rather than
// accumulating them. The ZeroGrad method is a no-op because SGD does not maintain
// any internal state that requires resetting between iterations.
type SGD struct {
	LR float64
}

func NewSGD(lr float64) *SGD {
	return &SGD{LR: lr}
}

func (sgd *SGD) Step(weights *tensor.Dense, gradients *tensor.Dense) *tensor.Dense {
	// Create a new tensor with updated weights: weights = weights - lr * gradients
	weightData := weights.Data().([]float64)
	gradData := gradients.Data().([]float64)

	// Create a copy of the weight data to avoid modifying the original tensor
	updatedWeightData := make([]float64, len(weightData))
	copy(updatedWeightData, weightData)

	// Update weights
	for i := range updatedWeightData {
		updatedWeightData[i] -= sgd.LR * gradData[i]
	}

	// Return a new tensor with updated weights
	shape := weights.Shape()
	return tensor.New(tensor.WithShape(shape...), tensor.WithBacking(updatedWeightData))
}

func (sgd *SGD) StepBias(biases *tensor.Dense, biasGradients *tensor.Dense) *tensor.Dense {
	if biases == nil || biasGradients == nil {
		return biases
	}

	// Create a new tensor with updated biases: biases = biases - lr * biasGradients
	biasData := biases.Data().([]float64)
	biasGradData := biasGradients.Data().([]float64)

	// Create a copy of the bias data to avoid modifying the original tensor
	updatedBiasData := make([]float64, len(biasData))
	copy(updatedBiasData, biasData)

	// Update biases
	for i := range updatedBiasData {
		updatedBiasData[i] -= sgd.LR * biasGradData[i]
	}

	// Return a new tensor with updated biases
	shape := biases.Shape()
	return tensor.New(tensor.WithShape(shape...), tensor.WithBacking(updatedBiasData))
}

// ZeroGrad is a no-op for basic SGD because:
// 1. This implementation does not accumulate gradients across iterations
// 2. Layers in this framework overwrite their gradients in each backward pass
// 3. SGD does not maintain any internal state that needs to be reset
//
// In frameworks where gradients are accumulated, this method would typically
// iterate through all registered parameters and zero their gradient tensors.
func (sgd *SGD) ZeroGrad() {}

func (sgd *SGD) GetLearningRate() float64 {
	return sgd.LR
}
