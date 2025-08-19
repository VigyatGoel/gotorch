package optimizer

import "gorgonia.org/tensor"

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

func (sgd *SGD) ZeroGrad() {}

func (sgd *SGD) GetLearningRate() float64 {
	return sgd.LR
}
