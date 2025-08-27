package optimizer

import (
	"fmt"

	"gorgonia.org/tensor"
)

// SGDMomentum implements the Stochastic Gradient Descent optimizer with Momentum.
// It follows the standard PyTorch formulation:
//
//	v_t = momentum * v_{t-1} + gradient
//	parameter = parameter - learning_rate * v_t
//
// where v_t is the velocity at time step t, momentum is the momentum coefficient,
// gradient is the gradient of the loss with respect to the parameter,
// and learning_rate is the learning rate.
//
// This implementation uses the default PyTorch parameters:
// - dampening = 0 (no dampening)
// - nesterov = false (standard momentum, not Nesterov momentum)
type SGDMomentum struct {
	LR       float64
	Momentum float64
	v        map[string]*tensor.Dense
	vb       map[string]*tensor.Dense
}

// NewSGDMomentum creates a new SGDMomentum optimizer with the specified learning rate and momentum.
// The momentum parameter should typically be between 0 and 1, with common values like 0.9 or 0.99.
func NewSGDMomentum(lr float64, momentum float64) *SGDMomentum {
	return &SGDMomentum{
		LR:       lr,
		Momentum: momentum,
		v:        make(map[string]*tensor.Dense),
		vb:       make(map[string]*tensor.Dense),
	}
}

// DefaultSGDMomentum creates a new SGDMomentum optimizer with the specified learning rate
// and default momentum value of 0.9.
func DefaultSGDMomentum(lr float64) *SGDMomentum {
	return NewSGDMomentum(lr, 0.9)
}

func (sgd *SGDMomentum) Step(weights *tensor.Dense, gradients *tensor.Dense) *tensor.Dense {
	// Handle nil inputs
	if weights == nil || gradients == nil {
		return weights
	}

	shape := weights.Shape()
	if len(shape) == 0 {
		fmt.Println("SGD optimizer skipping update - empty weights or gradients")
		return weights
	}

	gShape := gradients.Shape()
	if len(shape) != len(gShape) {
		fmt.Printf("SGD optimizer dimension mismatch: weights %v vs gradients %v\n", shape, gShape)
		return weights
	}

	for i := range shape {
		if shape[i] != gShape[i] {
			fmt.Printf("SGD optimizer dimension mismatch: weights %v vs gradients %v\n", shape, gShape)
			return weights
		}
	}

	key := fmt.Sprintf("weights_%v", shape)
	if _, ok := sgd.v[key]; !ok {
		// Initialize velocity tensor with zeros
		size := 1
		for _, dim := range shape {
			size *= dim
		}
		zeroData := make([]float64, size)
		sgd.v[key] = tensor.New(tensor.WithShape(shape...), tensor.WithBacking(zeroData))
	}

	v := sgd.v[key]
	weightData := weights.Data().([]float64)
	gradData := gradients.Data().([]float64)
	vData := v.Data().([]float64)

	// Create a copy of the weight data to avoid modifying the original tensor
	updatedWeightData := make([]float64, len(weightData))
	copy(updatedWeightData, weightData)

	// Create a copy of the velocity data to avoid modifying the original tensor
	updatedVData := make([]float64, len(vData))
	copy(updatedVData, vData)

	for i := range updatedWeightData {
		// Standard momentum update: v = momentum * v + gradient
		updatedVData[i] = sgd.Momentum*updatedVData[i] + gradData[i]
		// Parameter update: weight = weight - lr * v
		updatedWeightData[i] = updatedWeightData[i] - sgd.LR*updatedVData[i]
	}

	// Update the velocity tensor with the new values
	copy(v.Data().([]float64), updatedVData)

	// Return a new tensor with updated weights
	return tensor.New(tensor.WithShape(shape...), tensor.WithBacking(updatedWeightData))
}

func (sgd *SGDMomentum) StepBias(biases *tensor.Dense, biasGradients *tensor.Dense) *tensor.Dense {
	// Handle nil inputs
	if biases == nil || biasGradients == nil {
		return biases
	}

	shape := biases.Shape()
	if len(shape) == 0 {
		return biases
	}

	gShape := biasGradients.Shape()
	if len(shape) != len(gShape) {
		fmt.Printf("SGD optimizer dimension mismatch: biases %v vs biasGradients %v\n", shape, gShape)
		return biases
	}

	for i := range shape {
		if shape[i] != gShape[i] {
			fmt.Printf("SGD optimizer dimension mismatch: biases %v vs biasGradients %v\n", shape, gShape)
			return biases
		}
	}

	key := fmt.Sprintf("bias_%v", shape)
	if _, ok := sgd.vb[key]; !ok {
		// Initialize velocity tensor with zeros
		size := 1
		for _, dim := range shape {
			size *= dim
		}
		zeroData := make([]float64, size)
		sgd.vb[key] = tensor.New(tensor.WithShape(shape...), tensor.WithBacking(zeroData))
	}

	vb := sgd.vb[key]
	biasData := biases.Data().([]float64)
	biasGradData := biasGradients.Data().([]float64)
	vbData := vb.Data().([]float64)

	// Create a copy of the bias data to avoid modifying the original tensor
	updatedBiasData := make([]float64, len(biasData))
	copy(updatedBiasData, biasData)

	// Create a copy of the velocity data to avoid modifying the original tensor
	updatedVBData := make([]float64, len(vbData))
	copy(updatedVBData, vbData)

	for i := range updatedBiasData {
		// Standard momentum update: v = momentum * v + gradient
		updatedVBData[i] = sgd.Momentum*updatedVBData[i] + biasGradData[i]
		// Parameter update: bias = bias - lr * v
		updatedBiasData[i] = updatedBiasData[i] - sgd.LR*updatedVBData[i]
	}

	// Update the velocity tensor with the new values
	copy(vb.Data().([]float64), updatedVBData)

	// Return a new tensor with updated biases
	return tensor.New(tensor.WithShape(shape...), tensor.WithBacking(updatedBiasData))
}

func (sgd *SGDMomentum) ZeroGrad() {
	// Reset velocity tensors
	for _, v := range sgd.v {
		vData := v.Data().([]float64)
		for i := range vData {
			vData[i] = 0.0
		}
	}
	for _, vb := range sgd.vb {
		vbData := vb.Data().([]float64)
		for i := range vbData {
			vbData[i] = 0.0
		}
	}
}

func (sgd *SGDMomentum) GetLearningRate() float64 {
	return sgd.LR
}
