package optimizer

import (
	"fmt"

	"gorgonia.org/tensor"
)

type SGDMomentum struct {
	LR       float64
	Momentum float64
	v        map[string]*tensor.Dense
	vb       map[string]*tensor.Dense
}

func NewSGDMomentum(lr float64, momentum float64) *SGDMomentum {
	return &SGDMomentum{
		LR:       lr,
		Momentum: momentum,
		v:        make(map[string]*tensor.Dense),
		vb:       make(map[string]*tensor.Dense),
	}
}

func DefaultSGDMomentum(lr float64) *SGDMomentum {
	return NewSGDMomentum(lr, 0.9)
}

func (sgd *SGDMomentum) Step(weights *tensor.Dense, gradients *tensor.Dense) *tensor.Dense {
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
		updatedVData[i] = sgd.Momentum*updatedVData[i] - sgd.LR*gradData[i]
		updatedWeightData[i] += updatedVData[i]
	}

	// Update the velocity tensor with the new values
	copy(v.Data().([]float64), updatedVData)

	// Return a new tensor with updated weights
	return tensor.New(tensor.WithShape(shape...), tensor.WithBacking(updatedWeightData))
}

func (sgd *SGDMomentum) StepBias(biases *tensor.Dense, biasGradients *tensor.Dense) *tensor.Dense {
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
		updatedVBData[i] = sgd.Momentum*updatedVBData[i] - sgd.LR*biasGradData[i]
		updatedBiasData[i] += updatedVBData[i]
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
