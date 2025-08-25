package optimizer

import (
	"math"
	"testing"

	"gorgonia.org/tensor"
)

const epsilon = 1e-6

func TestSGDMomentumStep(t *testing.T) {
	// Create an optimizer with learning rate 0.1 and momentum 0.9
	sgd := NewSGDMomentum(0.1, 0.9)

	// Create simple weights and gradients
	weights := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1.0, 2.0, 3.0, 4.0}))
	gradients := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{0.1, 0.2, 0.3, 0.4}))

	// Perform first update step
	updatedWeights1 := sgd.Step(weights, gradients)

	// Check the updated values after first step
	// v = momentum * v + gradient = 0.9 * 0 + [0.1, 0.2, 0.3, 0.4] = [0.1, 0.2, 0.3, 0.4]
	// weight = weight - lr * v = [1.0, 2.0, 3.0, 4.0] - 0.1 * [0.1, 0.2, 0.3, 0.4] = [0.99, 1.98, 2.97, 3.96]
	expected1 := []float64{0.99, 1.98, 2.97, 3.96}
	actual1 := updatedWeights1.Data().([]float64)

	for i, v := range expected1 {
		if math.Abs(actual1[i]-v) > epsilon {
			t.Errorf("After first step, expected %f, got %f at index %d", v, actual1[i], i)
		}
	}

	// Perform second update step with same gradients
	updatedWeights2 := sgd.Step(updatedWeights1, gradients)

	// Check the updated values after second step
	// v = momentum * v + gradient = 0.9 * [0.1, 0.2, 0.3, 0.4] + [0.1, 0.2, 0.3, 0.4] = [0.19, 0.38, 0.57, 0.76]
	// weight = weight - lr * v = [0.99, 1.98, 2.97, 3.96] - 0.1 * [0.19, 0.38, 0.57, 0.76] = [0.971, 1.942, 2.913, 3.884]
	expected2 := []float64{0.971, 1.942, 2.913, 3.884}
	actual2 := updatedWeights2.Data().([]float64)

	for i, v := range expected2 {
		if math.Abs(actual2[i]-v) > epsilon {
			t.Errorf("After second step, expected %f, got %f at index %d", v, actual2[i], i)
		}
	}
}

func TestSGDMomentumStepBias(t *testing.T) {
	// Create an optimizer with learning rate 0.1 and momentum 0.9
	sgd := NewSGDMomentum(0.1, 0.9)

	// Create simple biases and bias gradients
	biases := tensor.New(tensor.WithShape(1, 3), tensor.WithBacking([]float64{1.0, 2.0, 3.0}))
	biasGradients := tensor.New(tensor.WithShape(1, 3), tensor.WithBacking([]float64{0.1, 0.2, 0.3}))

	// Perform first update step
	updatedBiases1 := sgd.StepBias(biases, biasGradients)

	// Check the updated values after first step
	// v = momentum * v + gradient = 0.9 * 0 + [0.1, 0.2, 0.3] = [0.1, 0.2, 0.3]
	// bias = bias - lr * v = [1.0, 2.0, 3.0] - 0.1 * [0.1, 0.2, 0.3] = [0.99, 1.98, 2.97]
	expected1 := []float64{0.99, 1.98, 2.97}
	actual1 := updatedBiases1.Data().([]float64)

	for i, v := range expected1 {
		if math.Abs(actual1[i]-v) > epsilon {
			t.Errorf("After first step, expected %f, got %f at index %d", v, actual1[i], i)
		}
	}

	// Perform second update step with same gradients
	updatedBiases2 := sgd.StepBias(updatedBiases1, biasGradients)

	// Check the updated values after second step
	// v = momentum * v + gradient = 0.9 * [0.1, 0.2, 0.3] + [0.1, 0.2, 0.3] = [0.19, 0.38, 0.57]
	// bias = bias - lr * v = [0.99, 1.98, 2.97] - 0.1 * [0.19, 0.38, 0.57] = [0.971, 1.942, 2.913]
	expected2 := []float64{0.971, 1.942, 2.913}
	actual2 := updatedBiases2.Data().([]float64)

	for i, v := range expected2 {
		if math.Abs(actual2[i]-v) > epsilon {
			t.Errorf("After second step, expected %f, got %f at index %d", v, actual2[i], i)
		}
	}
}

func TestSGDMomentumZeroGrad(t *testing.T) {
	// Create an optimizer with learning rate 0.1 and momentum 0.9
	sgd := NewSGDMomentum(0.1, 0.9)

	// Create simple weights and gradients
	weights := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1.0, 2.0, 3.0, 4.0}))
	gradients := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{0.1, 0.2, 0.3, 0.4}))

	// Perform an update step to initialize velocity
	updatedWeights := sgd.Step(weights, gradients)

	// Check that velocity tensor exists by performing another step
	updatedWeights2 := sgd.Step(updatedWeights, gradients)
	
	// If we get here without a panic, the velocity tensor was created correctly
	_ = updatedWeights2

	// Call ZeroGrad
	sgd.ZeroGrad()

	// Verify that the learning rate is still correct
	if sgd.GetLearningRate() != 0.1 {
		t.Errorf("Expected learning rate 0.1, got %f", sgd.GetLearningRate())
	}
}