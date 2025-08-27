package optimizer

import (
	"math"
	"testing"

	"gorgonia.org/tensor"
)

func floatEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-6
}

func TestSGDStep(t *testing.T) {
	// Create a simple optimizer
	sgd := NewSGD(0.1)

	// Create simple weights and gradients
	weights := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1.0, 2.0, 3.0, 4.0}))
	gradients := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{0.1, 0.2, 0.3, 0.4}))

	// Perform an update step
	updatedWeights := sgd.Step(weights, gradients)

	// Check the updated values
	expected := []float64{0.99, 1.98, 2.97, 3.96} // 1.0 - 0.1*0.1 = 0.99, etc.
	actual := updatedWeights.Data().([]float64)

	for i, v := range expected {
		if !floatEqual(actual[i], v) {
			t.Errorf("Expected %f, got %f at index %d", v, actual[i], i)
		}
	}
}

func TestSGDStepBias(t *testing.T) {
	// Create a simple optimizer
	sgd := NewSGD(0.1)

	// Create simple biases and bias gradients
	biases := tensor.New(tensor.WithShape(1, 3), tensor.WithBacking([]float64{1.0, 2.0, 3.0}))
	biasGradients := tensor.New(tensor.WithShape(1, 3), tensor.WithBacking([]float64{0.1, 0.2, 0.3}))

	// Perform an update step
	updatedBiases := sgd.StepBias(biases, biasGradients)

	// Check the updated values
	expected := []float64{0.99, 1.98, 2.97} // 1.0 - 0.1*0.1 = 0.99, etc.
	actual := updatedBiases.Data().([]float64)

	for i, v := range expected {
		if !floatEqual(actual[i], v) {
			t.Errorf("Expected %f, got %f at index %d", v, actual[i], i)
		}
	}
}

func TestSGDZeroGrad(t *testing.T) {
	// Create a simple optimizer
	sgd := NewSGD(0.1)

	// Call ZeroGrad - it should not panic or cause any issues
	sgd.ZeroGrad()

	// Verify that the learning rate is still correct
	if sgd.GetLearningRate() != 0.1 {
		t.Errorf("Expected learning rate 0.1, got %f", sgd.GetLearningRate())
	}
}
