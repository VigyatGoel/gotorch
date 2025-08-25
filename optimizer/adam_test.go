package optimizer

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestAdamStep(t *testing.T) {
	// Create an optimizer with learning rate 0.1, beta1=0.9, beta2=0.999, and epsilon=1e-8
	adam := NewAdam(0.1, 0.9, 0.999, 1e-8)

	// Create simple weights and gradients
	weights := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1.0, 2.0, 3.0, 4.0}))
	gradients := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{0.1, 0.2, 0.3, 0.4}))

	// Perform first update step
	updatedWeights1 := adam.Step(weights, gradients)

	// Check the updated values after first step
	// For Adam:
	// t = 1
	// beta1_t = beta1^t = 0.9^1 = 0.9
	// beta2_t = beta2^t = 0.999^1 = 0.999
	// 
	// For each parameter:
	// m = beta1 * m_prev + (1 - beta1) * gradient = 0.9 * 0 + (1 - 0.9) * gradient = 0.1 * gradient
	// v = beta2 * v_prev + (1 - beta2) * gradient^2 = 0.999 * 0 + (1 - 0.999) * gradient^2 = 0.001 * gradient^2
	// 
	// m_hat = m / (1 - beta1_t) = (0.1 * gradient) / (1 - 0.9) = (0.1 * gradient) / 0.1 = gradient
	// v_hat = v / (1 - beta2_t) = (0.001 * gradient^2) / (1 - 0.999) = (0.001 * gradient^2) / 0.001 = gradient^2
	// 
	// update = learning_rate * m_hat / (sqrt(v_hat) + epsilon)
	// parameter = parameter - update
	// 
	// For gradient = 0.1:
	// m = 0.1 * 0.1 = 0.01
	// v = 0.001 * 0.1^2 = 0.00001
	// m_hat = 0.01 / 0.1 = 0.1
	// v_hat = 0.00001 / 0.001 = 0.01
	// update = 0.1 * 0.1 / (sqrt(0.01) + 1e-8) = 0.01 / (0.1 + 1e-8) ≈ 0.1
	// parameter = 1.0 - 0.1 = 0.9
	// 
	// For gradient = 0.2:
	// m = 0.1 * 0.2 = 0.02
	// v = 0.001 * 0.2^2 = 0.00004
	// m_hat = 0.02 / 0.1 = 0.2
	// v_hat = 0.00004 / 0.001 = 0.04
	// update = 0.1 * 0.2 / (sqrt(0.04) + 1e-8) = 0.02 / (0.2 + 1e-8) ≈ 0.1
	// parameter = 2.0 - 0.1 = 1.9
	// 
	// For gradient = 0.3:
	// m = 0.1 * 0.3 = 0.03
	// v = 0.001 * 0.3^2 = 0.00009
	// m_hat = 0.03 / 0.1 = 0.3
	// v_hat = 0.00009 / 0.001 = 0.09
	// update = 0.1 * 0.3 / (sqrt(0.09) + 1e-8) = 0.03 / (0.3 + 1e-8) ≈ 0.1
	// parameter = 3.0 - 0.1 = 2.9
	// 
	// For gradient = 0.4:
	// m = 0.1 * 0.4 = 0.04
	// v = 0.001 * 0.4^2 = 0.00016
	// m_hat = 0.04 / 0.1 = 0.4
	// v_hat = 0.00016 / 0.001 = 0.16
	// update = 0.1 * 0.4 / (sqrt(0.16) + 1e-8) = 0.04 / (0.4 + 1e-8) ≈ 0.1
	// parameter = 4.0 - 0.1 = 3.9
	// 
	// Wait, that's not right. Let me recalculate:
	// 
	// For gradient = 0.1:
	// update = 0.1 * 0.1 / (sqrt(0.01) + 1e-8) = 0.01 / (0.1 + 1e-8) = 0.1
	// parameter = 1.0 - 0.1 = 0.9
	// 
	// For gradient = 0.2:
	// update = 0.1 * 0.2 / (sqrt(0.04) + 1e-8) = 0.02 / (0.2 + 1e-8) = 0.1
	// parameter = 2.0 - 0.1 = 1.9
	// 
	// For gradient = 0.3:
	// update = 0.1 * 0.3 / (sqrt(0.09) + 1e-8) = 0.03 / (0.3 + 1e-8) = 0.1
	// parameter = 3.0 - 0.1 = 2.9
	// 
	// For gradient = 0.4:
	// update = 0.1 * 0.4 / (sqrt(0.16) + 1e-8) = 0.04 / (0.4 + 1e-8) = 0.1
	// parameter = 4.0 - 0.1 = 3.9
	// 
	// Hmm, that's interesting that they all have the same update. That's because the ratio 
	// of gradient to sqrt(gradient^2) is always 1 (or -1) when gradient is positive.
	// 
	// Let's just check that the values are close to what we expect.
	expected1 := []float64{0.9, 1.9, 2.9, 3.9}
	actual1 := updatedWeights1.Data().([]float64)

	for i, v := range expected1 {
		if !floatEqual(actual1[i], v) {
			t.Errorf("After first step, expected %f, got %f at index %d", v, actual1[i], i)
		}
	}

	// Perform second update step with same gradients
	updatedWeights2 := adam.Step(updatedWeights1, gradients)

	// For the second step (t=2):
	// beta1_t = 0.9^2 = 0.81
	// beta2_t = 0.999^2 = 0.998001
	// 
	// For gradient = 0.1:
	// m = 0.9 * 0.01 + 0.1 * 0.1 = 0.009 + 0.01 = 0.019
	// v = 0.999 * 0.00001 + 0.001 * 0.01 = 0.00000999 + 0.00001 = 0.00001999
	// m_hat = 0.019 / (1 - 0.81) = 0.019 / 0.19 = 0.1
	// v_hat = 0.00001999 / (1 - 0.998001) = 0.00001999 / 0.001999 ≈ 0.010005
	// update = 0.1 * 0.1 / (sqrt(0.010005) + 1e-8) ≈ 0.1 * 0.1 / 0.100025 ≈ 0.099975
	// parameter = 0.9 - 0.099975 ≈ 0.800025
	// 
	// Let's just check that the values are in the right direction.
	actual2 := updatedWeights2.Data().([]float64)
	
	// All parameters should have decreased further
	for i := range actual1 {
		if actual2[i] >= actual1[i] {
			t.Errorf("After second step, expected parameter %d to decrease, but went from %f to %f", i, actual1[i], actual2[i])
		}
	}
}

func TestAdamStepBias(t *testing.T) {
	// Create an optimizer with learning rate 0.1, beta1=0.9, beta2=0.999, and epsilon=1e-8
	adam := NewAdam(0.1, 0.9, 0.999, 1e-8)

	// Create simple biases and bias gradients
	biases := tensor.New(tensor.WithShape(1, 3), tensor.WithBacking([]float64{1.0, 2.0, 3.0}))
	biasGradients := tensor.New(tensor.WithShape(1, 3), tensor.WithBacking([]float64{0.1, 0.2, 0.3}))

	// Perform first update step
	updatedBiases1 := adam.StepBias(biases, biasGradients)

	// Check that the values have changed in the right direction
	actual1 := updatedBiases1.Data().([]float64)
	expected := []float64{1.0, 2.0, 3.0}
	
	// All parameters should have decreased
	for i := range expected {
		if actual1[i] >= expected[i] {
			t.Errorf("After first step, expected bias %d to decrease, but went from %f to %f", i, expected[i], actual1[i])
		}
	}

	// Perform second update step with same gradients
	updatedBiases2 := adam.StepBias(updatedBiases1, biasGradients)
	
	// All parameters should have decreased further
	actual2 := updatedBiases2.Data().([]float64)
	
	for i := range actual1 {
		if actual2[i] >= actual1[i] {
			t.Errorf("After second step, expected bias %d to decrease, but went from %f to %f", i, actual1[i], actual2[i])
		}
	}
}

func TestAdamZeroGrad(t *testing.T) {
	// Create an optimizer with learning rate 0.1, beta1=0.9, beta2=0.999, and epsilon=1e-8
	adam := DefaultAdam(0.1)

	// Create simple weights and gradients
	weights := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1.0, 2.0, 3.0, 4.0}))
	gradients := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{0.1, 0.2, 0.3, 0.4}))

	// Perform an update step to initialize momentum and velocity
	adam.Step(weights, gradients)

	// Check that momentum and velocity tensors exist
	if len(adam.m) == 0 || len(adam.v) == 0 {
		t.Errorf("Momentum and velocity tensors not created")
	}

	// Call ZeroGrad
	adam.ZeroGrad()

	// Check that time step is reset
	if adam.T != 0 {
		t.Errorf("Expected time step to be reset to 0, got %d", adam.T)
	}

	// Check that momentum and velocity maps are empty
	if len(adam.m) != 0 || len(adam.v) != 0 {
		t.Errorf("Expected momentum and velocity maps to be empty after ZeroGrad")
	}

	// Check that bias momentum and velocity maps are empty
	if len(adam.mb) != 0 || len(adam.vb) != 0 {
		t.Errorf("Expected bias momentum and velocity maps to be empty after ZeroGrad")
	}

	// Verify that the learning rate is still correct
	if adam.GetLearningRate() != 0.1 {
		t.Errorf("Expected learning rate 0.1, got %f", adam.GetLearningRate())
	}
}

func TestDefaultAdam(t *testing.T) {
	// Test that DefaultAdam creates an Adam optimizer with the correct default parameters
	adam := DefaultAdam(0.001)
	
	if adam.LR != 0.001 {
		t.Errorf("Expected learning rate 0.001, got %f", adam.LR)
	}
	
	if adam.Beta1 != 0.9 {
		t.Errorf("Expected beta1 0.9, got %f", adam.Beta1)
	}
	
	if adam.Beta2 != 0.999 {
		t.Errorf("Expected beta2 0.999, got %f", adam.Beta2)
	}
	
	if adam.Epsilon != 1e-8 {
		t.Errorf("Expected epsilon 1e-8, got %f", adam.Epsilon)
	}
}