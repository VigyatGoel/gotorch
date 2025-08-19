package utils

import (
	"math"

	"gorgonia.org/tensor"
)

func Zeros(shape ...int) *tensor.Dense {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return tensor.New(tensor.WithShape(shape...), tensor.WithBacking(make([]float64, size)))
}

func Ones(shape ...int) *tensor.Dense {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float64, size)
	for i := range data {
		data[i] = 1.0
	}
	return tensor.New(tensor.WithShape(shape...), tensor.WithBacking(data))
}

func Dot(a, b *tensor.Dense) *tensor.Dense {
	result, err := tensor.MatMul(a, b)
	if err != nil {
		panic("Dimension mismatch for matrix multiplication")
	}
	return result.(*tensor.Dense)
}

func Add(a, b *tensor.Dense) *tensor.Dense {
	result, err := tensor.Add(a, b)
	if err != nil {
		panic("Dimension mismatch for tensor addition")
	}
	return result.(*tensor.Dense)
}

func Transpose(a *tensor.Dense) *tensor.Dense {
	result, err := tensor.Transpose(a)
	if err != nil {
		panic("Error transposing tensor")
	}
	return result.(*tensor.Dense)
}

func ApplyFunc(a *tensor.Dense, f func(float64) float64) *tensor.Dense {
	result := a.Clone().(*tensor.Dense)
	data := result.Data().([]float64)
	for i, v := range data {
		data[i] = f(v)
	}
	return result
}

func ApplyFuncDense(a *tensor.Dense, f func(float64) float64) *tensor.Dense {
	return ApplyFunc(a, f)
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	s := Sigmoid(x)
	return s * (1 - s)
}

func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func ReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func LeakyReLU(x float64, alpha float64) float64 {
	if x > 0 {
		return x
	}
	return alpha * x
}

func LeakyReLUDerivative(x float64, alpha float64) float64 {
	if x > 0 {
		return 1
	}
	return alpha
}

func SiLU(x float64) float64 {
	return x * Sigmoid(x)
}

func SiLUDerivative(x float64) float64 {
	s := Sigmoid(x)
	return s + (x * s * (1 - s))
}

func Softmax(logits *tensor.Dense) *tensor.Dense {
	result := logits.Clone().(*tensor.Dense)
	
	// Get data slice
	data := result.Data().([]float64)
	shape := result.Shape()
	
	// For 2D tensor, apply softmax along the last dimension (columns)
	if len(shape) == 2 {
		rows, cols := shape[0], shape[1]
		for i := 0; i < rows; i++ {
			// Find max value for numerical stability
			maxVal := data[i*cols]
			for j := 1; j < cols; j++ {
				if data[i*cols+j] > maxVal {
					maxVal = data[i*cols+j]
				}
			}
			
			// Calculate sum of exponentials
			sum := 0.0
			for j := 0; j < cols; j++ {
				data[i*cols+j] = math.Exp(data[i*cols+j] - maxVal)
				sum += data[i*cols+j]
			}
			
			// Normalize
			for j := 0; j < cols; j++ {
				data[i*cols+j] /= sum
			}
		}
	} else {
		// For 1D tensor
		maxVal := data[0]
		for i := 1; i < len(data); i++ {
			if data[i] > maxVal {
				maxVal = data[i]
			}
		}
		
		sum := 0.0
		for i := 0; i < len(data); i++ {
			data[i] = math.Exp(data[i] - maxVal)
			sum += data[i]
		}
		
		for i := 0; i < len(data); i++ {
			data[i] /= sum
		}
	}
	
	return result
}

func SoftmaxDense(logits *tensor.Dense) *tensor.Dense {
	return Softmax(logits)
}

func GetMaxIndexRow(m *tensor.Dense, row int) int {
	shape := m.Shape()
	cols := shape[len(shape)-1]
	
	data := m.Data().([]float64)
	
	// Calculate the starting index for this row
	startIdx := row * cols
	
	maxIdx := 0
	maxVal := data[startIdx]
	
	for j := 1; j < cols; j++ {
		val := data[startIdx+j]

		if val > maxVal {
			maxVal = val
			maxIdx = j
		}
	}
	return maxIdx
}
