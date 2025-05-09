package utils

import (
	"math"

	"gonum.org/v1/gonum/mat"
)


func Zeros(rows, cols int) *mat.Dense {
	return mat.NewDense(rows, cols, nil)
}

func Ones(rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = 1.0
	}
	return mat.NewDense(rows, cols, data)
}

func Dot(a, b *mat.Dense) *mat.Dense {
	aRows, aCols := a.Dims()
	bRows, bCols := b.Dims()

	if aCols != bRows {
		panic("Dimension mismatch for matrix multiplication")
	}

	result := mat.NewDense(aRows, bCols, nil)
	result.Mul(a, b)
	return result
}

func Add(a, b *mat.Dense) *mat.Dense {
	aRows, aCols := a.Dims()
	bRows, bCols := b.Dims()

	if bRows == 1 && aRows > 1 {
		result := mat.NewDense(aRows, aCols, nil)
		aRaw := a.RawMatrix()
		bRaw := b.RawMatrix()
		resultRaw := result.RawMatrix()
		for i := 0; i < aRows; i++ {
			for j := 0; j < aCols; j++ {
				resultRaw.Data[i*resultRaw.Stride+j] = aRaw.Data[i*aRaw.Stride+j] + bRaw.Data[j]
			}
		}
		return result
	}

	if aRows != bRows || aCols != bCols {
		panic("Dimension mismatch for matrix addition")
	}

	result := mat.NewDense(aRows, aCols, nil)
	result.Add(a, b)
	return result
}

func Transpose(a *mat.Dense) *mat.Dense {
	rows, cols := a.Dims()
	result := mat.NewDense(cols, rows, nil)
	aRaw := a.RawMatrix()
	resultRaw := result.RawMatrix()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			resultRaw.Data[j*resultRaw.Stride+i] = aRaw.Data[i*aRaw.Stride+j]
		}
	}
	return result
}

func ApplyFunc(a *mat.Dense, f func(float64) float64) *mat.Dense {
	rows, cols := a.Dims()
	result := mat.NewDense(rows, cols, nil)
	result.Apply(func(_, _ int, v float64) float64 {
		return f(v)
	}, a)
	return result
}

func ApplyFuncDense(a *mat.Dense, f func(float64) float64) *mat.Dense {
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

func Softmax(logits *mat.Dense) *mat.Dense {
	rows, cols := logits.Dims()
	result := mat.NewDense(rows, cols, nil)
	logitsRaw := logits.RawMatrix()
	resultRaw := result.RawMatrix()
	for i := 0; i < rows; i++ {
		maxVal := logitsRaw.Data[i*logitsRaw.Stride]
		for j := 1; j < cols; j++ {
			val := logitsRaw.Data[i*logitsRaw.Stride+j]
			if val > maxVal {
				maxVal = val
			}
		}
		sum := 0.0
		expValues := make([]float64, cols)
		for j := 0; j < cols; j++ {
			expValues[j] = math.Exp(logitsRaw.Data[i*logitsRaw.Stride+j] - maxVal)
			sum += expValues[j]
		}
		for j := 0; j < cols; j++ {
			resultRaw.Data[i*resultRaw.Stride+j] = expValues[j] / sum
		}
	}
	return result
}

func SoftmaxDense(logits *mat.Dense) *mat.Dense {
	return Softmax(logits)
}

func GetMaxIndexRow(m *mat.Dense, row int) int {
	_, cols := m.Dims()
	maxIdx := 0
	maxVal := m.At(row, 0)
	for j := 1; j < cols; j++ {
		val := m.At(row, j)
		if val > maxVal {
			maxVal = val
			maxIdx = j
		}
	}
	return maxIdx
}
