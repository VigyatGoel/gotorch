package utils

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Zeros(r, c int) *mat.Dense {
	return mat.NewDense(r, c, nil)
}

func Ones(r, c int) *mat.Dense {
	data := make([]float64, r*c)
	for i := range data {
		data[i] = 1.0
	}
	return mat.NewDense(r, c, data)
}

func Dot(a, b *mat.Dense) *mat.Dense {
	ra, _ := a.Dims()
	_, cb := b.Dims()
	result := mat.NewDense(ra, cb, nil)
	result.Mul(a, b)
	return result
}

func Add(a, b *mat.Dense) *mat.Dense {
	r, c := a.Dims()
	result := mat.NewDense(r, c, nil)
	result.Add(a, b)
	return result
}

func Transpose(a *mat.Dense) *mat.Dense {
	return mat.DenseCopyOf(a.T())
}

func ApplyFunc(a *mat.Dense, f func(float64) float64) *mat.Dense {
	r, c := a.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(func(_, _ int, v float64) float64 { return f(v) }, a)
	return result
}

func Softmax(logits *mat.Dense) *mat.Dense {
	result := mat.DenseCopyOf(logits)
	data := result.RawMatrix().Data
	rows, cols := result.Dims()
	for i := 0; i < rows; i++ {
		maxVal := data[i*cols]
		for j := 1; j < cols; j++ {
			if data[i*cols+j] > maxVal {
				maxVal = data[i*cols+j]
			}
		}
		sum := 0.0
		for j := 0; j < cols; j++ {
			data[i*cols+j] = math.Exp(data[i*cols+j] - maxVal)
			sum += data[i*cols+j]
		}
		for j := 0; j < cols; j++ {
			data[i*cols+j] /= sum
		}
	}
	return result
}

func GetMaxIndexRow(m *mat.Dense, row int) int {
	_, cols := m.Dims()
	data := m.RawMatrix().Data
	startIdx := row * cols
	maxIdx := 0
	maxVal := data[startIdx]
	for j := 1; j < cols; j++ {
		if data[startIdx+j] > maxVal {
			maxVal = data[startIdx+j]
			maxIdx = j
		}
	}
	return maxIdx
}
