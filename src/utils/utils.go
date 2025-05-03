package utils

import "math"

func Dot(a, b [][]float64) [][]float64 {
	m, n, p := len(a), len(b), len(b[0])
	res := make([][]float64, m)
	for i := 0; i < m; i++ {
		res[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			for k := 0; k < n; k++ {
				res[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return res
}

func Add(a, b [][]float64) [][]float64 {
	out := make([][]float64, len(a))
	for i := range a {
		out[i] = make([]float64, len(a[0]))
		for j := range a[0] {
			if len(b) == 1 {
				out[i][j] = a[i][j] + b[0][j]
			} else {
				out[i][j] = a[i][j] + b[i][j]
			}
		}
	}
	return out
}

func Transpose(a [][]float64) [][]float64 {
	m, n := len(a), len(a[0])
	res := make([][]float64, n)
	for i := range res {
		res[i] = make([]float64, m)
		for j := range res[0] {
			res[i][j] = a[j][i]
		}
	}
	return res
}

func ApplyFunc(a [][]float64, f func(float64) float64) [][]float64 {
	out := make([][]float64, len(a))
	for i := range a {
		out[i] = make([]float64, len(a[0]))
		for j := range a[0] {
			out[i][j] = f(a[i][j])
		}
	}
	return out
}

func Zeros(rows, cols int) [][]float64 {
	out := make([][]float64, rows)
	for i := range out {
		out[i] = make([]float64, cols)
		for j := range out[i] {
			out[i][j] = 0.0
		}
	}
	return out
}

func Ones(rows, cols int) [][]float64 {
	out := make([][]float64, rows)
	for i := range out {
		out[i] = make([]float64, cols)
		for j := range out[i] {
			out[i][j] = 1.0
		}
	}
	return out
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

func Softmax(logits [][]float64) [][]float64 {
	out := make([][]float64, len(logits))
	for i := range logits {
		maxLogit := logits[i][0]
		for _, val := range logits[i] {
			if val > maxLogit {
				maxLogit = val
			}
		}

		sum := 0.0
		out[i] = make([]float64, len(logits[0]))
		for j := range logits[0] {
			out[i][j] = math.Exp(logits[i][j] - maxLogit)
			sum += out[i][j]
		}
		for j := range logits[0] {
			out[i][j] /= sum
		}
	}
	return out
}
