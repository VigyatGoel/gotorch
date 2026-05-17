package layer

import "gonum.org/v1/gonum/mat"

// LeakyReLU implements Leaky ReLU activation: f(x) = x if x > 0, else alpha * x
type LeakyReLU struct {
	Alpha  float64
	input  *mat.Dense
	output *mat.Dense
}

func NewLeakyReLU(alpha float64) *LeakyReLU {
	return &LeakyReLU{Alpha: alpha}
}

func (l *LeakyReLU) Forward(x *mat.Dense) *mat.Dense {
	l.input = x
	l.output = mat.DenseCopyOf(x)
	data := l.output.RawMatrix().Data
	for i, v := range data {
		if v < 0 {
			data[i] = v * l.Alpha
		}
	}
	return l.output
}

func (l *LeakyReLU) Backward(gradOutput *mat.Dense) *mat.Dense {
	rows, cols := l.input.Dims()
	inputData := l.input.RawMatrix().Data
	gradData := gradOutput.RawMatrix().Data
	resultData := make([]float64, rows*cols)
	for i, v := range inputData {
		if v > 0 {
			resultData[i] = gradData[i]
		} else {
			resultData[i] = l.Alpha * gradData[i]
		}
	}
	return mat.NewDense(rows, cols, resultData)
}

func (l *LeakyReLU) GetWeights() *mat.Dense       { return nil }
func (l *LeakyReLU) GetGradients() *mat.Dense     { return nil }
func (l *LeakyReLU) UpdateWeights(w *mat.Dense)   {}
func (l *LeakyReLU) GetBiases() *mat.Dense        { return nil }
func (l *LeakyReLU) GetBiasGradients() *mat.Dense { return nil }
func (l *LeakyReLU) UpdateBiases(b *mat.Dense)    {}
func (l *LeakyReLU) ClearCache()                  { l.input = nil; l.output = nil }
