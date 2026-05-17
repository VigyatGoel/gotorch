package layer

import "gonum.org/v1/gonum/mat"

// ReLU implements Rectified Linear Unit activation: f(x) = max(0, x)
type ReLU struct {
	input *mat.Dense
}

func NewReLU() *ReLU {
	return &ReLU{}
}

func (r *ReLU) Forward(x *mat.Dense) *mat.Dense {
	r.input = x
	result := mat.DenseCopyOf(x)
	data := result.RawMatrix().Data
	for i, v := range data {
		if v < 0 {
			data[i] = 0
		}
	}
	return result
}

func (r *ReLU) Backward(gradOutput *mat.Dense) *mat.Dense {
	rows, cols := r.input.Dims()
	inputData := r.input.RawMatrix().Data
	gradData := gradOutput.RawMatrix().Data
	resultData := make([]float64, rows*cols)
	for i, v := range inputData {
		if v > 0 {
			resultData[i] = gradData[i]
		}
	}
	return mat.NewDense(rows, cols, resultData)
}

func (r *ReLU) GetWeights() *mat.Dense       { return nil }
func (r *ReLU) GetGradients() *mat.Dense     { return nil }
func (r *ReLU) UpdateWeights(w *mat.Dense)   {}
func (r *ReLU) GetBiases() *mat.Dense        { return nil }
func (r *ReLU) GetBiasGradients() *mat.Dense { return nil }
func (r *ReLU) UpdateBiases(b *mat.Dense)    {}
func (r *ReLU) ClearCache()                  { r.input = nil }
