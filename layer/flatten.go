package layer

import "gonum.org/v1/gonum/mat"

// Flatten reshapes multi-dimensional input to 2D (batch_size, features)
type Flatten struct {
	inputRows int
	inputCols int
}

func NewFlatten() *Flatten {
	return &Flatten{}
}

func (f *Flatten) Forward(input *mat.Dense) *mat.Dense {
	rows, cols := input.Dims()
	f.inputRows = rows
	f.inputCols = cols
	data := input.RawMatrix().Data
	cp := make([]float64, len(data))
	copy(cp, data)
	return mat.NewDense(rows, cols, cp)
}

func (f *Flatten) Backward(gradOutput *mat.Dense) *mat.Dense {
	data := gradOutput.RawMatrix().Data
	cp := make([]float64, len(data))
	copy(cp, data)
	return mat.NewDense(f.inputRows, f.inputCols, cp)
}

func (f *Flatten) GetWeights() *mat.Dense       { return nil }
func (f *Flatten) GetGradients() *mat.Dense     { return nil }
func (f *Flatten) UpdateWeights(w *mat.Dense)   {}
func (f *Flatten) GetBiases() *mat.Dense        { return nil }
func (f *Flatten) GetBiasGradients() *mat.Dense { return nil }
func (f *Flatten) UpdateBiases(b *mat.Dense)    {}
func (f *Flatten) ClearCache()                  { f.inputRows = 0; f.inputCols = 0 }
