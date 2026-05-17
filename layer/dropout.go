package layer

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Dropout implements dropout regularization to prevent overfitting
type Dropout struct {
	DropoutRate float64
	training    bool
	mask        *mat.Dense
}

func NewDropout(p float64) *Dropout {
	return &Dropout{DropoutRate: p, training: true}
}

func (d *Dropout) SetTraining(training bool) {
	d.training = training
}

func (d *Dropout) Forward(x *mat.Dense) *mat.Dense {
	if !d.training || d.DropoutRate == 0.0 {
		return mat.DenseCopyOf(x)
	}
	rows, cols := x.Dims()
	xData := x.RawMatrix().Data
	resultData := make([]float64, rows*cols)
	maskData := make([]float64, rows*cols)
	scale := 1.0 / (1.0 - d.DropoutRate)
	for i, v := range xData {
		if rand.Float64() < d.DropoutRate {
			maskData[i] = 0.0
			resultData[i] = 0.0
		} else {
			maskData[i] = scale
			resultData[i] = v * scale
		}
	}
	d.mask = mat.NewDense(rows, cols, maskData)
	return mat.NewDense(rows, cols, resultData)
}

func (d *Dropout) Backward(gradOutput *mat.Dense) *mat.Dense {
	if d.mask == nil {
		return mat.DenseCopyOf(gradOutput)
	}
	rows, cols := gradOutput.Dims()
	result := mat.NewDense(rows, cols, nil)
	result.MulElem(gradOutput, d.mask)
	return result
}

func (d *Dropout) GetWeights() *mat.Dense       { return nil }
func (d *Dropout) GetGradients() *mat.Dense     { return nil }
func (d *Dropout) UpdateWeights(w *mat.Dense)   {}
func (d *Dropout) GetBiases() *mat.Dense        { return nil }
func (d *Dropout) GetBiasGradients() *mat.Dense { return nil }
func (d *Dropout) UpdateBiases(b *mat.Dense)    {}
func (d *Dropout) ClearCache()                  { d.mask = nil }
