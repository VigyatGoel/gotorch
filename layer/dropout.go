package layer

import (
	"gorgonia.org/tensor"
	"math/rand"
)

// Dropout implements dropout regularization to prevent overfitting
type Dropout struct {
	p        float64       // dropout probability (0 to 1)
	training bool          // training mode flag
	mask     *tensor.Dense // binary mask for dropped neurons
}

// NewDropout creates a new dropout layer with given probability
func NewDropout(p float64) *Dropout {
	return &Dropout{
		p:        p,
		training: true,
	}
}

// SetTraining enables/disables dropout (only active during training)
func (d *Dropout) SetTraining(training bool) {
	d.training = training
}

// Forward applies dropout mask during training, scales by 1/(1-p)
func (d *Dropout) Forward(x *tensor.Dense) *tensor.Dense {
	if !d.training || d.p == 0.0 {
		return x.Clone().(*tensor.Dense)
	}

	result := x.Clone().(*tensor.Dense)
	data := result.Data().([]float64)
	maskData := make([]float64, len(data))
	scale := 1.0 / (1.0 - d.p)
	for i := range data {
		if rand.Float64() < d.p {
			maskData[i] = 0.0
			data[i] = 0.0
		} else {
			maskData[i] = scale
			data[i] *= scale
		}
	}
	d.mask = tensor.New(tensor.WithShape(x.Shape()...), tensor.WithBacking(maskData))
	return result
}

// Backward applies the same mask to gradients
func (d *Dropout) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	if d.mask == nil {
		return gradOutput.Clone().(*tensor.Dense)
	}
	result, _ := tensor.Mul(gradOutput, d.mask)
	return result.(*tensor.Dense)
}

func (d *Dropout) GetWeights() *tensor.Dense                 { return nil }
func (d *Dropout) GetGradients() *tensor.Dense               { return nil }
func (d *Dropout) UpdateWeights(weightsUpdate *tensor.Dense) {}
func (d *Dropout) GetBiases() *tensor.Dense                  { return nil }
func (d *Dropout) GetBiasGradients() *tensor.Dense           { return nil }
func (d *Dropout) UpdateBiases(biasUpdate *tensor.Dense)     {}

// ClearCache releases dropout mask to prevent memory leaks
func (d *Dropout) ClearCache() {
	d.mask = nil
}
