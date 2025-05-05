package layer

import (
	"github.com/VigyatGoel/gotorch/utils"
	"gonum.org/v1/gonum/mat"
)

type Softmax struct {
	output *mat.Dense
}

func NewSoftmax() *Softmax {
	return &Softmax{}
}

func (s *Softmax) Forward(x *mat.Dense) *mat.Dense {
	s.output = utils.SoftmaxDense(x)
	return s.output
}

func (s *Softmax) Backward(gradOutput *mat.Dense) *mat.Dense {
	return gradOutput
}

func (s *Softmax) GetWeights() *mat.Dense {
	return nil
}

func (s *Softmax) GetGradients() *mat.Dense {
	return nil
}

func (s *Softmax) UpdateWeights(weightsUpdate *mat.Dense) {
}

func (s *Softmax) GetBiases() *mat.Dense {
	return nil
}

func (s *Softmax) GetBiasGradients() *mat.Dense {
	return nil
}

func (s *Softmax) UpdateBiases(biasUpdate *mat.Dense) {
}
