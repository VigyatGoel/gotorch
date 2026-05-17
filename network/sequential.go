package network

import (
	"github.com/VigyatGoel/gotorch/layer"
	"github.com/VigyatGoel/gotorch/optimizer"
	"github.com/VigyatGoel/gotorch/persistence"
	"gonum.org/v1/gonum/mat"
)

type Sequential struct {
	Layers    []layer.Layer
	Optimizer optimizer.Optimizer
}

func (s *Sequential) GetLayers() []layer.Layer {
	return s.Layers
}

func (s *Sequential) GetOptimizer() optimizer.Optimizer {
	return s.Optimizer
}

func NewSequential(layers ...layer.Layer) *Sequential {
	return &Sequential{Layers: layers}
}

func (s *Sequential) SetOptimizer(opt optimizer.Optimizer) {
	s.Optimizer = opt
}

func (s *Sequential) Add(l layer.Layer) {
	s.Layers = append(s.Layers, l)
}

func (s *Sequential) Forward(input *mat.Dense) *mat.Dense {
	output := input
	for _, l := range s.Layers {
		output = l.Forward(output)
	}
	return output
}

func (s *Sequential) Backward(gradOutput *mat.Dense) *mat.Dense {
	for i := len(s.Layers) - 1; i >= 0; i-- {
		gradOutput = s.Layers[i].Backward(gradOutput)
	}
	if s.Optimizer != nil {
		for _, l := range s.Layers {
			if w, g := l.GetWeights(), l.GetGradients(); w != nil && g != nil {
				l.UpdateWeights(s.Optimizer.Step(w, g))
			}
			if b, bg := l.GetBiases(), l.GetBiasGradients(); b != nil && bg != nil {
				l.UpdateBiases(s.Optimizer.StepBias(b, bg))
			}
		}
	}
	return gradOutput
}

func (s *Sequential) Predict(input *mat.Dense) *mat.Dense {
	return s.Forward(input)
}

func (s *Sequential) Train() {
	for _, l := range s.Layers {
		if dropout, ok := l.(*layer.Dropout); ok {
			dropout.SetTraining(true)
		}
	}
}

func (s *Sequential) Eval() {
	for _, l := range s.Layers {
		if dropout, ok := l.(*layer.Dropout); ok {
			dropout.SetTraining(false)
		}
	}
}

func (s *Sequential) ClearCache() {
	for _, l := range s.Layers {
		if clearable, ok := l.(interface{ ClearCache() }); ok {
			clearable.ClearCache()
		}
	}
}

func (s *Sequential) Save(filePath string) error {
	return persistence.SaveModel(s, filePath)
}

func Load(filePath string) (*Sequential, error) {
	modelData, err := persistence.LoadModelData(filePath)
	if err != nil {
		return nil, err
	}
	return &Sequential{
		Layers:    modelData.Layers,
		Optimizer: modelData.Optimizer,
	}, nil
}
