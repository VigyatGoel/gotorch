package network

import (
	layer "github.com/VigyatGoel/gotorch/layers"
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
	return &Sequential{
		Layers: layers,
	}
}

func (s *Sequential) SetOptimizer(opt optimizer.Optimizer) {
	s.Optimizer = opt
}

func (s *Sequential) Add(layer layer.Layer) {
	s.Layers = append(s.Layers, layer)
}

func (s *Sequential) Forward(input *mat.Dense) *mat.Dense {
	output := input
	for _, layer := range s.Layers {
		output = layer.Forward(output)
	}
	return output
}

func (s *Sequential) Backward(gradOutput *mat.Dense) *mat.Dense {
	for i := len(s.Layers) - 1; i >= 0; i-- {
		gradOutput = s.Layers[i].Backward(gradOutput)
	}

	if s.Optimizer != nil {
		for _, l := range s.Layers {
			weights := l.GetWeights()
			gradients := l.GetGradients()
			if weights != nil && gradients != nil {
				updatedWeights := s.Optimizer.Step(weights, gradients)
				l.UpdateWeights(updatedWeights)
			}

			biases := l.GetBiases()
			biasGradients := l.GetBiasGradients()
			if biases != nil && biasGradients != nil {
				updatedBiases := s.Optimizer.StepBias(biases, biasGradients)
				l.UpdateBiases(updatedBiases)
			}
		}
	}

	return gradOutput
}

func (s *Sequential) Predict(input *mat.Dense) *mat.Dense {
	return s.Forward(input)
}

func (s *Sequential) Save(filePath string) error {
	return persistence.SaveModel(s, filePath)
}

func Load(filePath string) (*Sequential, error) {
	modelData, err := persistence.LoadModelData(filePath)
	if err != nil {
		return nil, err
	}

	model := &Sequential{
		Layers:    modelData.Layers,
		Optimizer: modelData.Optimizer,
	}

	return model, nil
}
