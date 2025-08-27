package network

import (
	"github.com/VigyatGoel/gotorch/layer"
	"github.com/VigyatGoel/gotorch/optimizer"
	"github.com/VigyatGoel/gotorch/persistence"
	"gorgonia.org/tensor"
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

func (s *Sequential) Forward(input *tensor.Dense) *tensor.Dense {
	output := input
	for _, layer := range s.Layers {
		output = layer.Forward(output)
	}
	return output
}

func (s *Sequential) Backward(gradOutput *tensor.Dense) *tensor.Dense {
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

func (s *Sequential) Predict(input *tensor.Dense) *tensor.Dense {
	return s.Forward(input)
}

// ClearCache clears cached data from all layers to prevent memory leaks
func (s *Sequential) ClearCache() {
	for _, l := range s.Layers {
		// Try to call ClearCache if the layer has it
		if clearable, ok := l.(interface{ ClearCache() }); ok {
			clearable.ClearCache()
		}
		// For layers that store input data, explicitly clear it
		if conv, ok := l.(*layer.Conv2D); ok {
			conv.ClearCache()
		} else if maxpool, ok := l.(*layer.MaxPool2D); ok {
			maxpool.ClearCache()
		} else if flatten, ok := l.(*layer.Flatten); ok {
			flatten.ClearCache()
		} else if relu, ok := l.(*layer.ReLU); ok {
			relu.ClearCache()
		} else if softmax, ok := l.(*layer.Softmax); ok {
			softmax.ClearCache()
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

	model := &Sequential{
		Layers:    modelData.Layers,
		Optimizer: modelData.Optimizer,
	}

	return model, nil
}
