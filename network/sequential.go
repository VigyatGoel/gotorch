package network

import (
	layer "github.com/VigyatGoel/gotorch/layers"
)

type Sequential struct {
	Layers []layer.Layer
}

func NewSequential(layers ...layer.Layer) *Sequential {
	return &Sequential{
		Layers: layers,
	}
}

func (s *Sequential) Add(layer layer.Layer) {
	s.Layers = append(s.Layers, layer)
}

func (s *Sequential) Forward(input [][]float64) [][]float64 {
	output := input
	for _, layer := range s.Layers {
		output = layer.Forward(output)
	}
	return output
}

func (s *Sequential) Backward(gradOutput [][]float64, learningRate float64) [][]float64 {
	for i := len(s.Layers) - 1; i >= 0; i-- {
		gradOutput = s.Layers[i].Backward(gradOutput, learningRate)
	}
	return gradOutput
}

func (s *Sequential) Predict(input [][]float64) [][]float64 {
	return s.Forward(input)
}
