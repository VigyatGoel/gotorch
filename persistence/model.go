package persistence

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"

	layer "github.com/VigyatGoel/gotorch/layers"
	"github.com/VigyatGoel/gotorch/optimizer"
	"strings"
)

type ModelInterface interface {
	GetLayers() []layer.Layer
	GetOptimizer() optimizer.Optimizer
}

type LayerConfig struct {
	Type        string        `json:"type"`
	InFeatures  int           `json:"in_features,omitempty"`
	OutFeatures int           `json:"out_features,omitempty"`
	Weights     [][][]float64 `json:"weights,omitempty"`
	Biases      [][][]float64 `json:"biases,omitempty"`
}

type OptimizerConfig struct {
	Type     string  `json:"type"`
	LR       float64 `json:"learning_rate"`
	Beta1    float64 `json:"beta1,omitempty"`
	Beta2    float64 `json:"beta2,omitempty"`
	Epsilon  float64 `json:"epsilon,omitempty"`
	Momentum float64 `json:"momentum,omitempty"`
}

type ModelConfig struct {
	Layers    []LayerConfig   `json:"layers"`
	Optimizer OptimizerConfig `json:"optimizer,omitempty"`
}

func SaveModel(model ModelInterface, filePath string) error {

	ext := filepath.Ext(filePath)

	if ext == "" {
		filePath = strings.TrimSuffix(filePath, ext) + ".gth"
	} else if ext != ".gth" {
		return fmt.Errorf("expected .gth file extension, got %s", ext)
	}

	modelConfig := ModelConfig{
		Layers: make([]LayerConfig, len(model.GetLayers())),
	}

	if model.GetOptimizer() != nil {
		optimizerConfig := getOptimizerConfig(model.GetOptimizer())
		modelConfig.Optimizer = optimizerConfig
	}

	for i, l := range model.GetLayers() {
		layerType := reflect.TypeOf(l).Elem().Name()
		layerConfig := LayerConfig{
			Type: layerType,
		}

		switch layerType {
		case "Linear":
			linear, _ := l.(*layer.Linear)
			inFeatures := len(linear.GetWeights())
			outFeatures := 0
			if inFeatures > 0 {
				outFeatures = len(linear.GetWeights()[0])
			}

			layerConfig.InFeatures = inFeatures
			layerConfig.OutFeatures = outFeatures

			layerConfig.Weights = [][][]float64{linear.GetWeights()}
			layerConfig.Biases = [][][]float64{linear.GetBiases()}

		case "ReLU", "Sigmoid", "Softmax":
		}

		modelConfig.Layers[i] = layerConfig
	}

	dirPath := filepath.Dir(filePath)
	if err := os.MkdirAll(dirPath, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create directory: %v", err)
	}

	jsonData, err := json.MarshalIndent(modelConfig, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal model configuration: %v", err)
	}

	if err := os.WriteFile(filePath, jsonData, 0644); err != nil {
		return fmt.Errorf("failed to write model to file: %v", err)
	}

	return nil
}

type ModelData struct {
	Layers    []layer.Layer
	Optimizer optimizer.Optimizer
}

func LoadModelData(filePath string) (*ModelData, error) {
	if ext := filepath.Ext(filePath); ext != ".gth" {
		return nil, fmt.Errorf("expected .gth file extension, got %s", ext)
	}

	jsonData, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read model file: %v", err)
	}

	var modelConfig ModelConfig
	if err := json.Unmarshal(jsonData, &modelConfig); err != nil {
		return nil, fmt.Errorf("failed to unmarshal model configuration: %v", err)
	}

	modelData := &ModelData{
		Layers: make([]layer.Layer, 0, len(modelConfig.Layers)),
	}

	for _, layerConfig := range modelConfig.Layers {
		var newLayer layer.Layer

		switch layerConfig.Type {
		case "Linear":
			linear := layer.NewLinear(layerConfig.InFeatures, layerConfig.OutFeatures)

			if layerConfig.Weights != nil && len(layerConfig.Weights) > 0 {
				linear.UpdateWeights(layerConfig.Weights[0])
			}

			if layerConfig.Biases != nil && len(layerConfig.Biases) > 0 {
				linear.UpdateBiases(layerConfig.Biases[0])
			}

			newLayer = linear

		case "ReLU":
			newLayer = layer.NewReLU()

		case "Sigmoid":
			newLayer = layer.NewSigmoid()

		case "Softmax":
			newLayer = layer.NewSoftmax()

		default:
			return nil, fmt.Errorf("unsupported layer type: %s", layerConfig.Type)
		}

		modelData.Layers = append(modelData.Layers, newLayer)
	}

	if modelConfig.Optimizer.Type != "" {
		opt := createOptimizer(modelConfig.Optimizer)
		if opt != nil {
			modelData.Optimizer = opt
		}
	}

	return modelData, nil
}

func getOptimizerConfig(opt optimizer.Optimizer) OptimizerConfig {
	config := OptimizerConfig{
		LR: opt.GetLearningRate(),
	}

	switch o := opt.(type) {
	case *optimizer.Adam:
		config.Type = "Adam"
		config.Beta1 = o.Beta1
		config.Beta2 = o.Beta2
		config.Epsilon = o.Epsilon
	case *optimizer.SGDMomentum:
		config.Type = "SGDMomentum"
		config.Momentum = o.Momentum
	case *optimizer.SGD:
		config.Type = "SGD"
	}

	return config
}

func createOptimizer(config OptimizerConfig) optimizer.Optimizer {
	switch config.Type {
	case "Adam":
		return optimizer.NewAdam(config.LR, config.Beta1, config.Beta2, config.Epsilon)
	case "SGDMomentum":
		return optimizer.NewSGDMomentum(config.LR, config.Momentum)
	case "SGD":
		return optimizer.NewSGD(config.LR)
	default:
		return nil
	}
}
