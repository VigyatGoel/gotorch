package persistence

import (
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
	"reflect"

	"strings"

	layer "github.com/VigyatGoel/gotorch/layers"
	"github.com/VigyatGoel/gotorch/optimizer"
	"gorgonia.org/tensor"
)

func tensorDenseToSerializable(t *tensor.Dense) (data []float64, shape []int) {
	if t == nil {
		return nil, nil
	}
	shape = t.Shape()
	data = make([]float64, len(t.Data().([]float64)))
	copy(data, t.Data().([]float64))
	return data, shape
}

func serializableToTensorDense(data []float64, shape []int) *tensor.Dense {
	if data == nil || len(shape) == 0 {
		return nil
	}
	// Create a copy of the data to avoid sharing memory
	dataCopy := make([]float64, len(data))
	copy(dataCopy, data)
	return tensor.New(tensor.WithShape(shape...), tensor.WithBacking(dataCopy))
}

type ModelInterface interface {
	GetLayers() []layer.Layer
	GetOptimizer() optimizer.Optimizer
}

type LayerConfig struct {
	Type        string    `json:"type"`
	InFeatures  int       `json:"in_features,omitempty"`
	OutFeatures int       `json:"out_features,omitempty"`
	Weights     []float64 `json:"weights,omitempty"`
	WeightShape []int     `json:"weight_shape,omitempty"`
	Biases      []float64 `json:"biases,omitempty"`
	BiasShape   []int     `json:"bias_shape,omitempty"`
	Alpha       float64   `json:"alpha,omitempty"`
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
			wData, wShape := tensorDenseToSerializable(linear.GetWeights())
			bData, bShape := tensorDenseToSerializable(linear.GetBiases())
			if len(wShape) == 2 {
				layerConfig.InFeatures = wShape[0]
				layerConfig.OutFeatures = wShape[1]
			}
			layerConfig.Weights = wData
			layerConfig.WeightShape = wShape
			layerConfig.Biases = bData
			layerConfig.BiasShape = bShape

		case "LeakyReLU":
			leaky, _ := l.(*layer.LeakyReLU)
			layerConfig.Alpha = leaky.Alpha

		case "ReLU", "Sigmoid", "Softmax", "SiLU":
		}

		modelConfig.Layers[i] = layerConfig
	}

	dirPath := filepath.Dir(filePath)
	if err := os.MkdirAll(dirPath, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create directory: %v", err)
	}

	f, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create model file: %v", err)
	}
	defer f.Close()

	encoder := gob.NewEncoder(f)
	if err := encoder.Encode(modelConfig); err != nil {
		return fmt.Errorf("failed to encode model configuration: %v", err)
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

	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open model file: %v", err)
	}
	defer f.Close()

	var modelConfig ModelConfig
	decoder := gob.NewDecoder(f)
	if err := decoder.Decode(&modelConfig); err != nil {
		return nil, fmt.Errorf("failed to decode model configuration: %v", err)
	}

	modelData := &ModelData{
		Layers: make([]layer.Layer, 0, len(modelConfig.Layers)),
	}

	for _, layerConfig := range modelConfig.Layers {
		var newLayer layer.Layer

		switch layerConfig.Type {
		case "Linear":
			linear := layer.NewLinear(layerConfig.InFeatures, layerConfig.OutFeatures)
			if layerConfig.Weights != nil && len(layerConfig.WeightShape) > 0 {
				linear.UpdateWeights(serializableToTensorDense(layerConfig.Weights, layerConfig.WeightShape))
			}
			if layerConfig.Biases != nil && len(layerConfig.BiasShape) > 0 {
				linear.UpdateBiases(serializableToTensorDense(layerConfig.Biases, layerConfig.BiasShape))
			}
			newLayer = linear

		case "LeakyReLU":
			leaky := layer.NewLeakyReLU(layerConfig.Alpha)
			newLayer = leaky

		case "ReLU":
			newLayer = layer.NewReLU()

		case "Sigmoid":
			newLayer = layer.NewSigmoid()

		case "Softmax":
			newLayer = layer.NewSoftmax()

		case "SiLU":
			newLayer = layer.NewSiLU()

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
