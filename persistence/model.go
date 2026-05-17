package persistence

import (
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/VigyatGoel/gotorch/layer"
	"github.com/VigyatGoel/gotorch/optimizer"
	"gonum.org/v1/gonum/mat"
)

func matDenseToSerializable(m *mat.Dense) (data []float64, shape []int) {
	if m == nil {
		return nil, nil
	}
	rows, cols := m.Dims()
	raw := m.RawMatrix().Data
	data = make([]float64, len(raw))
	copy(data, raw)
	return data, []int{rows, cols}
}

func serializableToMatDense(data []float64, shape []int) *mat.Dense {
	if data == nil || len(shape) != 2 {
		return nil
	}
	cp := make([]float64, len(data))
	copy(cp, data)
	return mat.NewDense(shape[0], shape[1], cp)
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
	DropoutRate float64   `json:"dropout_rate,omitempty"`
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
		modelConfig.Optimizer = getOptimizerConfig(model.GetOptimizer())
	}

	for i, l := range model.GetLayers() {
		layerType := reflect.TypeOf(l).Elem().Name()
		layerConfig := LayerConfig{Type: layerType}

		switch typedLayer := l.(type) {
		case *layer.Linear:
			wData, wShape := matDenseToSerializable(typedLayer.GetWeights())
			bData, bShape := matDenseToSerializable(typedLayer.GetBiases())
			if len(wShape) == 2 {
				layerConfig.InFeatures = wShape[0]
				layerConfig.OutFeatures = wShape[1]
			}
			layerConfig.Weights = wData
			layerConfig.WeightShape = wShape
			layerConfig.Biases = bData
			layerConfig.BiasShape = bShape
		case *layer.LeakyReLU:
			layerConfig.Alpha = typedLayer.Alpha
		case *layer.Dropout:
			layerConfig.DropoutRate = typedLayer.DropoutRate
		}

		modelConfig.Layers[i] = layerConfig
	}

	if err := os.MkdirAll(filepath.Dir(filePath), os.ModePerm); err != nil {
		return fmt.Errorf("failed to create directory: %v", err)
	}

	f, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create model file: %v", err)
	}
	defer f.Close()

	if err := gob.NewEncoder(f).Encode(modelConfig); err != nil {
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
	if err := gob.NewDecoder(f).Decode(&modelConfig); err != nil {
		return nil, fmt.Errorf("failed to decode model configuration: %v", err)
	}

	modelData := &ModelData{Layers: make([]layer.Layer, 0, len(modelConfig.Layers))}

	for _, cfg := range modelConfig.Layers {
		var newLayer layer.Layer
		switch cfg.Type {
		case "Linear":
			linear := layer.NewLinear(cfg.InFeatures, cfg.OutFeatures)
			if cfg.Weights != nil && len(cfg.WeightShape) > 0 {
				linear.UpdateWeights(serializableToMatDense(cfg.Weights, cfg.WeightShape))
			}
			if cfg.Biases != nil && len(cfg.BiasShape) > 0 {
				linear.UpdateBiases(serializableToMatDense(cfg.Biases, cfg.BiasShape))
			}
			newLayer = linear
		case "Flatten":
			newLayer = layer.NewFlatten()
		case "LeakyReLU":
			newLayer = layer.NewLeakyReLU(cfg.Alpha)
		case "ReLU":
			newLayer = layer.NewReLU()
		case "Sigmoid":
			newLayer = layer.NewSigmoid()
		case "Softmax":
			newLayer = layer.NewSoftmax()
		case "SiLU":
			newLayer = layer.NewSiLU()
		case "Dropout":
			newLayer = layer.NewDropout(cfg.DropoutRate)
		default:
			return nil, fmt.Errorf("unsupported layer type: %s", cfg.Type)
		}
		modelData.Layers = append(modelData.Layers, newLayer)
	}

	if modelConfig.Optimizer.Type != "" {
		if opt := createOptimizer(modelConfig.Optimizer); opt != nil {
			modelData.Optimizer = opt
		}
	}
	return modelData, nil
}

func getOptimizerConfig(opt optimizer.Optimizer) OptimizerConfig {
	config := OptimizerConfig{LR: opt.GetLearningRate()}
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
	}
	return nil
}
