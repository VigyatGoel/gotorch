# GoTorch

[![Go Version](https://img.shields.io/badge/Go-1.24-blue.svg)](https://go.dev/)
[![Go Reference](https://pkg.go.dev/badge/github.com/VigyatGoel/gotorch.svg)](https://pkg.go.dev/github.com/VigyatGoel/gotorch)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

GoTorch is a deep learning framework implemented in pure Go, designed for simplicity and educational purposes. It provides the essential building blocks for creating and training neural networks without external dependencies.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Examples](#examples)
- [Customization](#customization)
- [Model Persistence](#model-persistence)
- [Documentation](#documentation)
- [Future](#future)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Pure Go Implementation**: No external C/C++ dependencies or bindings
- **Key Neural Network Components**:
  - Linear (Dense) layers
  - Activation functions (ReLU, Sigmoid, Softmax)
  - Loss functions (Cross-Entropy, MSE)
  - Optimizers: SGD(with momentum) and Adam supported
  - Sequential model architecture
- **Model Persistence**:
  - Save trained models to disk in JSON-based .gth format
  - Load models to perform inference without retraining
  - Preserves layers, weights, biases, and optimizer configurations
- **Data Processing**:
  - CSV data loading with automatic feature extraction
  - Training/testing data splitting
  - Classification and regression support
- **Modular Design**: Easy to extend with new layers and components

## Installation

```bash
# Option 1: Use go get to install the package
go get github.com/VigyatGoel/gotorch

# Option 2: Clone the repository
git clone https://github.com/VigyatGoel/gotorch.git
```

## Quick Start

```go
package main

import (
	"fmt"
	"log"

	"github.com/VigyatGoel/gotorch/data"
	layer "github.com/VigyatGoel/gotorch/layers"
	"github.com/VigyatGoel/gotorch/loss"
	"github.com/VigyatGoel/gotorch/network"
	"github.com/VigyatGoel/gotorch/optimizer"
)

const (
	BatchSize = 32
)

func main() {
	dataLoader := data.NewDataLoader("cmd/iris.csv", data.Classification, BatchSize)
	err := dataLoader.Load()
	if err != nil {
		log.Fatalf("Error loading data: %v", err)
	}

	dataLoader.NormalizeFeatures()

	numFeatures := dataLoader.NumFeatures()
	if numFeatures == 0 {
		log.Fatalf("Could not determine number of features from the data.")
	}
	fmt.Printf("Detected %d features from the dataset.\n", numFeatures)

	x_train, y_train, x_test, y_test := dataLoader.Split()

	model := createModel(numFeatures)
	criterion := loss.NewCrossEntropyLoss()
	epochs := 20

	fmt.Println("\nTRAINING WITH ADAM")
	adamOpt := optimizer.DefaultAdam(0.001)
	model.SetOptimizer(adamOpt)

	modelPath := "iris_model.gth"
	trainAndEvaluate(model, criterion, dataLoader, x_train, y_train, x_test, y_test, epochs, modelPath)

	loadAndUseModel(modelPath, x_test, y_test)
}

func createModel(inputFeatures int) *network.Sequential {
	return network.NewSequential(
		layer.NewLinear(inputFeatures, 128),
		layer.NewReLU(),
		layer.NewLinear(128, 64),
		layer.NewReLU(),
		layer.NewLinear(64, 32),
		layer.NewReLU(),
		layer.NewLinear(32, 3),
		layer.NewSoftmax(),
	)
}

func trainAndEvaluate(model *network.Sequential, criterion *loss.CrossEntropyLoss,
	dataLoader *data.DataLoader,
	x_train, y_train, x_test, y_test [][]float64, epochs int, modelPath string) {
	for epoch := range epochs {
		epochLoss := 0.0
		batches := dataLoader.GetBatches(x_train, y_train, epoch)

		for _, batch := range batches {
			preds := model.Forward(batch.Features)
			lossVal := criterion.Forward(preds, batch.Targets)
			grad := criterion.Backward()
			model.Backward(grad)
			epochLoss += lossVal
		}

		avgEpochLoss := epochLoss / float64(len(batches))
		fmt.Printf("Epoch [%d/%d] Average Loss: %.4f\n", epoch+1, epochs, avgEpochLoss)
	}

	preds := model.Predict(x_test)
	correct := 0
	for i := range x_test {
		predictedClass := getMaxIndex(preds[i])
		actualClass := getMaxIndex(y_test[i])
		if predictedClass == actualClass {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(x_test)) * 100
	fmt.Printf("Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, len(x_test))

	if modelPath != "" {
		err := model.Save(modelPath)
		if err != nil {
			fmt.Printf("Error saving model: %v\n", err)
		} else {
			fmt.Printf("Model saved successfully to %s\n", modelPath)
		}
	}
}

func loadAndUseModel(modelPath string, x_test, y_test [][]float64) {
	fmt.Printf("\nLoading model from %s\n", modelPath)
	loadedModel, err := network.Load(modelPath)
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		return
	}

	fmt.Println("Model loaded successfully! Evaluating...")

	preds := loadedModel.Predict(x_test)
	correct := 0
	for i := range x_test {
		predictedClass := getMaxIndex(preds[i])
		actualClass := getMaxIndex(y_test[i])
		if predictedClass == actualClass {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(x_test)) * 100
	fmt.Printf("Loaded model accuracy: %.2f%% (%d/%d)\n", accuracy, correct, len(x_test))
}

func getMaxIndex(values []float64) int {
	maxIdx := 0
	maxVal := values[0]

	for i, val := range values {
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}

	return maxIdx
}
```

## Architecture

GoTorch's architecture consists of the following key components:

### Layers

- **Linear**: Fully connected layer with weights and biases
- **ReLU**: Rectified Linear Unit activation function
- **Sigmoid**: Sigmoid activation function
- **Softmax**: Softmax activation for multi-class outputs

### Network

- **Sequential**: Container for stacking layers in sequence

### Loss Functions

- **CrossEntropyLoss**: For classification tasks
- **MSELoss**: Mean Squared Error for regression tasks

### Data Handling

- **DataLoader**: For loading and preprocessing CSV data

## Examples

The project includes an example classification model using the Iris dataset:

```bash
# Train a classification model on Iris dataset
go run examples/train_basic.go
```

## Customization

### Creating Custom Layers

To implement a custom layer, implement the Layer interface:

```go
type Layer interface {
    Forward(input [][]float64) [][]float64
    Backward(dout [][]float64, learningRate float64) [][]float64
}
```

## Model Persistence

GoTorch provides a straightforward way to save and load trained models:

### Saving Models

```go
// After training your model
modelPath := "my_model.gth"
err := model.Save(modelPath)
if err != nil {
    log.Fatalf("Error saving model: %v", err)
}
```

### Loading Models

```go
// Load a previously saved model
loadedModel, err := network.Load("my_model.gth")
if err != nil {
    log.Fatalf("Error loading model: %v", err)
}

// Use the loaded model for inference
predictions := loadedModel.Predict(newData)
```

### Model Format

Models are saved in a JSON-based `.gth` format that includes:
- Layer types and configurations
- Weights and biases for trainable layers
- Optimizer configuration (type, learning rate, and other parameters)

## Documentation

- [Layer Documentation](layers/)
- [Loss Functions](loss/)
- [Neural Networks](network/)
- [Data Processing](data/)

## Future

GoTorch is under active development, with plans to incorporate the following features and improvements:

- **Advanced Neural Network Architectures**:
  - Convolutional Neural Networks (CNN) for image processing
  - Recurrent Neural Networks (RNN) for sequential data
  - LSTM and GRU variants for improved sequence modeling

- **Performance Optimizations**:
  - Multi-threading support for parallel computations
  - Optimized matrix operations for faster training
  - GPU acceleration via CUDA bindings and Metal (Apple silicon)

- **Extended Functionality**:
  - Additional optimization algorithms (AdamW, RMSProp)
  - Regularization techniques (Dropout, L1/L2 regularization)
  - Learning rate schedulers
  - Early stopping and model checkpointing

- **Expanded Data Handling**:
  - Support for more data formats (JSON, parquet)
  - Image data preprocessing utilities
  - Text tokenization and embedding features
  - Time series data handling

- **Developer Experience**:
  - Improved documentation with more examples
  - Interactive visualization tools for model inspection
  - Integration with popular Go machine learning ecosystems

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Inspired by PyTorch's design philosophy
- Thanks to the Go community for the excellent standard library
