# GoTorch

[![Go Version](https://img.shields.io/badge/Go-1.24-blue.svg)](https://go.dev/)
[![Go Reference](https://pkg.go.dev/badge/github.com/VigyatGoel/gotorch.svg)](https://pkg.go.dev/github.com/VigyatGoel/gotorch)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

GoTorch is a deep learning framework implemented in pure Go, designed for simplicity and educational purposes. It provides the essential building blocks for creating and training neural networks without external dependencies.

## Features

- **Pure Go Implementation**: No external C/C++ dependencies or bindings
- **Key Neural Network Components**:
  - Linear (Dense) layers
  - Activation functions (ReLU, Sigmoid, Softmax)
  - Loss functions (Cross-Entropy, MSE)
  - Sequential model architecture
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

# Navigate to project directory
cd gotorch

# Build the project
go run cmd/main.go
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
)

func main() {
	dataLoader := data.NewDataLoader("cmd/iris.csv", data.Classification)

	err := dataLoader.Load()
	if err != nil {
		log.Fatalf("Error loading data: %v", err)
	}

	x_train, y_train, x_test, y_test := dataLoader.Split()

	model := network.NewSequential(
		layer.NewLinear(4, 128),
		layer.NewReLU(),
		layer.NewLinear(128, 64),
		layer.NewReLU(),
		layer.NewLinear(64, 32),
		layer.NewLinear(32, 3),
		layer.NewSoftmax(),
	)

	criterion := loss.NewCrossEntropyLoss()

	epochs := 100
	lr := 0.01

	for epoch := 0; epoch < epochs; epoch++ {
		preds := model.Forward(x_train)

		lossVal := criterion.Forward(preds, y_train)

		grad := criterion.Backward()
		model.Backward(grad, lr)

		fmt.Printf("Epoch [%d/%d] Loss: %.4f\n", epoch+1, epochs, lossVal)
	}

	fmt.Println("\nModel Evaluation:")
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
	fmt.Printf("\nAccuracy: %.2f%% (%d/%d)\n", accuracy, correct, len(x_test))
}

func getMaxIndex(values []float64) int {
	maxIndex := 0
	maxVal := values[0]

	for i, val := range values {
		if val > maxVal {
			maxVal = val
			maxIndex = i
		}
	}

	return maxIndex
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
# Run the Iris classification example
go run cmd/main.go
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

## Documentation

- [Layer Documentation](layers/)
- [Loss Functions](loss/)
- [Neural Networks](network/)
- [Data Processing](data/)

## Limitations

- Currently only supports dense feedforward networks
- No GPU acceleration
- Limited optimization algorithms (only SGD)
- Basic data handling capabilities

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
