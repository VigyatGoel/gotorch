# GoTorch

[![Go Version](https://img.shields.io/badge/Go-1.25-blue.svg)](https://go.dev/)
[![Go Reference](https://pkg.go.dev/badge/github.com/VigyatGoel/gotorch.svg)](https://pkg.go.dev/github.com/VigyatGoel/gotorch)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

GoTorch is a deep learning framework implemented in pure Go, designed for simplicity and educational purposes. It provides the essential building blocks for creating and training neural networks.

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
  - Convolutional layers (Conv2D)
  - Pooling layers (MaxPool2D)
  - Flatten layer for reshaping
  - Activation functions (ReLU, Leaky ReLU, Sigmoid, Softmax, SiLU/Swish)
  - Loss functions (Cross-Entropy, MSE)
  - Optimizers: SGD(with momentum) and Adam supported
  - Sequential model architecture
- **Model Persistence**:
  - Save trained models to disk in Binary based .gth format
  - Load models to perform inference without retraining
  - Preserves layers, weights, biases, and optimizer configurations
- **Data Processing**:
  - CSV data loading with automatic feature extraction
  - Training/testing data splitting
  - Data Normalization
  - Classification and regression support
- **Modular Design**: Easy to extend with new layers and components

## Installation

```bash
# Option 1: Use go get to install the package
go get github.com/VigyatGoel/gotorch
```

```bash
# Option 2: Clone the repository
git clone https://github.com/VigyatGoel/gotorch.git
```

After cloning, run:
```bash
go mod tidy
```

### CLI Tool Usage

GoTorch includes a CLI tool that simplifies running your models by automatically setting the required environment variables.

To build the CLI tool:
```bash
go build -o gotorch ./cmd/gotorch
```

To run a Go program that uses GoTorch:
```bash
./gotorch run your_program.go
```

You can also pass arguments to your Go program:
```bash
./gotorch run your_program.go arg1 arg2
```

This is equivalent to running:
```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 go run your_program.go arg1 arg2
```

## Quick Start

```bash
# Build the CLI tool
go build -o gotorch ./cmd/gotorch

# Run the example
./gotorch run examples/train_basic.go
```

## Architecture

GoTorch's architecture consists of the following key components:

### Layers

- **Linear**: Fully connected layer with weights and biases
- **Conv2d**: 2D convolutional layer for image processing
- **MaxPool2d**: 2D max pooling layer for downsampling
- **Flatten**: Flattens multi-dimensional input to 1D
- **ReLU**: Rectified Linear Unit activation function
- **LeakyReLU**: Leaky ReLU activation function with customizable negative slope
- **Sigmoid**: Sigmoid activation function
- **Softmax**: Softmax activation for multi-class outputs
- **SiLU/Swish**: Sigmoid Linear Unit (SiLU) activation function

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
./gotorch run examples/train_basic.go
```

## Customization

### Creating Custom Layers

To implement a custom layer, implement the Layer interface:

```go
type Layer interface {
    Forward(input *tensor.Dense) *tensor.Dense
    Backward(dout *tensor.Dense) *tensor.Dense
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
  - Recurrent Neural Networks (RNN) for sequential data
  - LSTM and GRU variants for improved sequence modeling

- **Performance Optimizations**:
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
