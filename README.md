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
  - Flatten layer for reshaping
  - Dropout layer for regularization
  - Activation functions (ReLU, Leaky ReLU, Sigmoid, Softmax, SiLU/Swish)
  - Loss functions (Cross-Entropy, MSE)
  - Optimizers: SGD (with momentum) and Adam
  - Sequential model architecture
- **PyTorch-Style API**:
  - Familiar training loops: `for batch := range dataLoader.TrainBatches(epoch)`
  - PyTorch-like model definition and training patterns
  - Easy migration from PyTorch concepts
- **Advanced Data Processing**:
  - **CSV Support**: Automatic feature extraction and preprocessing
  - **Batch Processing**: Configurable batch sizes with shuffling
  - Training/testing data splitting with configurable ratios
- **Model Persistence**:
  - Save/load models in `.gth` format (JSON-based)
  - Preserves complete model state including optimizer settings
- **CLI Tool**: User-friendly command-line interface with automatic environment setup

## Installation

### Install Library
```bash
go get github.com/VigyatGoel/gotorch
```

### Install CLI Tool (Recommended)
```bash
go install github.com/VigyatGoel/gotorch/cmd/gotorch@latest
```

### Alternative: Clone Repository
```bash
git clone https://github.com/VigyatGoel/gotorch.git
cd gotorch
go mod tidy
go install ./cmd/gotorch
```

### CLI Tool Usage

GoTorch includes a user-friendly CLI tool that automatically handles environment setup:

```bash
# Run any GoTorch program
gotorch run examples/train_basic.go
gotorch run your_program.go

# Pass arguments to your program
gotorch run train.go --epochs 100 --lr 0.001

# Get help
gotorch help
gotorch version
```

The CLI automatically sets `ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25` which is required for GoTorch.

## Quick Start

```bash
# Install GoTorch and CLI
go install github.com/VigyatGoel/gotorch/cmd/gotorch@latest

# Run minimal training example
gotorch run examples/train_basic.go

# Or train on tabular data
gotorch run examples/train_basic.go
```

## Architecture

GoTorch's architecture consists of the following key components:

### Layers

- **Linear**: Fully connected layer with weights and biases
- **Flatten**: Flattens multi-dimensional input to 1D
- **Dropout**: Regularization layer that randomly sets input units to 0
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

- **DataLoader**: Unified interface for CSV data
- **Batch Iteration**: PyTorch-style `for batch := range` loops

## Examples

GoTorch includes multiple example implementations:

### Tabular Data (CSV)
```bash
# Train on Iris dataset (CSV)
gotorch run examples/train_basic.go
```

### PyTorch-Style Training Loop
```go
for epoch := 0; epoch < epochs; epoch++ {
    runningLoss := 0.0
    correct, total := 0, 0
    
    for batch := range dataLoader.TrainBatches(epoch) {
        // Forward pass
        preds := model.Forward(batch.Features)
        loss := criterion.Forward(preds, batch.Targets)
        
        // Backward pass
        model.GetOptimizer().ZeroGrad()
        grad := criterion.Backward()
        model.Backward(grad)
        
        runningLoss += loss
        // Calculate accuracy...
    }
    
    fmt.Printf("Epoch [%d/%d] Loss: %.4f Train Acc: %.2f%%\n", 
               epoch+1, epochs, runningLoss/batchCount, trainAcc)
}
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
  - Regularization techniques (L1/L2 regularization)
  - Learning rate schedulers
  - Early stopping and model checkpointing

- **Expanded Data Handling**:
  - Support for more data formats (JSON, parquet)
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
