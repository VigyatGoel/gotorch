# GoTorch

[![Go Version](https://img.shields.io/badge/Go-1.26-blue.svg)](https://go.dev/)
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
  - Familiar training loop patterns
  - Easy migration from PyTorch concepts
- **Advanced Data Processing**:
  - **CSV Support**: Automatic feature extraction and preprocessing
  - **Batch Processing**: Configurable batch sizes with per-epoch shuffling
  - Training/testing data splitting with configurable ratios
- **Model Persistence**:
  - Save/load models in `.gth` format
  - Preserves complete model state including optimizer settings

## Installation

```bash
go get github.com/VigyatGoel/gotorch
```

## Quick Start

```bash
git clone https://github.com/VigyatGoel/gotorch.git
cd gotorch
go run examples/train_basic.go
```

## Architecture

### Layers

- **Linear**: Fully connected layer with weights and biases
- **Flatten**: Flattens multi-dimensional input to 2D
- **Dropout**: Regularization layer that randomly zeroes input units
- **ReLU**: Rectified Linear Unit activation
- **LeakyReLU**: Leaky ReLU with customizable negative slope
- **Sigmoid**: Sigmoid activation
- **Softmax**: Softmax activation for multi-class outputs
- **SiLU/Swish**: Sigmoid Linear Unit activation

### Network

- **Sequential**: Container for stacking layers in sequence

### Loss Functions

- **CrossEntropyLoss**: For classification tasks
- **MSELoss**: Mean Squared Error for regression tasks

### Data Handling

- **DataLoader**: Unified interface for CSV data with batching and normalization

## Examples

### Tabular Data (CSV)

```bash
go run examples/train_basic.go
```

### Training Loop

```go
for epoch := 0; epoch < epochs; epoch++ {
    runningLoss := 0.0
    correct, total := 0, 0

    for _, batch := range dataLoader.GetBatches(x_train, y_train, epoch) {
        preds := model.Forward(batch.Features)
        lossVal := criterion.Forward(preds, batch.Targets)

        model.GetOptimizer().ZeroGrad()
        grad := criterion.Backward()
        model.Backward(grad)

        runningLoss += lossVal
        rows, _ := batch.Features.Dims()
        for i := 0; i < rows; i++ {
            if utils.GetMaxIndexRow(preds, i) == utils.GetMaxIndexRow(batch.Targets, i) {
                correct++
            }
            total++
        }
    }

    fmt.Printf("Epoch [%d/%d] Loss: %.4f Train Acc: %.2f%%\n",
        epoch+1, epochs, runningLoss/float64(len(batches)), 100.0*float64(correct)/float64(total))
}
```

## Customization

### Creating Custom Layers

Implement the `Layer` interface:

```go
type Layer interface {
    Forward(input *mat.Dense) *mat.Dense
    Backward(dout *mat.Dense) *mat.Dense

    GetWeights() *mat.Dense
    GetGradients() *mat.Dense
    UpdateWeights(weightsUpdate *mat.Dense)

    GetBiases() *mat.Dense
    GetBiasGradients() *mat.Dense
    UpdateBiases(biasUpdate *mat.Dense)

    ClearCache()
}
```

## Model Persistence

### Saving Models

```go
err := model.Save("my_model.gth")
if err != nil {
    log.Fatalf("Error saving model: %v", err)
}
```

### Loading Models

```go
loadedModel, err := network.Load("my_model.gth")
if err != nil {
    log.Fatalf("Error loading model: %v", err)
}

predictions := loadedModel.Predict(newData)
```

### Model Format

Models are saved in `.gth` format (gob-encoded) and include:
- Layer types and configurations
- Weights and biases for trainable layers
- Optimizer configuration (type, learning rate, and other parameters)

## Documentation

- [Layer Documentation](layer/)
- [Loss Functions](loss/)
- [Neural Networks](network/)
- [Data Processing](data/)

## Future

GoTorch is under active development, with plans to incorporate:

- **Advanced Architectures**: RNN, LSTM, GRU
- **Performance**: GPU acceleration via CUDA/Metal
- **Optimizers**: AdamW, RMSProp, learning rate schedulers
- **Regularization**: L1/L2, early stopping, model checkpointing
- **Data**: JSON/parquet support, text tokenization, time series

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Inspired by PyTorch's design philosophy
- Built on [gonum](https://gonum.org/) for BLAS-accelerated matrix operations
