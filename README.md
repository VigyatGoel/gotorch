# GoTorch

[![Go Version](https://img.shields.io/badge/Go-1.24-blue.svg)](https://go.dev/)
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
# Clone the repository
git clone https://github.com/VigyatGoel/gotorch.git

# Navigate to project directory
cd gotorch

# Build the project
go run src/cmd/main.go
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
go run src/cmd/main.go
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

- [Layer Documentation](src/layers/)
- [Loss Functions](src/loss/)
- [Neural Networks](src/network/)
- [Data Processing](src/data/)

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
