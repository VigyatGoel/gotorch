package layer

import (
	"fmt"
	"math"
	"sync"

	"gorgonia.org/tensor"
)

type MaxPool2D struct {
	PoolSize int
	Stride   int

	// For backward pass
	input      *tensor.Dense
	maxIndices []int // Store indices of max values for backward pass
}

func NewMaxPool2D(poolSize, stride int) *MaxPool2D {
	if stride == 0 {
		stride = poolSize // Default stride equals pool size
	}
	return &MaxPool2D{
		PoolSize: poolSize,
		Stride:   stride,
	}
}

func (m *MaxPool2D) Forward(input *tensor.Dense) *tensor.Dense {
	// Store input for backward pass
	m.input = input.Clone().(*tensor.Dense)

	// Input shape: [batchSize, channels, height, width]
	inputShape := input.Shape()
	if len(inputShape) != 4 {
		panic(fmt.Sprintf("Input to MaxPool2D must be 4D tensor, got shape %v", inputShape))
	}

	batchSize := inputShape[0]
	channels := inputShape[1]
	height := inputShape[2]
	width := inputShape[3]

	// Calculate output dimensions
	outputHeight := (height-m.PoolSize)/m.Stride + 1
	outputWidth := (width-m.PoolSize)/m.Stride + 1

	// Initialize output and max indices
	outputData := make([]float64, batchSize*channels*outputHeight*outputWidth)
	m.maxIndices = make([]int, batchSize*channels*outputHeight*outputWidth)

	inputData := input.Data().([]float64)

	var wg sync.WaitGroup
	wg.Add(batchSize)

	// Perform max pooling
	for b := 0; b < batchSize; b++ {
		go func(b int) {
			defer wg.Done()
			for c := 0; c < channels; c++ {
				for oh := 0; oh < outputHeight; oh++ {
					for ow := 0; ow < outputWidth; ow++ {
						startH := oh * m.Stride
						startW := ow * m.Stride

						maxVal := math.Inf(-1)
						maxIdx := -1

						// Find max in pool window
						for ph := 0; ph < m.PoolSize; ph++ {
							for pw := 0; pw < m.PoolSize; pw++ {
								h := startH + ph
								w := startW + pw

								if h < height && w < width {
									inputIdx := b*(channels*height*width) +
										c*(height*width) +
										h*width +
										w

									if inputData[inputIdx] > maxVal {
										maxVal = inputData[inputIdx]
										maxIdx = inputIdx
									}
								}
							}
						}

						outputIdx := b*(channels*outputHeight*outputWidth) +
							c*(outputHeight*outputWidth) +
							oh*outputWidth +
							ow

						outputData[outputIdx] = maxVal
						m.maxIndices[outputIdx] = maxIdx
					}
				}
			}
		}(b)
	}
	wg.Wait()

	return tensor.New(
		tensor.WithShape(batchSize, channels, outputHeight, outputWidth),
		tensor.WithBacking(outputData),
	)
}

func (m *MaxPool2D) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	inputShape := m.input.Shape()
	batchSize := inputShape[0]
	channels := inputShape[1]
	height := inputShape[2]
	width := inputShape[3]

	outputShape := gradOutput.Shape()
	outputHeight := outputShape[2]
	outputWidth := outputShape[3]

	// Initialize gradient for input
	inputGradData := make([]float64, batchSize*channels*height*width)
	gradOutputData := gradOutput.Data().([]float64)

	var wg sync.WaitGroup
	wg.Add(batchSize)

	// Distribute gradients to max positions, parallelized over the batch
	for b := 0; b < batchSize; b++ {
		go func(b int) {
			defer wg.Done()
			for c := 0; c < channels; c++ {
				for oh := 0; oh < outputHeight; oh++ {
					for ow := 0; ow < outputWidth; ow++ {
						outputIdx := b*(channels*outputHeight*outputWidth) +
							c*(outputHeight*outputWidth) +
							oh*outputWidth +
							ow

						maxIdx := m.maxIndices[outputIdx]
						if maxIdx >= 0 {
							inputGradData[maxIdx] += gradOutputData[outputIdx]
						}
					}
				}
			}
		}(b)
	}
	wg.Wait()

	return tensor.New(
		tensor.WithShape(batchSize, channels, height, width),
		tensor.WithBacking(inputGradData),
	)
}

// Implement Layer interface
func (m *MaxPool2D) GetWeights() *tensor.Dense                 { return nil }
func (m *MaxPool2D) GetGradients() *tensor.Dense               { return nil }
func (m *MaxPool2D) UpdateWeights(weightsUpdate *tensor.Dense) {}
func (m *MaxPool2D) GetBiases() *tensor.Dense                  { return nil }
func (m *MaxPool2D) GetBiasGradients() *tensor.Dense           { return nil }
func (m *MaxPool2D) UpdateBiases(biasUpdate *tensor.Dense)     {}
