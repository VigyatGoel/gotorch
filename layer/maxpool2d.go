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

	input      *tensor.Dense
	maxIndices [][]int // Partitioned per batch to avoid race conditions
}

func NewMaxPool2D(poolSize, stride int) *MaxPool2D {
	if poolSize <= 0 {
		panic("poolSize must be greater than 0")
	}
	if stride <= 0 {
		stride = poolSize // Default stride equals pool size
	}
	return &MaxPool2D{
		PoolSize: poolSize,
		Stride:   stride,
	}
}

func (m *MaxPool2D) Forward(input *tensor.Dense) *tensor.Dense {
	m.input = input.Clone().(*tensor.Dense)

	inputShape := input.Shape()
	if len(inputShape) != 4 {
		panic(fmt.Sprintf("Input to MaxPool2D must be 4D tensor, got shape %v", inputShape))
	}

	batchSize := inputShape[0]
	channels := inputShape[1]
	height := inputShape[2]
	width := inputShape[3]

	outputHeight := (height-m.PoolSize)/m.Stride + 1
	outputWidth := (width-m.PoolSize)/m.Stride + 1

	if outputHeight <= 0 || outputWidth <= 0 {
		panic(fmt.Sprintf("Invalid output dimensions: %dx%d", outputHeight, outputWidth))
	}

	// Safe type assertion with error handling
	inputData, ok := input.Data().([]float64)
	if !ok {
		panic("Input tensor data must be []float64")
	}

	// Partition output data and max indices per batch to avoid race conditions
	batchOutputs := make([][]float64, batchSize)
	batchMaxIndices := make([][]int, batchSize)
	outChannelSize := outputHeight * outputWidth
	batchOutChannelSize := channels * outChannelSize

	for b := 0; b < batchSize; b++ {
		batchOutputs[b] = make([]float64, batchOutChannelSize)
		batchMaxIndices[b] = make([]int, batchOutChannelSize)
	}

	var wg sync.WaitGroup
	wg.Add(batchSize)

	for b := 0; b < batchSize; b++ {
		go func(b int) {
			defer wg.Done()

			inputBatchOffset := b * channels * height * width
			inChannelSize := height * width

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
									inputIdx := inputBatchOffset + c*inChannelSize + h*width + w
									if inputData[inputIdx] > maxVal {
										maxVal = inputData[inputIdx]
										maxIdx = inputIdx
									}
								}
							}
						}

						outputIdx := c*outChannelSize + oh*outputWidth + ow
						batchOutputs[b][outputIdx] = maxVal
						batchMaxIndices[b][outputIdx] = maxIdx
					}
				}
			}
		}(b)
	}
	wg.Wait()

	// Combine batch outputs and max indices
	outputData := make([]float64, batchSize*batchOutChannelSize)
	m.maxIndices = make([][]int, batchSize)

	for b := 0; b < batchSize; b++ {
		copy(outputData[b*batchOutChannelSize:(b+1)*batchOutChannelSize], batchOutputs[b])
		m.maxIndices[b] = batchMaxIndices[b]
	}

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

	// Safe type assertion with error handling
	gradOutputData, ok := gradOutput.Data().([]float64)
	if !ok {
		panic("GradOutput tensor data must be []float64")
	}

	// Partition input gradients per batch to avoid race conditions
	batchInputGrads := make([][]float64, batchSize)
	inChannelSize := height * width
	batchInChannelSize := channels * inChannelSize

	for b := 0; b < batchSize; b++ {
		batchInputGrads[b] = make([]float64, batchInChannelSize)
	}

	var wg sync.WaitGroup
	wg.Add(batchSize)

	for b := 0; b < batchSize; b++ {
		go func(b int) {
			defer wg.Done()

			outChannelSize := outputHeight * outputWidth
			batchOutChannelSize := channels * outChannelSize
			gradOutputBatchOffset := b * batchOutChannelSize

			for c := 0; c < channels; c++ {
				for oh := 0; oh < outputHeight; oh++ {
					for ow := 0; ow < outputWidth; ow++ {
						outputIdx := c*outChannelSize + oh*outputWidth + ow
						gradOutputIdx := gradOutputBatchOffset + outputIdx

						maxIdx := m.maxIndices[b][outputIdx]
						if maxIdx >= 0 {
							// Convert global index to batch-local index
							localMaxIdx := maxIdx - b*batchInChannelSize
							if localMaxIdx >= 0 && localMaxIdx < batchInChannelSize {
								batchInputGrads[b][localMaxIdx] += gradOutputData[gradOutputIdx]
							}
						}
					}
				}
			}
		}(b)
	}
	wg.Wait()

	// Combine input gradients from all batches
	inputGradData := make([]float64, batchSize*batchInChannelSize)
	for b := 0; b < batchSize; b++ {
		copy(inputGradData[b*batchInChannelSize:(b+1)*batchInChannelSize], batchInputGrads[b])
	}

	return tensor.New(
		tensor.WithShape(batchSize, channels, height, width),
		tensor.WithBacking(inputGradData),
	)
}

// Layer interface methods
func (m *MaxPool2D) GetWeights() *tensor.Dense                 { return nil }
func (m *MaxPool2D) GetGradients() *tensor.Dense               { return nil }
func (m *MaxPool2D) UpdateWeights(weightsUpdate *tensor.Dense) {}
func (m *MaxPool2D) GetBiases() *tensor.Dense                  { return nil }
func (m *MaxPool2D) GetBiasGradients() *tensor.Dense           { return nil }
func (m *MaxPool2D) UpdateBiases(biasUpdate *tensor.Dense)     {}

func (m *MaxPool2D) String() string {
	return fmt.Sprintf("MaxPool2D(pool_size=%d, stride=%d)", m.PoolSize, m.Stride)
}

// ClearCache clears cached data to prevent memory leaks
func (m *MaxPool2D) ClearCache() {
	m.input = nil
	m.maxIndices = nil
}
