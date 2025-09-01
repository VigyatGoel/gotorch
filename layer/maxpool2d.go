package layer

import (
	"fmt"
	"math"
	"sync"

	"gorgonia.org/tensor"
)

// MaxPool2D implements 2D max pooling for downsampling
type MaxPool2D struct {
	PoolSize int // size of pooling window
	Stride   int // stride for pooling operation

	input      *tensor.Dense // cached input for backward pass
	maxIndices [][]int       // indices of max values for gradient routing
}

// NewMaxPool2D creates a new 2D max pooling layer
func NewMaxPool2D(poolSize, stride int) *MaxPool2D {
	if poolSize <= 0 {
		panic("poolSize must be greater than 0")
	}
	if stride <= 0 {
		stride = poolSize
	}
	return &MaxPool2D{
		PoolSize: poolSize,
		Stride:   stride,
	}
}

// Forward performs 2D max pooling and records max indices
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

	inputData, ok := input.Data().([]float64)
	if !ok {
		panic("Input tensor data must be []float64")
	}

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

// Backward routes gradients back to max positions using cached indices
func (m *MaxPool2D) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	inputShape := m.input.Shape()
	batchSize := inputShape[0]
	channels := inputShape[1]
	height := inputShape[2]
	width := inputShape[3]

	outputShape := gradOutput.Shape()
	outputHeight := outputShape[2]
	outputWidth := outputShape[3]

	gradOutputData, ok := gradOutput.Data().([]float64)
	if !ok {
		panic("GradOutput tensor data must be []float64")
	}

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

	inputGradData := make([]float64, batchSize*batchInChannelSize)
	for b := 0; b < batchSize; b++ {
		copy(inputGradData[b*batchInChannelSize:(b+1)*batchInChannelSize], batchInputGrads[b])
	}

	return tensor.New(
		tensor.WithShape(batchSize, channels, height, width),
		tensor.WithBacking(inputGradData),
	)
}

func (m *MaxPool2D) GetWeights() *tensor.Dense                 { return nil }
func (m *MaxPool2D) GetGradients() *tensor.Dense               { return nil }
func (m *MaxPool2D) UpdateWeights(weightsUpdate *tensor.Dense) {}
func (m *MaxPool2D) GetBiases() *tensor.Dense                  { return nil }
func (m *MaxPool2D) GetBiasGradients() *tensor.Dense           { return nil }
func (m *MaxPool2D) UpdateBiases(biasUpdate *tensor.Dense)     {}

func (m *MaxPool2D) String() string {
	return fmt.Sprintf("MaxPool2D(pool_size=%d, stride=%d)", m.PoolSize, m.Stride)
}

// ClearCache releases cached data to prevent memory leaks
func (m *MaxPool2D) ClearCache() {
	m.input = nil
	m.maxIndices = nil
}
