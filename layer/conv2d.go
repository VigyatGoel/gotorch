package layer

import (
	"fmt"
	"math"
	"math/rand"
	"sync"

	"gorgonia.org/tensor"
)

type Conv2D struct {
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int

	// Weights and biases
	weight *tensor.Dense // Shape: [outChannels, inChannels, kernelSize, kernelSize]
	bias   *tensor.Dense // Shape: [outChannels]

	// Gradients
	dweight *tensor.Dense // Shape: [outChannels, inChannels, kernelSize, kernelSize]
	dbias   *tensor.Dense // Shape: [outChannels]

	// For backward pass
	input *tensor.Dense
}

func NewConv2D(inChannels, outChannels, kernelSize, stride, padding int) *Conv2D {
	// Initialize weights with Xavier/Glorot initialization
	fanIn := inChannels * kernelSize * kernelSize
	fanOut := outChannels * kernelSize * kernelSize
	limit := math.Sqrt(6.0 / float64(fanIn+fanOut))

	// Initialize weight tensor
	weightData := make([]float64, outChannels*inChannels*kernelSize*kernelSize)
	for i := range weightData {
		weightData[i] = (rand.Float64()*2 - 1) * limit
	}
	weight := tensor.New(
		tensor.WithShape(outChannels, inChannels, kernelSize, kernelSize),
		tensor.WithBacking(weightData),
	)

	// Initialize bias tensor
	biasData := make([]float64, outChannels)
	bias := tensor.New(
		tensor.WithShape(outChannels),
		tensor.WithBacking(biasData),
	)

	// Initialize gradient tensors
	dweight := tensor.New(
		tensor.WithShape(outChannels, inChannels, kernelSize, kernelSize),
		tensor.WithBacking(make([]float64, outChannels*inChannels*kernelSize*kernelSize)),
	)

	dbias := tensor.New(
		tensor.WithShape(outChannels),
		tensor.WithBacking(make([]float64, outChannels)),
	)

	return &Conv2D{
		InChannels:  inChannels,
		OutChannels: outChannels,
		KernelSize:  kernelSize,
		Stride:      stride,
		Padding:     padding,
		weight:      weight,
		bias:        bias,
		dweight:     dweight,
		dbias:       dbias,
	}
}

func (c *Conv2D) Forward(input *tensor.Dense) *tensor.Dense {
	// Store input for backward pass
	c.input = input.Clone().(*tensor.Dense)

	// Input shape: [batchSize, inChannels, height, width]
	inputShape := input.Shape()
	if len(inputShape) != 4 {
		panic(fmt.Sprintf("Input to Conv2D must be 4D tensor, got shape %v", inputShape))
	}

	batchSize := inputShape[0]
	inChannels := inputShape[1]
	height := inputShape[2]
	width := inputShape[3]

	// Verify input channels match
	if inChannels != c.InChannels {
		panic(fmt.Sprintf("Expected %d input channels, got %d", c.InChannels, inChannels))
	}

	// Calculate output dimensions
	outputHeight := (height+2*c.Padding-c.KernelSize)/c.Stride + 1
	outputWidth := (width+2*c.Padding-c.KernelSize)/c.Stride + 1

	// Validate output dimensions
	if outputHeight <= 0 || outputWidth <= 0 {
		panic(fmt.Sprintf("Invalid output dimensions: %dx%d. Check kernel size, stride, and padding.", outputHeight, outputWidth))
	}

	// Pad input if needed
	var paddedInput *tensor.Dense
	if c.Padding > 0 {
		paddedInput = c.padInput(input, c.Padding)
	} else {
		paddedInput = input
	}

	// Output shape: [batchSize, outChannels, outputHeight, outputWidth]
	outputData := make([]float64, batchSize*c.OutChannels*outputHeight*outputWidth)

	// Perform convolution with optimized loop ordering
	paddedInputData := paddedInput.Data().([]float64)
	weightData := c.weight.Data().([]float64)
	biasData := c.bias.Data().([]float64)

	paddedShape := paddedInput.Shape()
	paddedHeight := paddedShape[2]
	paddedWidth := paddedShape[3]

	// Pre-calculate frequently used values
	kernelSizeSquared := c.KernelSize * c.KernelSize
	inChannelSize := paddedHeight * paddedWidth
	outChannelSize := outputHeight * outputWidth
	batchInChannelSize := c.InChannels * inChannelSize
	batchOutChannelSize := c.OutChannels * outChannelSize
	weightOutChannelSize := c.InChannels * kernelSizeSquared

	var wg sync.WaitGroup
	wg.Add(batchSize)

	// For each batch
	for b := 0; b < batchSize; b++ {
		go func(b int) {
			defer wg.Done()

			// Pre-calculate batch offset
			batchOffset := b * batchOutChannelSize
			inputBatchOffset := b * batchInChannelSize

			// For each output position (better cache locality)
			for oh := 0; oh < outputHeight; oh++ {
				for ow := 0; ow < outputWidth; ow++ {
					// Calculate starting position in padded input
					startH := oh * c.Stride
					startW := ow * c.Stride

					// Pre-calculate output position offset
					outPosOffset := oh*outputWidth + ow

					// For each output channel
					for oc := 0; oc < c.OutChannels; oc++ {
						// Convolution operation
						sum := 0.0

						// For each input channel
						for ic := 0; ic < c.InChannels; ic++ {
							// Pre-calculate channel offsets
							inputChannelOffset := inputBatchOffset + ic*inChannelSize
							weightChannelOffset := oc*weightOutChannelSize + ic*kernelSizeSquared

							// For each position in kernel
							for kh := 0; kh < c.KernelSize; kh++ {
								for kw := 0; kw < c.KernelSize; kw++ {
									// Input index (NCHW format)
									inputIdx := inputChannelOffset +
										(startH+kh)*paddedWidth +
										(startW + kw)

									// Weight index (OIHW format)
									weightIdx := weightChannelOffset +
										kh*c.KernelSize +
										kw

									sum += paddedInputData[inputIdx] * weightData[weightIdx]
								}
							}
						}

						// Add bias
						sum += biasData[oc]

						// Output index (NCHW format)
						outputIdx := batchOffset +
							oc*outChannelSize +
							outPosOffset

						outputData[outputIdx] = sum
					}
				}
			}
		}(b)
	}
	wg.Wait()

	output := tensor.New(
		tensor.WithShape(batchSize, c.OutChannels, outputHeight, outputWidth),
		tensor.WithBacking(outputData),
	)

	return output
}

func (c *Conv2D) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	// Input shape: [batchSize, inChannels, height, width]
	inputShape := c.input.Shape()
	batchSize := inputShape[0]
	inChannels := inputShape[1]
	inputHeight := inputShape[2]
	inputWidth := inputShape[3]

	// GradOutput shape: [batchSize, outChannels, outputHeight, outputWidth]
	gradOutputShape := gradOutput.Shape()
	outputHeight := gradOutputShape[2]
	outputWidth := gradOutputShape[3]

	// Initialize gradients to zero
	dweightData := c.dweight.Data().([]float64)
	for i := range dweightData {
		dweightData[i] = 0
	}

	dbiasData := c.dbias.Data().([]float64)
	for i := range dbiasData {
		dbiasData[i] = 0
	}

	// Get data arrays
	gradOutputData := gradOutput.Data().([]float64)
	weightData := c.weight.Data().([]float64)

	// Pad input if needed for gradient computation
	var paddedInput *tensor.Dense
	var paddedInputData []float64
	if c.Padding > 0 {
		paddedInput = c.padInput(c.input, c.Padding)
		paddedInputData = paddedInput.Data().([]float64)
	} else {
		paddedInputData = c.input.Data().([]float64)
	}

	paddedHeight := inputHeight + 2*c.Padding
	paddedWidth := inputWidth + 2*c.Padding

	// Pre-calculate frequently used values
	kernelSizeSquared := c.KernelSize * c.KernelSize
	inChannelSize := paddedHeight * paddedWidth
	outChannelSize := outputHeight * outputWidth
	batchInChannelSize := c.InChannels * inChannelSize
	batchOutChannelSize := c.OutChannels * outChannelSize
	weightOutChannelSize := c.InChannels * kernelSizeSquared

	// Use a mutex to protect gradient accumulation
	var gradMutex sync.Mutex

	// Calculate weight and bias gradients
	// Parallelize over batches like in MaxPool2D
	var wg1 sync.WaitGroup
	wg1.Add(batchSize)

	for b := 0; b < batchSize; b++ {
		go func(b int) {
			defer wg1.Done()

			// Local gradients for this batch
			localDWeight := make([]float64, len(dweightData))
			localDBias := make([]float64, len(dbiasData))

			// Pre-calculate batch offsets
			inputBatchOffset := b * batchInChannelSize
			gradOutputBatchOffset := b * batchOutChannelSize

			// Calculate gradients for this batch with optimized loop ordering
			// For each output position (better cache locality)
			for oh := 0; oh < outputHeight; oh++ {
				for ow := 0; ow < outputWidth; ow++ {
					// Pre-calculate position offsets
					outPosOffset := oh*outputWidth + ow
					gradOutputPosOffset := gradOutputBatchOffset + outPosOffset

					// For each output channel
					for oc := 0; oc < c.OutChannels; oc++ {
						// GradOutput index (NCHW format)
						gradOutputIdx := gradOutputPosOffset + oc*outChannelSize
						gradVal := gradOutputData[gradOutputIdx]

						// Calculate bias gradient
						localDBias[oc] += gradVal

						// For each input channel
						for ic := 0; ic < c.InChannels; ic++ {
							// Pre-calculate channel offsets
							inputChannelOffset := inputBatchOffset + ic*inChannelSize
							weightChannelOffset := oc*weightOutChannelSize + ic*kernelSizeSquared

							// For each position in kernel
							for kh := 0; kh < c.KernelSize; kh++ {
								for kw := 0; kw < c.KernelSize; kw++ {
									// Input position
									inputH := oh*c.Stride + kh
									inputW := ow*c.Stride + kw

									// Input index (NCHW format)
									inputIdx := inputChannelOffset +
										inputH*paddedWidth +
										inputW

									// Weight gradient index (OIHW format)
									dweightIdx := weightChannelOffset +
										kh*c.KernelSize +
										kw

									localDWeight[dweightIdx] += paddedInputData[inputIdx] * gradVal
								}
							}
						}
					}
				}
			}

			// Safely accumulate gradients using mutex
			gradMutex.Lock()
			for i := range dweightData {
				dweightData[i] += localDWeight[i]
			}
			for i := range dbiasData {
				dbiasData[i] += localDBias[i]
			}
			gradMutex.Unlock()
		}(b)
	}
	wg1.Wait()

	// Calculate input gradients
	// Initialize padded input gradient buffer
	paddedInputGradData := make([]float64, batchSize*c.InChannels*paddedHeight*paddedWidth)

	// Parallelize over batches like in MaxPool2D
	var wg2 sync.WaitGroup
	wg2.Add(batchSize)

	// For each batch
	for b := 0; b < batchSize; b++ {
		go func(b int) {
			defer wg2.Done()

			// Pre-calculate batch offsets
			gradOutputBatchOffset := b * batchOutChannelSize
			paddedInputGradBatchOffset := b * batchInChannelSize

			// For each output position (better cache locality)
			for oh := 0; oh < outputHeight; oh++ {
				for ow := 0; ow < outputWidth; ow++ {
					// Pre-calculate position offset
					outPosOffset := oh*outputWidth + ow
					gradOutputPosOffset := gradOutputBatchOffset + outPosOffset

					// For each output channel
					for oc := 0; oc < c.OutChannels; oc++ {
						// GradOutput index (NCHW format)
						gradOutputIdx := gradOutputPosOffset + oc*outChannelSize
						gradVal := gradOutputData[gradOutputIdx]

						// Pre-calculate weight channel offset
						weightChannelOffset := oc * weightOutChannelSize

						// For each input channel
						for ic := 0; ic < c.InChannels; ic++ {
							// Pre-calculate weight sub-channel offset
							weightSubChannelOffset := weightChannelOffset + ic*kernelSizeSquared
							paddedInputGradChannelOffset := paddedInputGradBatchOffset + ic*inChannelSize

							// For each position in kernel
							for kh := 0; kh < c.KernelSize; kh++ {
								for kw := 0; kw < c.KernelSize; kw++ {
									// Weight index (OIHW format)
									weightIdx := weightSubChannelOffset +
										kh*c.KernelSize +
										kw

									// Position in padded input
									inputH := oh*c.Stride + kh
									inputW := ow*c.Stride + kw

									// Padded input gradient index (NCHW format)
									paddedInputGradIdx := paddedInputGradChannelOffset +
										inputH*paddedWidth +
										inputW

									paddedInputGradData[paddedInputGradIdx] += weightData[weightIdx] * gradVal
								}
							}
						}
					}
				}
			}
		}(b)
	}
	wg2.Wait()

	// Remove padding from input gradients if needed
	var inputGradData []float64
	if c.Padding > 0 {
		inputGradData = make([]float64, batchSize*inChannels*inputHeight*inputWidth)
		for b := 0; b < batchSize; b++ {
			for ic := 0; ic < inChannels; ic++ {
				for h := 0; h < inputHeight; h++ {
					for w := 0; w < inputWidth; w++ {
						// Padded index (NCHW format)
						paddedIdx := b*(inChannels*paddedHeight*paddedWidth) +
							ic*(paddedHeight*paddedWidth) +
							(h+c.Padding)*paddedWidth +
							(w + c.Padding)

						// Original index (NCHW format)
						originalIdx := b*(inChannels*inputHeight*inputWidth) +
							ic*(inputHeight*inputWidth) +
							h*inputWidth +
							w

						inputGradData[originalIdx] = paddedInputGradData[paddedIdx]
					}
				}
			}
		}
	} else {
		inputGradData = paddedInputGradData
	}

	inputGrad := tensor.New(
		tensor.WithShape(batchSize, inChannels, inputHeight, inputWidth),
		tensor.WithBacking(inputGradData),
	)

	return inputGrad
}

func (c *Conv2D) padInput(input *tensor.Dense, padding int) *tensor.Dense {
	inputShape := input.Shape()
	batchSize := inputShape[0]
	channels := inputShape[1]
	height := inputShape[2]
	width := inputShape[3]

	newHeight := height + 2*padding
	newWidth := width + 2*padding

	// Initialize with zeros
	paddedData := make([]float64, batchSize*channels*newHeight*newWidth)
	inputData := input.Data().([]float64)

	// Pre-calculate frequently used values
	inChannelSize := height * width
	paddedChannelSize := newHeight * newWidth
	paddingOffset := padding*newWidth + padding

	// Use parallelization for better performance with large batches
	var wg sync.WaitGroup
	wg.Add(batchSize)

	for b := 0; b < batchSize; b++ {
		go func(b int) {
			defer wg.Done()

			// Pre-calculate batch offsets
			batchOffset := b * channels * inChannelSize
			paddedBatchOffset := b*channels*paddedChannelSize + paddingOffset

			for ch := 0; ch < channels; ch++ {
				// Pre-calculate channel offsets
				channelOffset := batchOffset + ch*inChannelSize
				paddedChannelOffset := paddedBatchOffset + ch*paddedChannelSize

				// Copy data row by row for better cache locality
				for h := 0; h < height; h++ {
					// Source index
					srcStart := channelOffset + h*width
					// Destination index
					dstStart := paddedChannelOffset + h*newWidth

					// Copy entire row at once
					copy(paddedData[dstStart:dstStart+width], inputData[srcStart:srcStart+width])
				}
			}
		}(b)
	}
	wg.Wait()

	return tensor.New(
		tensor.WithShape(batchSize, channels, newHeight, newWidth),
		tensor.WithBacking(paddedData),
	)
}

// Implement Layer interface methods
func (c *Conv2D) GetWeights() *tensor.Dense {
	return c.weight
}

func (c *Conv2D) GetGradients() *tensor.Dense {
	return c.dweight
}

func (c *Conv2D) UpdateWeights(weightsUpdate *tensor.Dense) {
	// In-place update
	weightData := c.weight.Data().([]float64)
	updateData := weightsUpdate.Data().([]float64)
	for i := range weightData {
		weightData[i] = updateData[i]
	}
}

func (c *Conv2D) GetBiases() *tensor.Dense {
	return c.bias
}

func (c *Conv2D) GetBiasGradients() *tensor.Dense {
	return c.dbias
}

func (c *Conv2D) UpdateBiases(biasUpdate *tensor.Dense) {
	// In-place update
	biasData := c.bias.Data().([]float64)
	updateData := biasUpdate.Data().([]float64)
	for i := range biasData {
		biasData[i] = updateData[i]
	}
}

// Helper methods
func (c *Conv2D) String() string {
	return fmt.Sprintf("Conv2D(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, padding=%d)",
		c.InChannels, c.OutChannels, c.KernelSize, c.Stride, c.Padding)
}

// ResetGradients resets the gradients to zero
func (c *Conv2D) ResetGradients() {
	dweightData := c.dweight.Data().([]float64)
	for i := range dweightData {
		dweightData[i] = 0
	}

	dbiasData := c.dbias.Data().([]float64)
	for i := range dbiasData {
		dbiasData[i] = 0
	}
}
