package layer

import (
	"fmt"
	"math"
	"math/rand"
	"sync"

	"gorgonia.org/tensor"
)

// Conv2D implements 2D convolution with learnable filters
type Conv2D struct {
	InChannels  int // number of input channels
	OutChannels int // number of output channels (filters)
	KernelSize  int // size of convolution kernel (square)
	Stride      int // stride for convolution
	Padding     int // zero-padding size

	weight  *tensor.Dense // learnable convolution filters
	bias    *tensor.Dense // learnable bias per output channel
	dweight *tensor.Dense // weight gradients
	dbias   *tensor.Dense // bias gradients
	input   *tensor.Dense // cached input for backward pass
}

// NewConv2D creates a new 2D convolution layer with Xavier initialization
func NewConv2D(inChannels, outChannels, kernelSize, stride, padding int) *Conv2D {
	fanIn := inChannels * kernelSize * kernelSize
	fanOut := outChannels * kernelSize * kernelSize
	limit := math.Sqrt(6.0 / float64(fanIn+fanOut))

	weightData := make([]float64, outChannels*inChannels*kernelSize*kernelSize)
	for i := range weightData {
		weightData[i] = (rand.Float64()*2 - 1) * limit
	}
	weight := tensor.New(
		tensor.WithShape(outChannels, inChannels, kernelSize, kernelSize),
		tensor.WithBacking(weightData),
	)

	biasData := make([]float64, outChannels)
	bias := tensor.New(tensor.WithShape(outChannels), tensor.WithBacking(biasData))

	dweight := tensor.New(
		tensor.WithShape(outChannels, inChannels, kernelSize, kernelSize),
		tensor.WithBacking(make([]float64, outChannels*inChannels*kernelSize*kernelSize)),
	)

	dbias := tensor.New(tensor.WithShape(outChannels), tensor.WithBacking(make([]float64, outChannels)))

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

// Forward performs 2D convolution with optional padding
func (c *Conv2D) Forward(input *tensor.Dense) *tensor.Dense {
	c.input = input.Clone().(*tensor.Dense)

	inputShape := input.Shape()
	if len(inputShape) != 4 {
		panic(fmt.Sprintf("Input to Conv2D must be 4D tensor, got shape %v", inputShape))
	}

	batchSize := inputShape[0]
	inChannels := inputShape[1]
	height := inputShape[2]
	width := inputShape[3]

	if inChannels != c.InChannels {
		panic(fmt.Sprintf("Expected %d input channels, got %d", c.InChannels, inChannels))
	}

	outputHeight := (height+2*c.Padding-c.KernelSize)/c.Stride + 1
	outputWidth := (width+2*c.Padding-c.KernelSize)/c.Stride + 1

	if outputHeight <= 0 || outputWidth <= 0 {
		panic(fmt.Sprintf("Invalid output dimensions: %dx%d", outputHeight, outputWidth))
	}

	var paddedInput *tensor.Dense
	if c.Padding > 0 {
		paddedInput = c.padInput(input, c.Padding)
	} else {
		paddedInput = input
	}

	paddedInputData := paddedInput.Data().([]float64)
	weightData := c.weight.Data().([]float64)
	biasData := c.bias.Data().([]float64)

	paddedShape := paddedInput.Shape()
	paddedHeight := paddedShape[2]
	paddedWidth := paddedShape[3]

	batchOutputs := make([][]float64, batchSize)
	outChannelSize := outputHeight * outputWidth
	batchOutChannelSize := c.OutChannels * outChannelSize

	for b := 0; b < batchSize; b++ {
		batchOutputs[b] = make([]float64, batchOutChannelSize)
	}

	var wg sync.WaitGroup
	wg.Add(batchSize)

	for b := 0; b < batchSize; b++ {
		go func(b int) {
			defer wg.Done()

			inputBatchOffset := b * c.InChannels * paddedHeight * paddedWidth
			kernelSizeSquared := c.KernelSize * c.KernelSize
			inChannelSize := paddedHeight * paddedWidth
			weightOutChannelSize := c.InChannels * kernelSizeSquared

			for oh := 0; oh < outputHeight; oh++ {
				for ow := 0; ow < outputWidth; ow++ {
					startH := oh * c.Stride
					startW := ow * c.Stride
					outPosOffset := oh*outputWidth + ow

					for oc := 0; oc < c.OutChannels; oc++ {
						sum := 0.0

						for ic := 0; ic < c.InChannels; ic++ {
							inputChannelOffset := inputBatchOffset + ic*inChannelSize
							weightChannelOffset := oc*weightOutChannelSize + ic*kernelSizeSquared

							for kh := 0; kh < c.KernelSize; kh++ {
								for kw := 0; kw < c.KernelSize; kw++ {
									inputIdx := inputChannelOffset + (startH+kh)*paddedWidth + (startW + kw)
									weightIdx := weightChannelOffset + kh*c.KernelSize + kw
									sum += paddedInputData[inputIdx] * weightData[weightIdx]
								}
							}
						}

						sum += biasData[oc]
						outputIdx := oc*outChannelSize + outPosOffset
						batchOutputs[b][outputIdx] = sum
					}
				}
			}
		}(b)
	}
	wg.Wait()

	// Combine batch outputs
	outputData := make([]float64, batchSize*batchOutChannelSize)
	for b := 0; b < batchSize; b++ {
		copy(outputData[b*batchOutChannelSize:(b+1)*batchOutChannelSize], batchOutputs[b])
	}

	return tensor.New(
		tensor.WithShape(batchSize, c.OutChannels, outputHeight, outputWidth),
		tensor.WithBacking(outputData),
	)
}

// Backward computes gradients for filters, biases, and input using convolution
func (c *Conv2D) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	inputShape := c.input.Shape()
	batchSize := inputShape[0]
	inChannels := inputShape[1]
	inputHeight := inputShape[2]
	inputWidth := inputShape[3]

	gradOutputShape := gradOutput.Shape()
	outputHeight := gradOutputShape[2]
	outputWidth := gradOutputShape[3]

	dweightData := c.dweight.Data().([]float64)
	for i := range dweightData {
		dweightData[i] = 0
	}
	dbiasData := c.dbias.Data().([]float64)
	for i := range dbiasData {
		dbiasData[i] = 0
	}

	gradOutputData := gradOutput.Data().([]float64)
	weightData := c.weight.Data().([]float64)

	var paddedInputData []float64
	if c.Padding > 0 {
		paddedInput := c.padInput(c.input, c.Padding)
		paddedInputData = paddedInput.Data().([]float64)
	} else {
		paddedInputData = c.input.Data().([]float64)
	}

	paddedHeight := inputHeight + 2*c.Padding
	paddedWidth := inputWidth + 2*c.Padding

	batchDWeights := make([][]float64, batchSize)
	batchDBiases := make([][]float64, batchSize)
	batchInputGrads := make([][]float64, batchSize)

	for b := 0; b < batchSize; b++ {
		batchDWeights[b] = make([]float64, len(dweightData))
		batchDBiases[b] = make([]float64, len(dbiasData))
		batchInputGrads[b] = make([]float64, inChannels*paddedHeight*paddedWidth)
	}

	var wg sync.WaitGroup
	wg.Add(batchSize)

	for b := 0; b < batchSize; b++ {
		go func(b int) {
			defer wg.Done()

			kernelSizeSquared := c.KernelSize * c.KernelSize
			inChannelSize := paddedHeight * paddedWidth
			outChannelSize := outputHeight * outputWidth
			batchInChannelSize := c.InChannels * inChannelSize
			batchOutChannelSize := c.OutChannels * outChannelSize
			weightOutChannelSize := c.InChannels * kernelSizeSquared

			inputBatchOffset := b * batchInChannelSize
			gradOutputBatchOffset := b * batchOutChannelSize

			for oh := 0; oh < outputHeight; oh++ {
				for ow := 0; ow < outputWidth; ow++ {
					outPosOffset := oh*outputWidth + ow

					for oc := 0; oc < c.OutChannels; oc++ {
						gradOutputIdx := gradOutputBatchOffset + oc*outChannelSize + outPosOffset
						gradVal := gradOutputData[gradOutputIdx]

						batchDBiases[b][oc] += gradVal

						for ic := 0; ic < c.InChannels; ic++ {
							inputChannelOffset := inputBatchOffset + ic*inChannelSize
							weightChannelOffset := oc*weightOutChannelSize + ic*kernelSizeSquared

							for kh := 0; kh < c.KernelSize; kh++ {
								for kw := 0; kw < c.KernelSize; kw++ {
									inputH := oh*c.Stride + kh
									inputW := ow*c.Stride + kw
									inputIdx := inputChannelOffset + inputH*paddedWidth + inputW
									dweightIdx := weightChannelOffset + kh*c.KernelSize + kw

									batchDWeights[b][dweightIdx] += paddedInputData[inputIdx] * gradVal
								}
							}
						}
					}
				}
			}

			for oh := 0; oh < outputHeight; oh++ {
				for ow := 0; ow < outputWidth; ow++ {
					outPosOffset := oh*outputWidth + ow

					for oc := 0; oc < c.OutChannels; oc++ {
						gradOutputIdx := gradOutputBatchOffset + oc*outChannelSize + outPosOffset
						gradVal := gradOutputData[gradOutputIdx]
						weightChannelOffset := oc * weightOutChannelSize

						for ic := 0; ic < c.InChannels; ic++ {
							weightSubChannelOffset := weightChannelOffset + ic*kernelSizeSquared
							inputGradChannelOffset := ic * inChannelSize

							for kh := 0; kh < c.KernelSize; kh++ {
								for kw := 0; kw < c.KernelSize; kw++ {
									weightIdx := weightSubChannelOffset + kh*c.KernelSize + kw
									inputH := oh*c.Stride + kh
									inputW := ow*c.Stride + kw
									inputGradIdx := inputGradChannelOffset + inputH*paddedWidth + inputW

									batchInputGrads[b][inputGradIdx] += weightData[weightIdx] * gradVal
								}
							}
						}
					}
				}
			}
		}(b)
	}
	wg.Wait()

	for b := 0; b < batchSize; b++ {
		for i := range dweightData {
			dweightData[i] += batchDWeights[b][i]
		}
		for i := range dbiasData {
			dbiasData[i] += batchDBiases[b][i]
		}
	}

	inputGradData := make([]float64, batchSize*inChannels*inputHeight*inputWidth)
	if c.Padding > 0 {
		for b := 0; b < batchSize; b++ {
			for ic := 0; ic < inChannels; ic++ {
				for h := 0; h < inputHeight; h++ {
					for w := 0; w < inputWidth; w++ {
						paddedIdx := ic*(paddedHeight*paddedWidth) + (h+c.Padding)*paddedWidth + (w + c.Padding)
						originalIdx := b*(inChannels*inputHeight*inputWidth) + ic*(inputHeight*inputWidth) + h*inputWidth + w
						inputGradData[originalIdx] = batchInputGrads[b][paddedIdx]
					}
				}
			}
		}
	} else {
		for b := 0; b < batchSize; b++ {
			copy(inputGradData[b*inChannels*inputHeight*inputWidth:], batchInputGrads[b])
		}
	}

	return tensor.New(
		tensor.WithShape(batchSize, inChannels, inputHeight, inputWidth),
		tensor.WithBacking(inputGradData),
	)
}

// padInput adds zero-padding around input tensor
func (c *Conv2D) padInput(input *tensor.Dense, padding int) *tensor.Dense {
	inputShape := input.Shape()
	batchSize := inputShape[0]
	channels := inputShape[1]
	height := inputShape[2]
	width := inputShape[3]

	newHeight := height + 2*padding
	newWidth := width + 2*padding

	paddedData := make([]float64, batchSize*channels*newHeight*newWidth)
	inputData := input.Data().([]float64)

	inChannelSize := height * width
	paddedChannelSize := newHeight * newWidth

	var wg sync.WaitGroup
	wg.Add(batchSize)

	for b := 0; b < batchSize; b++ {
		go func(b int) {
			defer wg.Done()

			batchOffset := b * channels * inChannelSize
			paddedBatchOffset := b * channels * paddedChannelSize

			for ch := 0; ch < channels; ch++ {
				channelOffset := batchOffset + ch*inChannelSize
				paddedChannelOffset := paddedBatchOffset + ch*paddedChannelSize + padding*newWidth + padding

				for h := 0; h < height; h++ {
					srcStart := channelOffset + h*width
					dstStart := paddedChannelOffset + h*newWidth
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

func (c *Conv2D) GetWeights() *tensor.Dense   { return c.weight }
func (c *Conv2D) GetGradients() *tensor.Dense { return c.dweight }
func (c *Conv2D) UpdateWeights(weightsUpdate *tensor.Dense) {
	c.weight = weightsUpdate.Clone().(*tensor.Dense)
}
func (c *Conv2D) GetBiases() *tensor.Dense              { return c.bias }
func (c *Conv2D) GetBiasGradients() *tensor.Dense       { return c.dbias }
func (c *Conv2D) UpdateBiases(biasUpdate *tensor.Dense) { c.bias = biasUpdate.Clone().(*tensor.Dense) }

func (c *Conv2D) String() string {
	return fmt.Sprintf("Conv2D(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, padding=%d)",
		c.InChannels, c.OutChannels, c.KernelSize, c.Stride, c.Padding)
}

// ResetGradients zeros out accumulated gradients
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

// ClearCache releases cached input to prevent memory leaks
func (c *Conv2D) ClearCache() {
	c.input = nil
}
