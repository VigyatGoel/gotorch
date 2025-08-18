package layer

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Conv2D struct {
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int
	Weights     [][][][]float64 // [out][in][k][k]
	Biases      []float64
	input       [][][][]float64 // [batch][in][h][w]
	output      *mat.Dense
	dWeights    [][][][]float64
	dBiases     []float64
	inH         int // store input height
	inW         int // store input width
}

func NewConv2D(inChannels, outChannels, kernelSize, stride, padding int) *Conv2D {
	conv := &Conv2D{
		InChannels:  inChannels,
		OutChannels: outChannels,
		KernelSize:  kernelSize,
		Stride:      stride,
		Padding:     padding,
	}
	conv.initWeights()
	return conv
}

func (c *Conv2D) Forward(x *mat.Dense) *mat.Dense {
	C := c.InChannels
	// Dynamically determine H and W from input
	flatLen := x.RawMatrix().Cols
	if flatLen%C != 0 {
		panic("Input size is not divisible by number of channels")
	}
	HW := flatLen / C
	H, W := 0, 0
	for h := 1; h <= HW; h++ {
		if HW%h == 0 {
			w := HW / h
			if h == w || h > w {
				H, W = h, w
				break
			}
		}
	}
	if H == 0 || W == 0 {
		panic("Unable to infer input height and width")
	}
	c.inH, c.inW = H, W
	KH, KW := c.KernelSize, c.KernelSize
	outH := (H+2*c.Padding-KH)/c.Stride + 1
	outW := (W+2*c.Padding-KW)/c.Stride + 1
	output := make([]float64, c.OutChannels*outH*outW)
	input := reshapeTo3D(x.RawRowView(0), C, H, W)
	c.input = [][][][]float64{input} // Save for backward
	for oc := 0; oc < c.OutChannels; oc++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				sum := c.Biases[oc]
				for ic := 0; ic < C; ic++ {
					for kh := 0; kh < KH; kh++ {
						for kw := 0; kw < KW; kw++ {
							ih := oh*c.Stride + kh - c.Padding
							iw := ow*c.Stride + kw - c.Padding
							if ih >= 0 && ih < H && iw >= 0 && iw < W {
								sum += input[ic][ih][iw] * c.Weights[oc][ic][kh][kw]
							}
						}
					}
				}
				output[oc*outH*outW+oh*outW+ow] = sum
			}
		}
	}
	return mat.NewDense(1, c.OutChannels*outH*outW, output)
}

func reshapeTo3D(flat []float64, c, h, w int) [][][]float64 {
	res := make([][][]float64, c)
	for i := 0; i < c; i++ {
		res[i] = make([][]float64, h)
		for j := 0; j < h; j++ {
			res[i][j] = make([]float64, w)
			for k := 0; k < w; k++ {
				res[i][j][k] = flat[i*h*w+j*w+k]
			}
		}
	}
	return res
}

func (c *Conv2D) inputH() int { return c.inH }
func (c *Conv2D) inputW() int { return c.inW }

func (c *Conv2D) initWeights() {
	c.Weights = make([][][][]float64, c.OutChannels)
	for oc := 0; oc < c.OutChannels; oc++ {
		c.Weights[oc] = make([][][]float64, c.InChannels)
		for ic := 0; ic < c.InChannels; ic++ {
			c.Weights[oc][ic] = make([][]float64, c.KernelSize)
			for kh := 0; kh < c.KernelSize; kh++ {
				c.Weights[oc][ic][kh] = make([]float64, c.KernelSize)
				for kw := 0; kw < c.KernelSize; kw++ {
					c.Weights[oc][ic][kh][kw] = rand.NormFloat64() * 0.1
				}
			}
		}
	}
	c.Biases = make([]float64, c.OutChannels)
}

func (c *Conv2D) Backward(dout *mat.Dense) *mat.Dense {
	// Assume dout is (1, OutChannels*outH*outW)
	C, H, W := c.InChannels, c.inputH(), c.inputW()
	KH, KW := c.KernelSize, c.KernelSize
	outH := (H+2*c.Padding-KH)/c.Stride + 1
	outW := (W+2*c.Padding-KW)/c.Stride + 1
	input := c.input[0] // Already [][][]float64
	gradInput := make([][][]float64, C)
	for ic := 0; ic < C; ic++ {
		gradInput[ic] = make([][]float64, H)
		for ih := 0; ih < H; ih++ {
			gradInput[ic][ih] = make([]float64, W)
		}
	}
	c.dWeights = make([][][][]float64, c.OutChannels)
	c.dBiases = make([]float64, c.OutChannels)
	for oc := 0; oc < c.OutChannels; oc++ {
		c.dWeights[oc] = make([][][]float64, C)
		for ic := 0; ic < C; ic++ {
			c.dWeights[oc][ic] = make([][]float64, KH)
			for kh := 0; kh < KH; kh++ {
				c.dWeights[oc][ic][kh] = make([]float64, KW)
			}
		}
	}
	for oc := 0; oc < c.OutChannels; oc++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				grad := dout.At(0, oc*outH*outW+oh*outW+ow)
				c.dBiases[oc] += grad
				for ic := 0; ic < C; ic++ {
					for kh := 0; kh < KH; kh++ {
						for kw := 0; kw < KW; kw++ {
							ih := oh*c.Stride + kh - c.Padding
							iw := ow*c.Stride + kw - c.Padding
							if ih >= 0 && ih < H && iw >= 0 && iw < W {
								c.dWeights[oc][ic][kh][kw] += input[ic][ih][iw] * grad
								gradInput[ic][ih][iw] += c.Weights[oc][ic][kh][kw] * grad
							}
						}
					}
				}
			}
		}
	}
	// Flatten gradInput to (1, C*H*W)
	flatGrad := make([]float64, C*H*W)
	for ic := 0; ic < C; ic++ {
		for ih := 0; ih < H; ih++ {
			for iw := 0; iw < W; iw++ {
				flatGrad[ic*H*W+ih*W+iw] = gradInput[ic][ih][iw]
			}
		}
	}
	return mat.NewDense(1, C*H*W, flatGrad)
}

func (c *Conv2D) GetWeights() *mat.Dense {
	// Flatten weights to (OutChannels, InChannels*KernelSize*KernelSize)
	rows := c.OutChannels
	cols := c.InChannels * c.KernelSize * c.KernelSize
	flat := make([]float64, rows*cols)
	for oc := 0; oc < c.OutChannels; oc++ {
		idx := 0
		for ic := 0; ic < c.InChannels; ic++ {
			for kh := 0; kh < c.KernelSize; kh++ {
				for kw := 0; kw < c.KernelSize; kw++ {
					flat[oc*cols+idx] = c.Weights[oc][ic][kh][kw]
					idx++
				}
			}
		}
	}
	return mat.NewDense(rows, cols, flat)
}

func (c *Conv2D) GetGradients() *mat.Dense {
	// Flatten dWeights to (OutChannels, InChannels*KernelSize*KernelSize)
	rows := c.OutChannels
	cols := c.InChannels * c.KernelSize * c.KernelSize
	flat := make([]float64, rows*cols)
	for oc := 0; oc < c.OutChannels; oc++ {
		idx := 0
		for ic := 0; ic < c.InChannels; ic++ {
			for kh := 0; kh < c.KernelSize; kh++ {
				for kw := 0; kw < c.KernelSize; kw++ {
					flat[oc*cols+idx] = c.dWeights[oc][ic][kh][kw]
					idx++
				}
			}
		}
	}
	return mat.NewDense(rows, cols, flat)
}

func (c *Conv2D) UpdateWeights(weightsUpdate *mat.Dense) {
	rows, _ := weightsUpdate.Dims()
	for oc := 0; oc < rows; oc++ {
		idx := 0
		for ic := 0; ic < c.InChannels; ic++ {
			for kh := 0; kh < c.KernelSize; kh++ {
				for kw := 0; kw < c.KernelSize; kw++ {
					c.Weights[oc][ic][kh][kw] = weightsUpdate.At(oc, idx)
					idx++
				}
			}
		}
	}
}

func (c *Conv2D) GetBiases() *mat.Dense {
	return mat.NewDense(1, len(c.Biases), c.Biases)
}

func (c *Conv2D) GetBiasGradients() *mat.Dense {
	return mat.NewDense(1, len(c.dBiases), c.dBiases)
}

func (c *Conv2D) UpdateBiases(biasUpdate *mat.Dense) {
	_, cols := biasUpdate.Dims()
	for i := 0; i < cols; i++ {
		c.Biases[i] = biasUpdate.At(0, i)
	}
}
