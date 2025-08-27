package layer

import (
	"math"
	"math/rand"

	"gorgonia.org/tensor"
)

var rng = rand.New(rand.NewSource(42))

type Linear struct {
	inputMat   *tensor.Dense
	weightMat  *tensor.Dense
	biasMat    *tensor.Dense
	dWeightMat *tensor.Dense
	dBiasMat   *tensor.Dense
}

func NewLinear(inFeatures, outFeatures int) *Linear {
	// Initialize weights with Xavier/Glorot initialization
	weightData := make([]float64, inFeatures*outFeatures)
	limit := math.Sqrt(6.0 / float64(inFeatures+outFeatures))
	for i := range weightData {
		weightData[i] = (rng.Float64()*2 - 1) * limit
	}
	weightMat := tensor.New(tensor.WithShape(inFeatures, outFeatures), tensor.WithBacking(weightData))

	// Initialize biases with zeros
	biasData := make([]float64, outFeatures)
	biasMat := tensor.New(tensor.WithShape(1, outFeatures), tensor.WithBacking(biasData))

	// Initialize gradient tensors
	dWeightMat := tensor.New(tensor.WithShape(inFeatures, outFeatures), tensor.WithBacking(make([]float64, inFeatures*outFeatures)))
	dBiasMat := tensor.New(tensor.WithShape(1, outFeatures), tensor.WithBacking(make([]float64, outFeatures)))

	return &Linear{
		weightMat:  weightMat,
		biasMat:    biasMat,
		dWeightMat: dWeightMat,
		dBiasMat:   dBiasMat,
	}
}

func (l *Linear) Forward(x *tensor.Dense) *tensor.Dense {
	l.inputMat = x

	// Matrix multiplication: x * weightMat
	result, err := tensor.MatMul(x, l.weightMat)
	if err != nil {
		panic(err)
	}

	// Get the result shape
	resultShape := result.Shape()
	batchSize := resultShape[0]

	// Create a bias tensor with the same shape as result by repeating the bias
	biasData := l.biasMat.Data().([]float64)
	expandedBiasData := make([]float64, batchSize*len(biasData))
	for i := 0; i < batchSize; i++ {
		copy(expandedBiasData[i*len(biasData):(i+1)*len(biasData)], biasData)
	}

	// Create the expanded bias tensor
	expandedBias := tensor.New(tensor.WithShape(resultShape...), tensor.WithBacking(expandedBiasData))

	// Add bias
	resultWithBias, err := tensor.Add(result, expandedBias)
	if err != nil {
		panic(err)
	}

	return resultWithBias.(*tensor.Dense)
}

func (l *Linear) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	// Calculate weight gradients: input.T * gradOutput
	inputT, _ := tensor.Transpose(l.inputMat)
	weightGrad, _ := tensor.MatMul(inputT, gradOutput)
	l.dWeightMat = weightGrad.(*tensor.Dense)

	// Calculate bias gradients: sum(gradOutput, axis=0)
	gradShape := gradOutput.Shape()
	if len(gradShape) > 1 {
		axis := make([]int, len(gradShape)-1)
		for i := 0; i < len(axis); i++ {
			axis[i] = i
		}
		biasGrad, _ := tensor.Sum(gradOutput, axis...)
		if bd, ok := biasGrad.(*tensor.Dense); ok {
			l.dBiasMat = bd
		} else {
			// Handle case where Sum returns a different type
			biasData := biasGrad.Data().([]float64)
			origShape := biasGrad.Shape()
			l.dBiasMat = tensor.New(tensor.WithShape(origShape...), tensor.WithBacking(biasData))
		}
		// Ensure bias gradient has the right shape (1, outFeatures)
		l.dBiasMat.Reshape(1, l.dBiasMat.Shape()[len(l.dBiasMat.Shape())-1])
	} else {
		// If gradOutput is 1D, just copy it
		l.dBiasMat = gradOutput.Clone().(*tensor.Dense)
		l.dBiasMat.Reshape(1, gradOutput.Shape()[0])
	}

	// Calculate input gradients: gradOutput * weightMat.T
	weightT, _ := tensor.Transpose(l.weightMat)
	gradInputMat, _ := tensor.MatMul(gradOutput, weightT)

	return gradInputMat.(*tensor.Dense)
}

func (l *Linear) GetWeights() *tensor.Dense {
	return l.weightMat
}

func (l *Linear) GetGradients() *tensor.Dense {
	return l.dWeightMat
}

func (l *Linear) UpdateWeights(weightsUpdate *tensor.Dense) {
	l.weightMat = weightsUpdate.Clone().(*tensor.Dense)
}

func (l *Linear) GetBiases() *tensor.Dense {
	return l.biasMat
}

func (l *Linear) GetBiasGradients() *tensor.Dense {
	return l.dBiasMat
}

func (l *Linear) UpdateBiases(biasUpdate *tensor.Dense) {
	l.biasMat = biasUpdate.Clone().(*tensor.Dense)
}

// ClearCache clears cached data to prevent memory leaks
func (l *Linear) ClearCache() {
	l.inputMat = nil
}
