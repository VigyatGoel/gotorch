package layer

import (
	"math"
	"math/rand"

	"gorgonia.org/tensor"
)

var rng = rand.New(rand.NewSource(42))

// Linear represents a fully connected layer with learnable weights and biases
type Linear struct {
	inputMat   *tensor.Dense // cached input for backward pass
	weightMat  *tensor.Dense // learnable weight matrix
	biasMat    *tensor.Dense // learnable bias vector
	dWeightMat *tensor.Dense // weight gradients
	dBiasMat   *tensor.Dense // bias gradients
}

// NewLinear creates a new linear layer with Xavier initialization
func NewLinear(inFeatures, outFeatures int) *Linear {
	weightData := make([]float64, inFeatures*outFeatures)
	limit := math.Sqrt(6.0 / float64(inFeatures+outFeatures))
	for i := range weightData {
		weightData[i] = (rng.Float64()*2 - 1) * limit
	}
	weightMat := tensor.New(tensor.WithShape(inFeatures, outFeatures), tensor.WithBacking(weightData))

	biasData := make([]float64, outFeatures)
	biasMat := tensor.New(tensor.WithShape(1, outFeatures), tensor.WithBacking(biasData))

	dWeightMat := tensor.New(tensor.WithShape(inFeatures, outFeatures), tensor.WithBacking(make([]float64, inFeatures*outFeatures)))
	dBiasMat := tensor.New(tensor.WithShape(1, outFeatures), tensor.WithBacking(make([]float64, outFeatures)))

	return &Linear{
		weightMat:  weightMat,
		biasMat:    biasMat,
		dWeightMat: dWeightMat,
		dBiasMat:   dBiasMat,
	}
}

// Forward performs linear transformation: output = input * weight + bias
func (l *Linear) Forward(x *tensor.Dense) *tensor.Dense {
	l.inputMat = x

	result, err := tensor.MatMul(x, l.weightMat)
	if err != nil {
		panic(err)
	}

	resultShape := result.Shape()
	batchSize := resultShape[0]

	biasData := l.biasMat.Data().([]float64)
	expandedBiasData := make([]float64, batchSize*len(biasData))
	for i := 0; i < batchSize; i++ {
		copy(expandedBiasData[i*len(biasData):(i+1)*len(biasData)], biasData)
	}

	expandedBias := tensor.New(tensor.WithShape(resultShape...), tensor.WithBacking(expandedBiasData))

	resultWithBias, err := tensor.Add(result, expandedBias)
	if err != nil {
		panic(err)
	}

	return resultWithBias.(*tensor.Dense)
}

// Backward computes gradients for weights, biases, and input
func (l *Linear) Backward(gradOutput *tensor.Dense) *tensor.Dense {
	inputT, _ := tensor.Transpose(l.inputMat)
	weightGrad, _ := tensor.MatMul(inputT, gradOutput)
	l.dWeightMat = weightGrad.(*tensor.Dense)

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
			biasData := biasGrad.Data().([]float64)
			origShape := biasGrad.Shape()
			l.dBiasMat = tensor.New(tensor.WithShape(origShape...), tensor.WithBacking(biasData))
		}
		l.dBiasMat.Reshape(1, l.dBiasMat.Shape()[len(l.dBiasMat.Shape())-1])
	} else {
		l.dBiasMat = gradOutput.Clone().(*tensor.Dense)
		l.dBiasMat.Reshape(1, gradOutput.Shape()[0])
	}

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

// ClearCache releases cached input to prevent memory leaks
func (l *Linear) ClearCache() {
	l.inputMat = nil
}
