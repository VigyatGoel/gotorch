package layer

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

var rng = rand.New(rand.NewSource(42))

type Linear struct {
	inputMat   *mat.Dense
	weightMat  *mat.Dense
	biasMat    *mat.Dense
	dWeightMat *mat.Dense
	dBiasMat   *mat.Dense
}

func NewLinear(inFeatures, outFeatures int) *Linear {
	weightData := make([]float64, inFeatures*outFeatures)
	limit := math.Sqrt(6.0 / float64(inFeatures+outFeatures))
	for i := range weightData {
		weightData[i] = (rng.Float64()*2 - 1) * limit
	}
	return &Linear{
		weightMat:  mat.NewDense(inFeatures, outFeatures, weightData),
		biasMat:    mat.NewDense(1, outFeatures, make([]float64, outFeatures)),
		dWeightMat: mat.NewDense(inFeatures, outFeatures, make([]float64, inFeatures*outFeatures)),
		dBiasMat:   mat.NewDense(1, outFeatures, make([]float64, outFeatures)),
	}
}

func (l *Linear) Forward(x *mat.Dense) *mat.Dense {
	l.inputMat = x
	batchSize, _ := x.Dims()
	_, outF := l.weightMat.Dims()

	result := mat.NewDense(batchSize, outF, nil)
	result.Mul(x, l.weightMat)

	biasData := l.biasMat.RawMatrix().Data
	expandedBiasData := make([]float64, batchSize*outF)
	for i := 0; i < batchSize; i++ {
		copy(expandedBiasData[i*outF:(i+1)*outF], biasData)
	}
	expandedBias := mat.NewDense(batchSize, outF, expandedBiasData)
	result.Add(result, expandedBias)
	return result
}

func (l *Linear) Backward(gradOutput *mat.Dense) *mat.Dense {
	inF, outF := l.weightMat.Dims()
	batchSize, _ := gradOutput.Dims()

	weightGrad := mat.NewDense(inF, outF, nil)
	weightGrad.Mul(l.inputMat.T(), gradOutput)
	l.dWeightMat = weightGrad

	gradData := gradOutput.RawMatrix().Data
	biasGradData := make([]float64, outF)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < outF; j++ {
			biasGradData[j] += gradData[i*outF+j]
		}
	}
	l.dBiasMat = mat.NewDense(1, outF, biasGradData)

	gradInput := mat.NewDense(batchSize, inF, nil)
	gradInput.Mul(gradOutput, l.weightMat.T())
	return gradInput
}

func (l *Linear) GetWeights() *mat.Dense       { return l.weightMat }
func (l *Linear) GetGradients() *mat.Dense     { return l.dWeightMat }
func (l *Linear) UpdateWeights(w *mat.Dense)   { l.weightMat = mat.DenseCopyOf(w) }
func (l *Linear) GetBiases() *mat.Dense        { return l.biasMat }
func (l *Linear) GetBiasGradients() *mat.Dense { return l.dBiasMat }
func (l *Linear) UpdateBiases(b *mat.Dense)    { l.biasMat = mat.DenseCopyOf(b) }
func (l *Linear) ClearCache()                  { l.inputMat = nil }
