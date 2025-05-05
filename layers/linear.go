package layer

import (
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
	weightMat := mat.NewDense(inFeatures, outFeatures, nil)
	biasMat := mat.NewDense(1, outFeatures, nil)
	for i := 0; i < inFeatures; i++ {
		for j := 0; j < outFeatures; j++ {
			weightMat.Set(i, j, rng.Float64()*2-1)
		}
	}
	dWeightMat := mat.NewDense(inFeatures, outFeatures, nil)
	dBiasMat := mat.NewDense(1, outFeatures, nil)
	return &Linear{
		weightMat:  weightMat,
		biasMat:    biasMat,
		dWeightMat: dWeightMat,
		dBiasMat:   dBiasMat,
	}
}

func (l *Linear) Forward(x *mat.Dense) *mat.Dense {
	l.inputMat = x
	rows, _ := x.Dims()
	_, cols := l.weightMat.Dims()
	result := mat.NewDense(rows, cols, nil)
	result.Mul(x, l.weightMat)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Set(i, j, result.At(i, j)+l.biasMat.At(0, j))
		}
	}
	return result
}

func (l *Linear) Backward(gradOutput *mat.Dense) *mat.Dense {
	inputRows, inputCols := l.inputMat.Dims()
	inputT := mat.NewDense(inputCols, inputRows, nil)
	for i := 0; i < inputRows; i++ {
		for j := 0; j < inputCols; j++ {
			inputT.Set(j, i, l.inputMat.At(i, j))
		}
	}
	l.dWeightMat.Mul(inputT, gradOutput)
	rows, cols := gradOutput.Dims()
	l.dBiasMat.Zero()
	for j := 0; j < cols; j++ {
		var sum float64
		for i := 0; i < rows; i++ {
			sum += gradOutput.At(i, j)
		}
		l.dBiasMat.Set(0, j, sum)
	}
	weightRows, weightCols := l.weightMat.Dims()
	weightT := mat.NewDense(weightCols, weightRows, nil)
	for i := 0; i < weightRows; i++ {
		for j := 0; j < weightCols; j++ {
			weightT.Set(j, i, l.weightMat.At(i, j))
		}
	}
	gradInputMat := mat.NewDense(rows, weightT.RawMatrix().Cols, nil)
	gradInputMat.Mul(gradOutput, weightT)
	return gradInputMat
}

func (l *Linear) GetWeights() *mat.Dense {
	return l.weightMat
}

func (l *Linear) GetGradients() *mat.Dense {
	return l.dWeightMat
}

func (l *Linear) UpdateWeights(weightsUpdate *mat.Dense) {
	l.weightMat.CloneFrom(weightsUpdate)
}

func (l *Linear) GetBiases() *mat.Dense {
	return l.biasMat
}

func (l *Linear) GetBiasGradients() *mat.Dense {
	return l.dBiasMat
}

func (l *Linear) UpdateBiases(biasUpdate *mat.Dense) {
	l.biasMat.CloneFrom(biasUpdate)
}
