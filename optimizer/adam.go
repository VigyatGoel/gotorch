package optimizer

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// Adam implements the Adam optimizer (Kingma & Ba 2015)
type Adam struct {
	LR      float64
	Beta1   float64
	Beta2   float64
	Epsilon float64
	T       int
	m       map[string]*mat.Dense
	v       map[string]*mat.Dense
	mb      map[string]*mat.Dense
	vb      map[string]*mat.Dense
}

func NewAdam(lr float64, beta1, beta2, epsilon float64) *Adam {
	return &Adam{
		LR: lr, Beta1: beta1, Beta2: beta2, Epsilon: epsilon,
		m:  make(map[string]*mat.Dense),
		v:  make(map[string]*mat.Dense),
		mb: make(map[string]*mat.Dense),
		vb: make(map[string]*mat.Dense),
	}
}

func DefaultAdam(lr float64) *Adam {
	return NewAdam(lr, 0.9, 0.999, 1e-8)
}

func (a *Adam) step(params, grads *mat.Dense, m, v map[string]*mat.Dense, prefix string) *mat.Dense {
	if params == nil || grads == nil {
		return params
	}
	rows, cols := params.Dims()
	key := fmt.Sprintf("%s_%d_%d", prefix, rows, cols)
	if _, ok := m[key]; !ok {
		m[key] = mat.NewDense(rows, cols, nil)
		v[key] = mat.NewDense(rows, cols, nil)
	}
	beta1_t := math.Pow(a.Beta1, float64(a.T))
	beta2_t := math.Pow(a.Beta2, float64(a.T))
	pData := params.RawMatrix().Data
	gData := grads.RawMatrix().Data
	mData := m[key].RawMatrix().Data
	vData := v[key].RawMatrix().Data
	resultData := make([]float64, rows*cols)
	for i := range resultData {
		mData[i] = a.Beta1*mData[i] + (1-a.Beta1)*gData[i]
		vData[i] = a.Beta2*vData[i] + (1-a.Beta2)*gData[i]*gData[i]
		mHat := mData[i] / (1 - beta1_t)
		vHat := vData[i] / (1 - beta2_t)
		resultData[i] = pData[i] - a.LR*mHat/(math.Sqrt(vHat)+a.Epsilon)
	}
	return mat.NewDense(rows, cols, resultData)
}

func (a *Adam) Step(weights *mat.Dense, gradients *mat.Dense) *mat.Dense {
	if weights == nil || gradients == nil {
		return weights
	}
	a.T++
	return a.step(weights, gradients, a.m, a.v, "w")
}

func (a *Adam) StepBias(biases *mat.Dense, biasGradients *mat.Dense) *mat.Dense {
	if biases == nil || biasGradients == nil {
		return biases
	}
	return a.step(biases, biasGradients, a.mb, a.vb, "b")
}

func (a *Adam) ZeroGrad() {
	a.T = 0
	a.m = make(map[string]*mat.Dense)
	a.v = make(map[string]*mat.Dense)
	a.mb = make(map[string]*mat.Dense)
	a.vb = make(map[string]*mat.Dense)
}

func (a *Adam) GetLearningRate() float64 {
	return a.LR
}
