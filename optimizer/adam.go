package optimizer

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

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
		LR:      lr,
		Beta1:   beta1,
		Beta2:   beta2,
		Epsilon: epsilon,
		T:       0,
		m:       make(map[string]*mat.Dense),
		v:       make(map[string]*mat.Dense),
		mb:      make(map[string]*mat.Dense),
		vb:      make(map[string]*mat.Dense),
	}
}

func DefaultAdam(lr float64) *Adam {
	return NewAdam(lr, 0.9, 0.999, 1e-8)
}

func (a *Adam) Step(weights *mat.Dense, gradients *mat.Dense) *mat.Dense {
	if weights == nil || gradients == nil {
		return weights
	}
	a.T++
	rows, cols := weights.Dims()
	key := fmt.Sprintf("w%dx%d", rows, cols)
	if _, ok := a.m[key]; !ok {
		a.m[key] = mat.NewDense(rows, cols, nil)
		a.v[key] = mat.NewDense(rows, cols, nil)
	}
	m := a.m[key]
	v := a.v[key]
	beta1_t := math.Pow(a.Beta1, float64(a.T))
	beta2_t := math.Pow(a.Beta2, float64(a.T))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			g := gradients.At(i, j)
			m.Set(i, j, a.Beta1*m.At(i, j)+(1-a.Beta1)*g)
			v.Set(i, j, a.Beta2*v.At(i, j)+(1-a.Beta2)*g*g)
			mHat := m.At(i, j) / (1 - beta1_t)
			vHat := v.At(i, j) / (1 - beta2_t)
			w := weights.At(i, j) - a.LR*mHat/(math.Sqrt(vHat)+a.Epsilon)
			weights.Set(i, j, w)
		}
	}
	return weights
}

func (a *Adam) StepBias(biases *mat.Dense, biasGradients *mat.Dense) *mat.Dense {
	if biases == nil || biasGradients == nil {
		return biases
	}
	rows, cols := biases.Dims()
	key := fmt.Sprintf("b%dx%d", rows, cols)
	if _, ok := a.mb[key]; !ok {
		a.mb[key] = mat.NewDense(rows, cols, nil)
		a.vb[key] = mat.NewDense(rows, cols, nil)
	}
	mb := a.mb[key]
	vb := a.vb[key]
	beta1_t := math.Pow(a.Beta1, float64(a.T))
	beta2_t := math.Pow(a.Beta2, float64(a.T))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			g := biasGradients.At(i, j)
			mb.Set(i, j, a.Beta1*mb.At(i, j)+(1-a.Beta1)*g)
			vb.Set(i, j, a.Beta2*vb.At(i, j)+(1-a.Beta2)*g*g)
			mHat := mb.At(i, j) / (1 - beta1_t)
			vHat := vb.At(i, j) / (1 - beta2_t)
			b := biases.At(i, j) - a.LR*mHat/(math.Sqrt(vHat)+a.Epsilon)
			biases.Set(i, j, b)
		}
	}
	return biases
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
