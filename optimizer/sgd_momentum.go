package optimizer

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// SGDMomentum implements SGD with momentum: v = momentum*v + grad; param -= lr*v
type SGDMomentum struct {
	LR       float64
	Momentum float64
	v        map[string]*mat.Dense
	vb       map[string]*mat.Dense
}

func NewSGDMomentum(lr float64, momentum float64) *SGDMomentum {
	return &SGDMomentum{
		LR: lr, Momentum: momentum,
		v:  make(map[string]*mat.Dense),
		vb: make(map[string]*mat.Dense),
	}
}

func DefaultSGDMomentum(lr float64) *SGDMomentum {
	return NewSGDMomentum(lr, 0.9)
}

func (sgd *SGDMomentum) step(weights, gradients *mat.Dense, cache map[string]*mat.Dense, prefix string) *mat.Dense {
	if weights == nil || gradients == nil {
		return weights
	}
	rows, cols := weights.Dims()
	gr, gc := gradients.Dims()
	if rows != gr || cols != gc {
		fmt.Printf("SGDMomentum dimension mismatch: weights %dx%d vs gradients %dx%d\n", rows, cols, gr, gc)
		return weights
	}
	key := fmt.Sprintf("%s_%d_%d", prefix, rows, cols)
	if _, ok := cache[key]; !ok {
		cache[key] = mat.NewDense(rows, cols, nil)
	}
	v := cache[key]
	wData := weights.RawMatrix().Data
	gData := gradients.RawMatrix().Data
	vData := v.RawMatrix().Data
	resultData := make([]float64, rows*cols)
	for i := range resultData {
		vData[i] = sgd.Momentum*vData[i] + gData[i]
		resultData[i] = wData[i] - sgd.LR*vData[i]
	}
	return mat.NewDense(rows, cols, resultData)
}

func (sgd *SGDMomentum) Step(weights *mat.Dense, gradients *mat.Dense) *mat.Dense {
	return sgd.step(weights, gradients, sgd.v, "w")
}

func (sgd *SGDMomentum) StepBias(biases *mat.Dense, biasGradients *mat.Dense) *mat.Dense {
	return sgd.step(biases, biasGradients, sgd.vb, "b")
}

func (sgd *SGDMomentum) ZeroGrad() {
	for _, v := range sgd.v {
		data := v.RawMatrix().Data
		for i := range data {
			data[i] = 0.0
		}
	}
	for _, vb := range sgd.vb {
		data := vb.RawMatrix().Data
		for i := range data {
			data[i] = 0.0
		}
	}
}

func (sgd *SGDMomentum) GetLearningRate() float64 {
	return sgd.LR
}
