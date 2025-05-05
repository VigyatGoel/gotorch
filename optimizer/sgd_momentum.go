package optimizer

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type SGDMomentum struct {
	LR       float64
	Momentum float64
	v        map[string]*mat.Dense
	vb       map[string]*mat.Dense
}

func NewSGDMomentum(lr float64, momentum float64) *SGDMomentum {
	return &SGDMomentum{
		LR:       lr,
		Momentum: momentum,
		v:        make(map[string]*mat.Dense),
		vb:       make(map[string]*mat.Dense),
	}
}

func DefaultSGDMomentum(lr float64) *SGDMomentum {
	return NewSGDMomentum(lr, 0.9)
}

func (sgd *SGDMomentum) Step(weights *mat.Dense, gradients *mat.Dense) *mat.Dense {
	rows, cols := weights.Dims()
	if rows == 0 || cols == 0 {
		fmt.Println("SGD optimizer skipping update - empty weights or gradients")
		return weights
	}

	gRows, gCols := gradients.Dims()
	if rows != gRows || cols != gCols {
		fmt.Printf("SGD optimizer dimension mismatch: weights %dx%d vs gradients %dx%d\n", rows, cols, gRows, gCols)
		return weights
	}

	key := fmt.Sprintf("weights_%dx%d", rows, cols)
	if _, ok := sgd.v[key]; !ok {
		sgd.v[key] = mat.NewDense(rows, cols, nil)
	}

	v := sgd.v[key]
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			v.Set(i, j, sgd.Momentum*v.At(i, j)-sgd.LR*gradients.At(i, j))
			weights.Set(i, j, weights.At(i, j)+v.At(i, j))
		}
	}

	return weights
}

func (sgd *SGDMomentum) StepBias(biases *mat.Dense, biasGradients *mat.Dense) *mat.Dense {
	if biases == nil || biasGradients == nil {
		return biases
	}

	rows, cols := biases.Dims()
	if rows == 0 || cols == 0 {
		return biases
	}

	gRows, gCols := biasGradients.Dims()
	if rows != gRows || cols != gCols {
		fmt.Printf("SGD optimizer dimension mismatch: biases %dx%d vs biasGradients %dx%d\n", rows, cols, gRows, gCols)
		return biases
	}

	key := fmt.Sprintf("bias_%dx%d", rows, cols)
	if _, ok := sgd.vb[key]; !ok {
		sgd.vb[key] = mat.NewDense(rows, cols, nil)
	}

	vb := sgd.vb[key]
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			vb.Set(i, j, sgd.Momentum*vb.At(i, j)-sgd.LR*biasGradients.At(i, j))
			biases.Set(i, j, biases.At(i, j)+vb.At(i, j))
		}
	}

	return biases
}

func (sgd *SGDMomentum) ZeroGrad() {
	sgd.v = make(map[string]*mat.Dense)
	sgd.vb = make(map[string]*mat.Dense)
}

func (sgd *SGDMomentum) GetLearningRate() float64 {
	return sgd.LR
}
