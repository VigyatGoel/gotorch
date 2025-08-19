package optimizer

import (
	"fmt"
	"math"

	"gorgonia.org/tensor"
)

type Adam struct {
	LR      float64
	Beta1   float64
	Beta2   float64
	Epsilon float64
	T       int
	m       map[string]*tensor.Dense
	v       map[string]*tensor.Dense
	mb      map[string]*tensor.Dense
	vb      map[string]*tensor.Dense
}

func NewAdam(lr float64, beta1, beta2, epsilon float64) *Adam {
	return &Adam{
		LR:      lr,
		Beta1:   beta1,
		Beta2:   beta2,
		Epsilon: epsilon,
		T:       0,
		m:       make(map[string]*tensor.Dense),
		v:       make(map[string]*tensor.Dense),
		mb:      make(map[string]*tensor.Dense),
		vb:      make(map[string]*tensor.Dense),
	}
}

func DefaultAdam(lr float64) *Adam {
	return NewAdam(lr, 0.9, 0.999, 1e-8)
}

func (a *Adam) Step(weights *tensor.Dense, gradients *tensor.Dense) *tensor.Dense {
	if weights == nil || gradients == nil {
		return weights
	}
	a.T++
	shape := weights.Shape()
	key := fmt.Sprintf("w%v", shape)
	if _, ok := a.m[key]; !ok {
		// Initialize momentum and velocity tensors with zeros
		size := 1
		for _, dim := range shape {
			size *= dim
		}
		zeroData := make([]float64, size)
		a.m[key] = tensor.New(tensor.WithShape(shape...), tensor.WithBacking(zeroData))
		a.v[key] = tensor.New(tensor.WithShape(shape...), tensor.WithBacking(zeroData))
	}
	m := a.m[key]
	v := a.v[key]
	beta1_t := math.Pow(a.Beta1, float64(a.T))
	beta2_t := math.Pow(a.Beta2, float64(a.T))
	
	// Create copies of the data to avoid modifying the original tensors
	weightData := make([]float64, len(weights.Data().([]float64)))
	copy(weightData, weights.Data().([]float64))
	gradData := gradients.Data().([]float64)
	
	// Create copies of momentum and velocity data to avoid modifying them in place
	mData := make([]float64, len(m.Data().([]float64)))
	copy(mData, m.Data().([]float64))
	vData := make([]float64, len(v.Data().([]float64)))
	copy(vData, v.Data().([]float64))
	
	for i := range weightData {
		g := gradData[i]
		mData[i] = a.Beta1*mData[i] + (1-a.Beta1)*g
		vData[i] = a.Beta2*vData[i] + (1-a.Beta2)*g*g
		mHat := mData[i] / (1 - beta1_t)
		vHat := vData[i] / (1 - beta2_t)
		weightData[i] = weightData[i] - a.LR*mHat/(math.Sqrt(vHat)+a.Epsilon)
	}
	
	// Update the momentum and velocity tensors with the new values
	copy(m.Data().([]float64), mData)
	copy(v.Data().([]float64), vData)
	
	// Return a new tensor with updated weights
	return tensor.New(tensor.WithShape(shape...), tensor.WithBacking(weightData))
}

func (a *Adam) StepBias(biases *tensor.Dense, biasGradients *tensor.Dense) *tensor.Dense {
	if biases == nil || biasGradients == nil {
		return biases
	}
	shape := biases.Shape()
	key := fmt.Sprintf("b%v", shape)
	if _, ok := a.mb[key]; !ok {
		// Initialize momentum and velocity tensors with zeros
		size := 1
		for _, dim := range shape {
			size *= dim
		}
		zeroData := make([]float64, size)
		a.mb[key] = tensor.New(tensor.WithShape(shape...), tensor.WithBacking(zeroData))
		a.vb[key] = tensor.New(tensor.WithShape(shape...), tensor.WithBacking(zeroData))
	}
	mb := a.mb[key]
	vb := a.vb[key]
	beta1_t := math.Pow(a.Beta1, float64(a.T))
	beta2_t := math.Pow(a.Beta2, float64(a.T))
	
	// Create copies of the data to avoid modifying the original tensors
	biasData := make([]float64, len(biases.Data().([]float64)))
	copy(biasData, biases.Data().([]float64))
	biasGradData := biasGradients.Data().([]float64)
	
	// Create copies of momentum and velocity data to avoid modifying them in place
	mbData := make([]float64, len(mb.Data().([]float64)))
	copy(mbData, mb.Data().([]float64))
	vbData := make([]float64, len(vb.Data().([]float64)))
	copy(vbData, vb.Data().([]float64))
	
	for i := range biasData {
		g := biasGradData[i]
		mbData[i] = a.Beta1*mbData[i] + (1-a.Beta1)*g
		vbData[i] = a.Beta2*vbData[i] + (1-a.Beta2)*g*g
		mHat := mbData[i] / (1 - beta1_t)
		vHat := vbData[i] / (1 - beta2_t)
		biasData[i] = biasData[i] - a.LR*mHat/(math.Sqrt(vHat)+a.Epsilon)
	}
	
	// Update the momentum and velocity tensors with the new values
	copy(mb.Data().([]float64), mbData)
	copy(vb.Data().([]float64), vbData)
	
	// Return a new tensor with updated biases
	return tensor.New(tensor.WithShape(shape...), tensor.WithBacking(biasData))
}

func (a *Adam) ZeroGrad() {
	a.T = 0
	a.m = make(map[string]*tensor.Dense)
	a.v = make(map[string]*tensor.Dense)
	a.mb = make(map[string]*tensor.Dense)
	a.vb = make(map[string]*tensor.Dense)
}

func (a *Adam) GetLearningRate() float64 {
	return a.LR
}
