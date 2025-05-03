package optimizer

import (
	"fmt"
	"math"
)

type Adam struct {
	LR            float64
	Beta1         float64
	Beta2         float64
	Epsilon       float64
	T             int
	MomentMap     map[string][][]float64
	VelocMap      map[string][][]float64
	BiasMomentMap map[string][][]float64
	BiasVelocMap  map[string][][]float64
}

func NewAdam(lr float64, beta1, beta2, epsilon float64) *Adam {
	return &Adam{
		LR:            lr,
		Beta1:         beta1,
		Beta2:         beta2,
		Epsilon:       epsilon,
		T:             0,
		MomentMap:     make(map[string][][]float64),
		VelocMap:      make(map[string][][]float64),
		BiasMomentMap: make(map[string][][]float64),
		BiasVelocMap:  make(map[string][][]float64),
	}
}

func DefaultAdam(lr float64) *Adam {
	return NewAdam(lr, 0.9, 0.999, 1e-8)
}

func (a *Adam) Step(weights [][]float64, gradients [][]float64) [][]float64 {
	if len(weights) == 0 || len(gradients) == 0 {
		fmt.Println("Adam optimizer skipping weight update - empty weights or gradients")
		return weights
	}

	if len(weights) != len(gradients) {
		fmt.Printf("Adam optimizer dimension mismatch: weights %dx%d vs gradients %dx%d\n",
			len(weights), len(weights[0]), len(gradients), len(gradients[0]))
		return weights
	}

	a.T++

	key := fmt.Sprintf("weights_%dx%d", len(weights), len(weights[0]))

	if _, ok := a.MomentMap[key]; !ok {
		a.MomentMap[key] = make([][]float64, len(weights))
		a.VelocMap[key] = make([][]float64, len(weights))

		for i := range weights {
			a.MomentMap[key][i] = make([]float64, len(weights[i]))
			a.VelocMap[key][i] = make([]float64, len(weights[i]))
		}
	}

	beta1_t := math.Pow(a.Beta1, float64(a.T))
	beta2_t := math.Pow(a.Beta2, float64(a.T))

	for i := range weights {
		if len(weights[i]) != len(gradients[i]) {
			fmt.Printf("Adam optimizer: Inner dimension mismatch at row %d\n", i)
			continue
		}

		for j := range weights[i] {
			a.MomentMap[key][i][j] = a.Beta1*a.MomentMap[key][i][j] + (1.0-a.Beta1)*gradients[i][j]
			a.VelocMap[key][i][j] = a.Beta2*a.VelocMap[key][i][j] + (1.0-a.Beta2)*gradients[i][j]*gradients[i][j]
			m_hat := a.MomentMap[key][i][j] / (1.0 - beta1_t)
			v_hat := a.VelocMap[key][i][j] / (1.0 - beta2_t)
			weights[i][j] -= a.LR * m_hat / (math.Sqrt(v_hat) + a.Epsilon)
		}
	}

	return weights
}

func (a *Adam) StepBias(biases [][]float64, biasGradients [][]float64) [][]float64 {
	if len(biases) == 0 || len(biasGradients) == 0 || len(biases[0]) == 0 || len(biasGradients[0]) == 0 {
		fmt.Println("Adam optimizer skipping bias update - empty biases or gradients")
		return biases
	}
	if len(biases[0]) != len(biasGradients[0]) {
		fmt.Printf("Adam optimizer bias dimension mismatch: biases %dx%d vs gradients %dx%d\n",
			len(biases), len(biases[0]), len(biasGradients), len(biasGradients[0]))
		return biases
	}

	key := fmt.Sprintf("bias_%d", len(biases[0]))

	if _, ok := a.BiasMomentMap[key]; !ok {
		a.BiasMomentMap[key] = make([][]float64, 1)
		a.BiasVelocMap[key] = make([][]float64, 1)
		a.BiasMomentMap[key][0] = make([]float64, len(biases[0]))
		a.BiasVelocMap[key][0] = make([]float64, len(biases[0]))
	}

	beta1_t := math.Pow(a.Beta1, float64(a.T))
	beta2_t := math.Pow(a.Beta2, float64(a.T))

	for j := range biases[0] {
		a.BiasMomentMap[key][0][j] = a.Beta1*a.BiasMomentMap[key][0][j] + (1.0-a.Beta1)*biasGradients[0][j]
		a.BiasVelocMap[key][0][j] = a.Beta2*a.BiasVelocMap[key][0][j] + (1.0-a.Beta2)*biasGradients[0][j]*biasGradients[0][j]
		m_hat := a.BiasMomentMap[key][0][j] / (1.0 - beta1_t)
		v_hat := a.BiasVelocMap[key][0][j] / (1.0 - beta2_t)
		biases[0][j] -= a.LR * m_hat / (math.Sqrt(v_hat) + a.Epsilon)
	}

	return biases
}

func (a *Adam) ZeroGrad() {
	a.T = 0
	a.MomentMap = make(map[string][][]float64)
	a.VelocMap = make(map[string][][]float64)
	a.BiasMomentMap = make(map[string][][]float64)
	a.BiasVelocMap = make(map[string][][]float64)
}

func (a *Adam) GetLearningRate() float64 {
	return a.LR
}
