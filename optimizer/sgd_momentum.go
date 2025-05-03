package optimizer

import "fmt"

type SGDMomentum struct {
	LR              float64
	Momentum        float64
	VelocityMap     map[string][][]float64
	BiasVelocityMap map[string][][]float64
}

func NewSGDMomentum(lr float64, momentum float64) *SGDMomentum {
	return &SGDMomentum{
		LR:              lr,
		Momentum:        momentum,
		VelocityMap:     make(map[string][][]float64),
		BiasVelocityMap: make(map[string][][]float64),
	}
}

func DefaultSGDMomentum(lr float64) *SGDMomentum {
	return NewSGDMomentum(lr, 0.9)
}

func (sgd *SGDMomentum) Step(weights [][]float64, gradients [][]float64) [][]float64 {
	if len(weights) == 0 || len(gradients) == 0 {
		fmt.Println("SGD optimizer skipping update - empty weights or gradients")
		return weights
	}

	if len(weights) != len(gradients) {
		fmt.Printf("SGD optimizer dimension mismatch: weights %dx%d vs gradients %dx%d\n",
			len(weights), len(weights[0]), len(gradients), len(gradients[0]))
		return weights
	}

	key := fmt.Sprintf("weights_%dx%d", len(weights), len(weights[0]))

	if _, ok := sgd.VelocityMap[key]; !ok {
		sgd.VelocityMap[key] = make([][]float64, len(weights))

		for i := range weights {
			sgd.VelocityMap[key][i] = make([]float64, len(weights[i]))
			for j := range weights[i] {
				sgd.VelocityMap[key][i][j] = 0.0
			}
		}
	}

	for i := range weights {
		if len(weights[i]) != len(gradients[i]) {
			fmt.Printf("SGD optimizer: Inner dimension mismatch at row %d\n", i)
			continue
		}

		for j := range weights[i] {
			sgd.VelocityMap[key][i][j] = sgd.Momentum*sgd.VelocityMap[key][i][j] - sgd.LR*gradients[i][j]
			weights[i][j] += sgd.VelocityMap[key][i][j]
		}
	}

	return weights
}

func (sgd *SGDMomentum) StepBias(biases [][]float64, biasGradients [][]float64) [][]float64 {
	if len(biases) == 0 || len(biasGradients) == 0 || len(biases[0]) == 0 || len(biasGradients[0]) == 0 {
		return biases
	}

	key := fmt.Sprintf("bias_%d", len(biases[0]))

	if _, ok := sgd.BiasVelocityMap[key]; !ok {
		sgd.BiasVelocityMap[key] = make([][]float64, 1)
		sgd.BiasVelocityMap[key][0] = make([]float64, len(biases[0]))
	}

	for j := range biases[0] {
		sgd.BiasVelocityMap[key][0][j] = sgd.Momentum*sgd.BiasVelocityMap[key][0][j] - sgd.LR*biasGradients[0][j]
		biases[0][j] += sgd.BiasVelocityMap[key][0][j]
	}
	return biases
}

func (sgd *SGDMomentum) ZeroGrad() {
	sgd.VelocityMap = make(map[string][][]float64)
	sgd.BiasVelocityMap = make(map[string][][]float64)
}

func (sgd *SGDMomentum) GetLearningRate() float64 {
	return sgd.LR
}
