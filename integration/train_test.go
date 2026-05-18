package integration_test

import (
	"math/rand"
	"testing"

	"github.com/VigyatGoel/gotorch/layer"
	"github.com/VigyatGoel/gotorch/loss"
	"github.com/VigyatGoel/gotorch/network"
	"github.com/VigyatGoel/gotorch/optimizer"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func TestRegressionConvergence(t *testing.T) {
	// Dataset: y = 2*x1 + 3*x2 + noise(σ=0.01), n=100, deterministic seed.
	rng := rand.New(rand.NewSource(42))
	n := 100
	featureData := make([]float64, n*2)
	targetData := make([]float64, n)
	for i := 0; i < n; i++ {
		x1 := rng.Float64()*2 - 1
		x2 := rng.Float64()*2 - 1
		featureData[i*2] = x1
		featureData[i*2+1] = x2
		targetData[i] = 2*x1 + 3*x2 + rng.NormFloat64()*0.01
	}
	features := mat.NewDense(n, 2, featureData)
	targets := mat.NewDense(n, 1, targetData)

	model := network.NewSequential(
		layer.NewLinear(2, 8),
		layer.NewReLU(),
		layer.NewLinear(8, 1),
	)
	model.SetOptimizer(optimizer.DefaultAdam(0.01))
	lossFunc := loss.NewMSELoss()

	model.Train()
	var initialLoss, finalLoss float64
	for epoch := 0; epoch < 200; epoch++ {
		pred := model.Forward(features)
		l := lossFunc.Forward(pred, targets)
		if epoch == 0 {
			initialLoss = l
		}
		grad := lossFunc.Backward()
		model.Backward(grad)
		finalLoss = l
	}

	t.Logf("initial loss: %.6f, final loss: %.6f", initialLoss, finalLoss)

	require.Greater(t, initialLoss, 0.0, "initial loss must be positive")
	require.Less(t, finalLoss, initialLoss*0.05,
		"final loss must be <5%% of initial (initial=%.4f, final=%.4f)", initialLoss, finalLoss)
	require.Less(t, finalLoss, 0.1,
		"final loss must be below absolute threshold (got %.4f)", finalLoss)
}
