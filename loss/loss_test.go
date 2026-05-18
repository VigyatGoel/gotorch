package loss

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
)

// ─── MSELoss ──────────────────────────────────────────────────────────────

type MSESuite struct{ suite.Suite }

func TestMSE(t *testing.T) { suite.Run(t, new(MSESuite)) }

func (s *MSESuite) TestForwardKnown() {
	// preds=[0,2], targets=[1,1]: MSE = ((0-1)^2 + (2-1)^2) / 2 = 1.0
	l := NewMSELoss()
	got := l.Forward(
		mat.NewDense(1, 2, []float64{0, 2}),
		mat.NewDense(1, 2, []float64{1, 1}),
	)
	s.InDelta(1.0, got, 1e-10)
}

func (s *MSESuite) TestForwardZero() {
	l := NewMSELoss()
	x := mat.NewDense(1, 3, []float64{1, 2, 3})
	s.InDelta(0.0, l.Forward(x, x), 1e-10)
}

func (s *MSESuite) TestBackwardFormula() {
	// grad[i] = 2*(pred[i]-target[i]) / n
	// preds=[0,2], targets=[1,1], n=2:
	// grad[0] = 2*(0-1)/2 = -1, grad[1] = 2*(2-1)/2 = 1
	l := NewMSELoss()
	l.Forward(
		mat.NewDense(1, 2, []float64{0, 2}),
		mat.NewDense(1, 2, []float64{1, 1}),
	)
	grad := l.Backward()
	s.InDelta(-1.0, grad.At(0, 0), 1e-10)
	s.InDelta(1.0, grad.At(0, 1), 1e-10)
}

func (s *MSESuite) TestBackwardGradCheck() {
	l := NewMSELoss()
	preds := mat.NewDense(1, 3, []float64{0.5, 1.0, 1.5})
	targets := mat.NewDense(1, 3, []float64{1.0, 0.5, 2.0})
	l.Forward(preds, targets)
	analytical := l.Backward()
	aData := analytical.RawMatrix().Data

	// aData is captured before the loop begins so the analytical gradient is
	// safe from the in-loop Forward calls that mutate predData.
	eps := 1e-4
	predData := preds.RawMatrix().Data
	for i := range predData {
		orig := predData[i]
		predData[i] = orig + eps
		fPlus := l.Forward(preds, targets)
		predData[i] = orig - eps
		fMinus := l.Forward(preds, targets)
		predData[i] = orig
		numerical := (fPlus - fMinus) / (2 * eps)
		assert.InDelta(s.T(), numerical, aData[i], 1e-5, "grad at %d", i)
	}
}

// ─── CrossEntropyLoss ─────────────────────────────────────────────────────

type CrossEntropySuite struct{ suite.Suite }

func TestCrossEntropy(t *testing.T) { suite.Run(t, new(CrossEntropySuite)) }

func (s *CrossEntropySuite) TestForwardKnown() {
	// preds=[0.9,0.1], target=[1,0]: loss = -log(0.9+1e-9) / 1
	l := NewCrossEntropyLoss()
	want := -math.Log(0.9 + 1e-9)
	got := l.Forward(
		mat.NewDense(1, 2, []float64{0.9, 0.1}),
		mat.NewDense(1, 2, []float64{1, 0}),
	)
	s.InDelta(want, got, 1e-8)
}

func (s *CrossEntropySuite) TestForwardPerfectPrediction() {
	// Prediction very close to 1 for the correct class → loss ≈ 0.
	l := NewCrossEntropyLoss()
	got := l.Forward(
		mat.NewDense(1, 2, []float64{1.0 - 1e-9, 1e-9}),
		mat.NewDense(1, 2, []float64{1, 0}),
	)
	s.Less(got, 1e-7)
}

func (s *CrossEntropySuite) TestBackwardFormula() {
	// grad[i] = (pred[i] - target[i]) / rows
	// preds=[0.7,0.3], targets=[1,0], rows=1:
	// grad[0]=(0.7-1)/1=-0.3, grad[1]=(0.3-0)/1=0.3
	l := NewCrossEntropyLoss()
	l.Forward(
		mat.NewDense(1, 2, []float64{0.7, 0.3}),
		mat.NewDense(1, 2, []float64{1, 0}),
	)
	grad := l.Backward()
	s.InDelta(-0.3, grad.At(0, 0), 1e-10)
	s.InDelta(0.3, grad.At(0, 1), 1e-10)
}

func (s *CrossEntropySuite) TestBackwardMultiRow() {
	// grad[i] = (pred[i] - target[i]) / rows
	// preds=[[0.7,0.2,0.1],[0.1,0.8,0.1]], targets=[[1,0,0],[0,1,0]], rows=2
	// grad[0,0]=(0.7-1)/2=-0.15, grad[0,1]=(0.2-0)/2=0.1, grad[0,2]=(0.1-0)/2=0.05
	// grad[1,0]=(0.1-0)/2=0.05, grad[1,1]=(0.8-1)/2=-0.1, grad[1,2]=(0.1-0)/2=0.05
	l := NewCrossEntropyLoss()
	preds := mat.NewDense(2, 3, []float64{0.7, 0.2, 0.1, 0.1, 0.8, 0.1})
	targets := mat.NewDense(2, 3, []float64{1, 0, 0, 0, 1, 0})
	l.Forward(preds, targets)
	grad := l.Backward()

	assert.InDelta(s.T(), -0.15, grad.At(0, 0), 1e-10)
	assert.InDelta(s.T(), 0.1, grad.At(0, 1), 1e-10)
	assert.InDelta(s.T(), 0.05, grad.At(0, 2), 1e-10)
	assert.InDelta(s.T(), 0.05, grad.At(1, 0), 1e-10)
	assert.InDelta(s.T(), -0.1, grad.At(1, 1), 1e-10)
	assert.InDelta(s.T(), 0.05, grad.At(1, 2), 1e-10)
}
