package optimizer

import (
	"math"
	"testing"

	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
)

// ─── SGD ──────────────────────────────────────────────────────────────────

type SGDSuite struct{ suite.Suite }

func TestSGD(t *testing.T) { suite.Run(t, new(SGDSuite)) }

func (s *SGDSuite) TestStep() {
	// w_new[i] = w[i] - lr*g[i]
	sgd := NewSGD(0.01)
	w := mat.NewDense(1, 2, []float64{1.0, 2.0})
	g := mat.NewDense(1, 2, []float64{0.1, 0.2})
	got := sgd.Step(w, g)
	s.InDelta(1.0-0.01*0.1, got.At(0, 0), 1e-10)
	s.InDelta(2.0-0.01*0.2, got.At(0, 1), 1e-10)
}

func (s *SGDSuite) TestStepBias() {
	sgd := NewSGD(0.1)
	b := mat.NewDense(1, 1, []float64{5.0})
	g := mat.NewDense(1, 1, []float64{1.0})
	got := sgd.StepBias(b, g)
	s.InDelta(4.9, got.At(0, 0), 1e-10)
}

func (s *SGDSuite) TestStepBiasNilSafe() {
	sgd := NewSGD(0.01)
	s.Nil(sgd.StepBias(nil, nil))
}

func (s *SGDSuite) TestZeroGradNoop() {
	sgd := NewSGD(0.01)
	sgd.ZeroGrad() // must not panic
}

func (s *SGDSuite) TestGetLearningRate() {
	s.InDelta(0.05, NewSGD(0.05).GetLearningRate(), 1e-10)
}

// ─── SGDMomentum ──────────────────────────────────────────────────────────

type SGDMomentumSuite struct{ suite.Suite }

func TestSGDMomentum(t *testing.T) { suite.Run(t, new(SGDMomentumSuite)) }

func (s *SGDMomentumSuite) TestTwoStepMomentumAccumulates() {
	// Step 1: v=g1=0.1; w=1.0-0.01*0.1=0.999
	// Step 2: v=0.9*0.1+0.2=0.29; w=0.999-0.01*0.29=0.9961
	sgd := NewSGDMomentum(0.01, 0.9)
	w := mat.NewDense(1, 1, []float64{1.0})
	g1 := mat.NewDense(1, 1, []float64{0.1})
	g2 := mat.NewDense(1, 1, []float64{0.2})
	w = sgd.Step(w, g1)
	s.InDelta(0.999, w.At(0, 0), 1e-10)
	w = sgd.Step(w, g2)
	s.InDelta(0.999-0.01*0.29, w.At(0, 0), 1e-10)
}

func (s *SGDMomentumSuite) TestZeroGradResetsVelocity() {
	// After ZeroGrad, a step with the same gradient produces the same result as the first step.
	sgd := NewSGDMomentum(0.01, 0.9)
	w := mat.NewDense(1, 1, []float64{1.0})
	g := mat.NewDense(1, 1, []float64{0.5})
	sgd.Step(w, g) // step 1: v=0.5, velocity stored
	sgd.ZeroGrad() // reset velocity to zero

	// Fresh start: same initial w, same g, velocity starts at 0 again.
	wFresh := mat.NewDense(1, 1, []float64{1.0})
	got := sgd.Step(wFresh, g) // v=0.5, w=1.0-0.01*0.5=0.995
	s.InDelta(0.995, got.At(0, 0), 1e-10)
}

func (s *SGDMomentumSuite) TestGetLearningRate() {
	s.InDelta(0.001, NewSGDMomentum(0.001, 0.9).GetLearningRate(), 1e-10)
}

// ─── Adam ─────────────────────────────────────────────────────────────────

type AdamSuite struct{ suite.Suite }

func TestAdam(t *testing.T) { suite.Run(t, new(AdamSuite)) }

func (s *AdamSuite) TestStep1BiasCorrection() {
	// Manual calculation for step 1:
	// m = (1-0.9)*0.1 = 0.01, v = (1-0.999)*0.01 = 0.00001
	// m_hat = 0.01/(1-0.9^1) = 0.1, v_hat = 0.00001/(1-0.999^1) = 0.01
	// w_new = 1.0 - 0.01*0.1/(sqrt(0.01)+1e-8)
	a := NewAdam(0.01, 0.9, 0.999, 1e-8)
	w := mat.NewDense(1, 1, []float64{1.0})
	g := mat.NewDense(1, 1, []float64{0.1})
	got := a.Step(w, g)
	mHat := 0.01 / (1 - 0.9)
	vHat := 0.00001 / (1 - 0.999)
	want := 1.0 - 0.01*mHat/(math.Sqrt(vHat)+1e-8)
	s.InDelta(want, got.At(0, 0), 1e-10)
}

func (s *AdamSuite) TestStepIncrementsT() {
	a := NewAdam(0.01, 0.9, 0.999, 1e-8)
	s.Equal(0, a.T)
	w := mat.NewDense(1, 1, []float64{1.0})
	g := mat.NewDense(1, 1, []float64{0.1})
	a.Step(w, g)
	s.Equal(1, a.T)
	a.Step(w, g)
	s.Equal(2, a.T)
}

func (s *AdamSuite) TestZeroGradResetsState() {
	a := NewAdam(0.01, 0.9, 0.999, 1e-8)
	w := mat.NewDense(1, 1, []float64{1.0})
	g := mat.NewDense(1, 1, []float64{0.1})
	a.Step(w, g)
	a.ZeroGrad()
	s.Equal(0, a.T)
	s.Empty(a.m)
	s.Empty(a.v)
	s.Empty(a.mb)
	s.Empty(a.vb)
}

func (s *AdamSuite) TestStepBiasUsesSeparateMoments() {
	// Verify weight and bias moments are tracked independently:
	// Step+StepBias on one Adam must equal Step alone on a fresh Adam (for weights)
	// and StepBias alone on another fresh Adam (for biases).
	aShared := NewAdam(0.01, 0.9, 0.999, 1e-8)
	aW := NewAdam(0.01, 0.9, 0.999, 1e-8)
	aB := NewAdam(0.01, 0.9, 0.999, 1e-8)
	w := mat.NewDense(1, 1, []float64{2.0})
	b := mat.NewDense(1, 1, []float64{3.0})
	g := mat.NewDense(1, 1, []float64{0.5})

	// Call both on shared instance.
	wShared := aShared.Step(w, g)
	bShared := aShared.StepBias(b, g)

	// Call each on a dedicated instance (T=1 for weight, T=0 for bias since StepBias doesn't increment T).
	wAlone := aW.Step(w, g)
	// For bias, StepBias uses whatever T is set on the Adam instance.
	// After aShared.Step, T=1; after aShared.StepBias, T=1 still.
	// aB has T=0, so we need to set T=1 to match shared state before calling StepBias.
	aB.T = 1
	bAlone := aB.StepBias(b, g)

	// Weight result should match: shared and standalone should be identical.
	s.InDelta(wAlone.At(0, 0), wShared.At(0, 0), 1e-10, "weight update independent of bias moments")
	// Bias result should match.
	s.InDelta(bAlone.At(0, 0), bShared.At(0, 0), 1e-10, "bias update independent of weight moments")
}

func (s *AdamSuite) TestStepNilSafe() {
	a := NewAdam(0.01, 0.9, 0.999, 1e-8)
	got := a.Step(nil, nil)
	s.Nil(got)
}

func (s *AdamSuite) TestGetLearningRate() {
	s.InDelta(0.003, NewAdam(0.003, 0.9, 0.999, 1e-8).GetLearningRate(), 1e-10)
}

func (s *AdamSuite) TestDefaultAdam() {
	a := DefaultAdam(0.001)
	s.InDelta(0.9, a.Beta1, 1e-10)
	s.InDelta(0.999, a.Beta2, 1e-10)
	s.InDelta(1e-8, a.Epsilon, 1e-12)
}
