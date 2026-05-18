package layer

import (
	"math"
	"testing"

	"github.com/VigyatGoel/gotorch/internal/testutil"
	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
)

type SigmoidSuite struct{ suite.Suite }

func TestSigmoid(t *testing.T) { suite.Run(t, new(SigmoidSuite)) }

func (s *SigmoidSuite) TestForwardAt0() {
	sig := NewSigmoid()
	out := sig.Forward(mat.NewDense(1, 1, []float64{0}))
	s.InDelta(0.5, out.At(0, 0), 1e-10)
}

func (s *SigmoidSuite) TestForwardLargePositive() {
	sig := NewSigmoid()
	out := sig.Forward(mat.NewDense(1, 1, []float64{100}))
	s.InDelta(1.0, out.At(0, 0), 1e-6)
}

func (s *SigmoidSuite) TestForwardLargeNegative() {
	sig := NewSigmoid()
	out := sig.Forward(mat.NewDense(1, 1, []float64{-100}))
	s.InDelta(0.0, out.At(0, 0), 1e-6)
}

func (s *SigmoidSuite) TestForwardKnownValue() {
	sig := NewSigmoid()
	x := 2.0
	want := 1.0 / (1.0 + math.Exp(-x))
	out := sig.Forward(mat.NewDense(1, 1, []float64{x}))
	s.InDelta(want, out.At(0, 0), 1e-10)
}

func (s *SigmoidSuite) TestBackwardFormula() {
	// Backward: s*(1-s)*grad where s = sigmoid(x)
	sig := NewSigmoid()
	x := 1.0
	sVal := 1.0 / (1.0 + math.Exp(-x))
	sig.Forward(mat.NewDense(1, 1, []float64{x}))
	grad := sig.Backward(mat.NewDense(1, 1, []float64{1.0}))
	s.InDelta(sVal*(1-sVal), grad.At(0, 0), 1e-10)
}

func (s *SigmoidSuite) TestClearCache() {
	sig := NewSigmoid()
	sig.Forward(mat.NewDense(1, 1, []float64{1}))
	s.NotNil(sig.output)
	sig.ClearCache()
	s.Nil(sig.output)
}

func (s *SigmoidSuite) TestGetWeightsNil() { s.Nil(NewSigmoid().GetWeights()) }
func (s *SigmoidSuite) TestGetBiasesNil()  { s.Nil(NewSigmoid().GetBiases()) }

func (s *SigmoidSuite) TestGradCheck() {
	sig := NewSigmoid()
	testutil.GradCheck(s.T(), sig, mat.NewDense(2, 3, []float64{-1.5, 0.5, 2.0, -0.3, 1.0, -2.0}))
}
