package layer

import (
	"math"
	"testing"

	"github.com/VigyatGoel/gotorch/internal/testutil"
	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
)

type SiLUSuite struct{ suite.Suite }

func TestSiLU(t *testing.T) { suite.Run(t, new(SiLUSuite)) }

func (s *SiLUSuite) TestForwardAt0() {
	silu := NewSiLU()
	out := silu.Forward(mat.NewDense(1, 1, []float64{0}))
	s.InDelta(0.0, out.At(0, 0), 1e-10)
}

func (s *SiLUSuite) TestForwardKnownValue() {
	// SiLU(1) = 1 * sigmoid(1) = 1 / (1 + exp(-1))
	silu := NewSiLU()
	x := 1.0
	want := x / (1.0 + math.Exp(-x))
	out := silu.Forward(mat.NewDense(1, 1, []float64{x}))
	s.InDelta(want, out.At(0, 0), 1e-10)
}

func (s *SiLUSuite) TestForwardNegative() {
	// SiLU(-1) = -1 * sigmoid(-1): result is negative
	silu := NewSiLU()
	x := -1.0
	want := x / (1.0 + math.Exp(-x))
	out := silu.Forward(mat.NewDense(1, 1, []float64{x}))
	s.InDelta(want, out.At(0, 0), 1e-10)
}

func (s *SiLUSuite) TestBackwardFormula() {
	// SiLU'(x) = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
	silu := NewSiLU()
	x := 1.0
	sig := 1.0 / (1.0 + math.Exp(-x))
	want := sig + x*sig*(1-sig)
	silu.Forward(mat.NewDense(1, 1, []float64{x}))
	grad := silu.Backward(mat.NewDense(1, 1, []float64{1.0}))
	s.InDelta(want, grad.At(0, 0), 1e-10)
}

func (s *SiLUSuite) TestClearCache() {
	silu := NewSiLU()
	silu.Forward(mat.NewDense(1, 1, []float64{1}))
	s.NotNil(silu.input)
	silu.ClearCache()
	s.Nil(silu.input)
	s.Nil(silu.output)
}

func (s *SiLUSuite) TestGetWeightsNil() { s.Nil(NewSiLU().GetWeights()) }
func (s *SiLUSuite) TestGetBiasesNil()  { s.Nil(NewSiLU().GetBiases()) }

func (s *SiLUSuite) TestGradCheck() {
	silu := NewSiLU()
	testutil.GradCheck(s.T(), silu, mat.NewDense(2, 3, []float64{-1.5, 0.5, 2.0, -0.3, 1.0, -2.0}))
}
