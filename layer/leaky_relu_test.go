package layer

import (
	"testing"

	"github.com/VigyatGoel/gotorch/internal/testutil"
	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
)

type LeakyReLUSuite struct{ suite.Suite }

func TestLeakyReLU(t *testing.T) { suite.Run(t, new(LeakyReLUSuite)) }

func (s *LeakyReLUSuite) TestAlphaField() {
	l := NewLeakyReLU(0.01)
	s.InDelta(0.01, l.Alpha, 1e-10)
}

func (s *LeakyReLUSuite) TestForwardNegativeScaled() {
	// f(-2) = -2 * 0.1 = -0.2
	l := NewLeakyReLU(0.1)
	out := l.Forward(mat.NewDense(1, 3, []float64{-2, 0, 3}))
	testutil.MatClose(s.T(), mat.NewDense(1, 3, []float64{-0.2, 0, 3}), out, 1e-10, "forward")
}

func (s *LeakyReLUSuite) TestForwardPositivePassthrough() {
	l := NewLeakyReLU(0.01)
	out := l.Forward(mat.NewDense(1, 3, []float64{1, 5, 100}))
	testutil.MatClose(s.T(), mat.NewDense(1, 3, []float64{1, 5, 100}), out, 1e-10, "positive passthrough")
}

func (s *LeakyReLUSuite) TestBackwardNegativeGrad() {
	// input=[-3,2], dout=[1,1]: grad=[-3→alpha*1, 2→1] = [0.1, 1]
	l := NewLeakyReLU(0.1)
	l.Forward(mat.NewDense(1, 2, []float64{-3, 2}))
	grad := l.Backward(mat.NewDense(1, 2, []float64{1, 1}))
	testutil.MatClose(s.T(), mat.NewDense(1, 2, []float64{0.1, 1}), grad, 1e-10, "backward")
}

func (s *LeakyReLUSuite) TestClearCache() {
	l := NewLeakyReLU(0.01)
	l.Forward(mat.NewDense(1, 2, []float64{1, -1}))
	s.NotNil(l.input)
	l.ClearCache()
	s.Nil(l.input)
	s.Nil(l.output)
}

func (s *LeakyReLUSuite) TestGetWeightsNil() { s.Nil(NewLeakyReLU(0.01).GetWeights()) }
func (s *LeakyReLUSuite) TestGetBiasesNil()  { s.Nil(NewLeakyReLU(0.01).GetBiases()) }

func (s *LeakyReLUSuite) TestGradCheck() {
	l := NewLeakyReLU(0.1)
	testutil.GradCheck(s.T(), l, mat.NewDense(2, 3, []float64{0.5, -0.7, 1.2, -0.3, 0.8, -1.5}))
}
