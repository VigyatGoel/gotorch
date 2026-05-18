package layer

import (
	"testing"

	"github.com/VigyatGoel/gotorch/internal/testutil"
	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
)

type ReLUSuite struct{ suite.Suite }

func TestReLU(t *testing.T) { suite.Run(t, new(ReLUSuite)) }

func (s *ReLUSuite) TestForwardZerosNegatives() {
	r := NewReLU()
	out := r.Forward(mat.NewDense(1, 4, []float64{-2, -1, 0, 3}))
	testutil.MatClose(s.T(), mat.NewDense(1, 4, []float64{0, 0, 0, 3}), out, 1e-10, "forward")
}

func (s *ReLUSuite) TestForwardPositivePassthrough() {
	r := NewReLU()
	out := r.Forward(mat.NewDense(1, 3, []float64{1, 2.5, 100}))
	testutil.MatClose(s.T(), mat.NewDense(1, 3, []float64{1, 2.5, 100}), out, 1e-10, "positive passthrough")
}

func (s *ReLUSuite) TestBackwardMask() {
	// input=[-1,2,-3,4], dout=[1,1,1,1] → grad=[0,1,0,1]
	r := NewReLU()
	r.Forward(mat.NewDense(1, 4, []float64{-1, 2, -3, 4}))
	grad := r.Backward(mat.NewDense(1, 4, []float64{1, 1, 1, 1}))
	testutil.MatClose(s.T(), mat.NewDense(1, 4, []float64{0, 1, 0, 1}), grad, 1e-10, "backward mask")
}

func (s *ReLUSuite) TestBackwardScalesGrad() {
	r := NewReLU()
	r.Forward(mat.NewDense(1, 2, []float64{1, -1}))
	grad := r.Backward(mat.NewDense(1, 2, []float64{3, 5}))
	testutil.MatClose(s.T(), mat.NewDense(1, 2, []float64{3, 0}), grad, 1e-10, "backward scale")
}

func (s *ReLUSuite) TestClearCache() {
	r := NewReLU()
	r.Forward(mat.NewDense(1, 2, []float64{1, 2}))
	s.NotNil(r.input)
	r.ClearCache()
	s.Nil(r.input)
}

func (s *ReLUSuite) TestGetWeightsNil()   { s.Nil(NewReLU().GetWeights()) }
func (s *ReLUSuite) TestGetBiasesNil()    { s.Nil(NewReLU().GetBiases()) }
func (s *ReLUSuite) TestGetGradientsNil() { s.Nil(NewReLU().GetGradients()) }

func (s *ReLUSuite) TestGradCheck() {
	// Avoid x=0 (non-differentiable point): use values clearly away from 0.
	r := NewReLU()
	testutil.GradCheck(s.T(), r, mat.NewDense(2, 3, []float64{0.5, -0.7, 1.2, -0.3, 0.8, -1.5}))
}
