package layer

import (
	"testing"

	"github.com/VigyatGoel/gotorch/internal/testutil"
	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
)

type LinearSuite struct{ suite.Suite }

func TestLinear(t *testing.T) { suite.Run(t, new(LinearSuite)) }

func (s *LinearSuite) TestForwardShape() {
	l := NewLinear(3, 4)
	out := l.Forward(mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6}))
	r, c := out.Dims()
	s.Equal(2, r)
	s.Equal(4, c)
}

func (s *LinearSuite) TestForwardValue() {
	// identity weight, zero bias: out == input
	l := NewLinear(2, 2)
	l.UpdateWeights(mat.NewDense(2, 2, []float64{1, 0, 0, 1}))
	l.UpdateBiases(mat.NewDense(1, 2, []float64{0, 0}))
	out := l.Forward(mat.NewDense(1, 2, []float64{2, 3}))
	testutil.MatClose(s.T(), mat.NewDense(1, 2, []float64{2, 3}), out, 1e-10, "forward value")
}

func (s *LinearSuite) TestForwardBias() {
	l := NewLinear(2, 2)
	l.UpdateWeights(mat.NewDense(2, 2, []float64{1, 0, 0, 1}))
	l.UpdateBiases(mat.NewDense(1, 2, []float64{1, 2}))
	out := l.Forward(mat.NewDense(1, 2, []float64{0, 0}))
	testutil.MatClose(s.T(), mat.NewDense(1, 2, []float64{1, 2}), out, 1e-10, "bias")
}

func (s *LinearSuite) TestForwardBatchBias() {
	// bias must broadcast across all rows of the batch
	l := NewLinear(2, 2)
	l.UpdateWeights(mat.NewDense(2, 2, []float64{1, 0, 0, 1}))
	l.UpdateBiases(mat.NewDense(1, 2, []float64{10, 20}))
	out := l.Forward(mat.NewDense(2, 2, []float64{0, 0, 0, 0}))
	testutil.MatClose(s.T(), mat.NewDense(2, 2, []float64{10, 20, 10, 20}), out, 1e-10, "batch bias")
}

func (s *LinearSuite) TestBackwardInputGradShape() {
	l := NewLinear(3, 4)
	x := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	out := l.Forward(x)
	outR, outC := out.Dims()
	dout := mat.NewDense(outR, outC, nil)
	for i := range dout.RawMatrix().Data {
		dout.RawMatrix().Data[i] = 1
	}
	grad := l.Backward(dout)
	r, c := grad.Dims()
	s.Equal(2, r)
	s.Equal(3, c)
}

func (s *LinearSuite) TestBackwardWeightGrad() {
	// With identity weight: dW = xT * dout = [[2],[3]] * [[1,1]] = [[2,2],[3,3]]
	l := NewLinear(2, 2)
	l.UpdateWeights(mat.NewDense(2, 2, []float64{1, 0, 0, 1}))
	l.UpdateBiases(mat.NewDense(1, 2, []float64{0, 0}))
	l.Forward(mat.NewDense(1, 2, []float64{2, 3}))
	l.Backward(mat.NewDense(1, 2, []float64{1, 1}))
	testutil.MatClose(s.T(), mat.NewDense(2, 2, []float64{2, 2, 3, 3}), l.GetGradients(), 1e-10, "weight grad")
}

func (s *LinearSuite) TestBackwardBiasGrad() {
	// dB = sum over batch dimension = [[1+1, 1+1]] = [[2, 2]]
	l := NewLinear(2, 2)
	l.UpdateWeights(mat.NewDense(2, 2, []float64{1, 0, 0, 1}))
	l.UpdateBiases(mat.NewDense(1, 2, []float64{0, 0}))
	l.Forward(mat.NewDense(2, 2, []float64{1, 2, 3, 4}))
	l.Backward(mat.NewDense(2, 2, []float64{1, 1, 1, 1}))
	testutil.MatClose(s.T(), mat.NewDense(1, 2, []float64{2, 2}), l.GetBiasGradients(), 1e-10, "bias grad")
}

func (s *LinearSuite) TestClearCache() {
	l := NewLinear(2, 2)
	l.Forward(mat.NewDense(1, 2, []float64{1, 2}))
	s.NotNil(l.inputMat)
	l.ClearCache()
	s.Nil(l.inputMat)
}

func (s *LinearSuite) TestGetWeightsNotNil() { s.NotNil(NewLinear(2, 3).GetWeights()) }
func (s *LinearSuite) TestGetBiasesNotNil()  { s.NotNil(NewLinear(2, 3).GetBiases()) }

func (s *LinearSuite) TestUpdateWeights() {
	l := NewLinear(2, 2)
	w := mat.NewDense(2, 2, []float64{9, 8, 7, 6})
	l.UpdateWeights(w)
	testutil.MatClose(s.T(), w, l.GetWeights(), 1e-10, "updated weights")
}

func (s *LinearSuite) TestUpdateBiases() {
	l := NewLinear(2, 2)
	b := mat.NewDense(1, 2, []float64{5, 6})
	l.UpdateBiases(b)
	testutil.MatClose(s.T(), b, l.GetBiases(), 1e-10, "updated biases")
}

func (s *LinearSuite) TestGradCheckInput() {
	l := NewLinear(3, 4)
	testutil.GradCheck(s.T(), l, mat.NewDense(2, 3, []float64{0.1, -0.2, 0.3, -0.4, 0.5, -0.6}))
}

func (s *LinearSuite) TestGradCheckWeights() {
	l := NewLinear(3, 4)
	testutil.GradCheckWeights(s.T(), l, mat.NewDense(2, 3, []float64{0.1, -0.2, 0.3, -0.4, 0.5, -0.6}))
}
