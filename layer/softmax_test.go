package layer

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
)

type SoftmaxSuite struct{ suite.Suite }

func TestSoftmax(t *testing.T) { suite.Run(t, new(SoftmaxSuite)) }

func (s *SoftmaxSuite) TestForwardRowSumTo1() {
	sm := NewSoftmax()
	out := sm.Forward(mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6}))
	rows, cols := out.Dims()
	for i := 0; i < rows; i++ {
		var rowSum float64
		for j := 0; j < cols; j++ {
			rowSum += out.At(i, j)
		}
		s.InDelta(1.0, rowSum, 1e-10)
	}
}

func (s *SoftmaxSuite) TestForwardPositiveOutputs() {
	sm := NewSoftmax()
	out := sm.Forward(mat.NewDense(1, 3, []float64{1, 2, 3}))
	rows, cols := out.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			assert.Greater(s.T(), out.At(i, j), 0.0)
		}
	}
}

func (s *SoftmaxSuite) TestForwardNumericallyStable() {
	// Large inputs must not produce NaN or Inf (max-subtraction stabilisation).
	sm := NewSoftmax()
	out := sm.Forward(mat.NewDense(1, 3, []float64{1000, 2000, 3000}))
	for _, v := range out.RawMatrix().Data {
		s.False(math.IsNaN(v), "NaN in output")
		s.False(math.IsInf(v, 0), "Inf in output")
	}
}

func (s *SoftmaxSuite) TestBackwardPassthrough() {
	// Softmax.Backward is a passthrough because it is paired with CrossEntropyLoss.
	sm := NewSoftmax()
	sm.Forward(mat.NewDense(1, 3, []float64{1, 2, 3}))
	dout := mat.NewDense(1, 3, []float64{0.1, 0.2, 0.3})
	got := sm.Backward(dout)
	for i, v := range got.RawMatrix().Data {
		s.InDelta(dout.RawMatrix().Data[i], v, 1e-10)
	}
}

func (s *SoftmaxSuite) TestClearCache() {
	sm := NewSoftmax()
	sm.Forward(mat.NewDense(1, 2, []float64{1, 2}))
	s.NotNil(sm.input)
	sm.ClearCache()
	s.Nil(sm.input)
	s.Nil(sm.output)
}

func (s *SoftmaxSuite) TestGetWeightsNil() { s.Nil(NewSoftmax().GetWeights()) }
func (s *SoftmaxSuite) TestGetBiasesNil()  { s.Nil(NewSoftmax().GetBiases()) }
