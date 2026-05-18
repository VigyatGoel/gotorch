package layer

import (
	"testing"

	"github.com/VigyatGoel/gotorch/internal/testutil"
	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
)

type FlattenSuite struct{ suite.Suite }

func TestFlatten(t *testing.T) { suite.Run(t, new(FlattenSuite)) }

func (s *FlattenSuite) TestForwardShapePreserved() {
	f := NewFlatten()
	out := f.Forward(mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6}))
	r, c := out.Dims()
	s.Equal(2, r)
	s.Equal(3, c)
}

func (s *FlattenSuite) TestForwardDataCopied() {
	f := NewFlatten()
	x := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	out := f.Forward(x)
	testutil.MatClose(s.T(), x, out, 1e-10, "data preserved")
}

func (s *FlattenSuite) TestForwardIndependentCopy() {
	// Mutating the original input must not affect the output.
	f := NewFlatten()
	x := mat.NewDense(1, 2, []float64{1, 2})
	out := f.Forward(x)
	x.RawMatrix().Data[0] = 99
	s.InDelta(1.0, out.At(0, 0), 1e-10)
}

func (s *FlattenSuite) TestBackwardShapeRestored() {
	f := NewFlatten()
	f.Forward(mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6}))
	grad := f.Backward(mat.NewDense(2, 3, []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}))
	r, c := grad.Dims()
	s.Equal(2, r)
	s.Equal(3, c)
}

func (s *FlattenSuite) TestClearCache() {
	f := NewFlatten()
	f.Forward(mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6}))
	s.Equal(2, f.inputRows)
	f.ClearCache()
	s.Equal(0, f.inputRows)
	s.Equal(0, f.inputCols)
}

func (s *FlattenSuite) TestGetWeightsNil() { s.Nil(NewFlatten().GetWeights()) }
func (s *FlattenSuite) TestGetBiasesNil()  { s.Nil(NewFlatten().GetBiases()) }
