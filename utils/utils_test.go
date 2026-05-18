package utils_test

import (
	"math"
	"testing"

	"github.com/VigyatGoel/gotorch/utils"
	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
)

type UtilsSuite struct{ suite.Suite }

func TestUtils(t *testing.T) { suite.Run(t, new(UtilsSuite)) }

func (s *UtilsSuite) TestZeros() {
	m := utils.Zeros(2, 3)
	r, c := m.Dims()
	s.Equal(2, r)
	s.Equal(3, c)
	for _, v := range m.RawMatrix().Data {
		s.InDelta(0.0, v, 1e-10)
	}
}

func (s *UtilsSuite) TestOnes() {
	m := utils.Ones(2, 3)
	r, c := m.Dims()
	s.Equal(2, r)
	s.Equal(3, c)
	for _, v := range m.RawMatrix().Data {
		s.InDelta(1.0, v, 1e-10)
	}
}

func (s *UtilsSuite) TestDot() {
	// [[1,2],[3,4]] * I = [[1,2],[3,4]]
	a := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	b := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	got := utils.Dot(a, b)
	s.InDelta(1.0, got.At(0, 0), 1e-10)
	s.InDelta(2.0, got.At(0, 1), 1e-10)
	s.InDelta(3.0, got.At(1, 0), 1e-10)
	s.InDelta(4.0, got.At(1, 1), 1e-10)
}

func (s *UtilsSuite) TestAdd() {
	a := mat.NewDense(1, 2, []float64{1, 2})
	b := mat.NewDense(1, 2, []float64{3, 4})
	got := utils.Add(a, b)
	s.InDelta(4.0, got.At(0, 0), 1e-10)
	s.InDelta(6.0, got.At(0, 1), 1e-10)
}

func (s *UtilsSuite) TestTranspose() {
	a := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	got := utils.Transpose(a)
	r, c := got.Dims()
	s.Equal(3, r)
	s.Equal(2, c)
	s.InDelta(1.0, got.At(0, 0), 1e-10)
	s.InDelta(4.0, got.At(0, 1), 1e-10)
	s.InDelta(2.0, got.At(1, 0), 1e-10)
}

func (s *UtilsSuite) TestApplyFunc() {
	a := mat.NewDense(1, 3, []float64{1, 4, 9})
	got := utils.ApplyFunc(a, math.Sqrt)
	s.InDelta(1.0, got.At(0, 0), 1e-10)
	s.InDelta(2.0, got.At(0, 1), 1e-10)
	s.InDelta(3.0, got.At(0, 2), 1e-10)
}

func (s *UtilsSuite) TestSoftmaxRowSumTo1() {
	got := utils.Softmax(mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6}))
	rows, cols := got.Dims()
	for i := 0; i < rows; i++ {
		var rowSum float64
		for j := 0; j < cols; j++ {
			rowSum += got.At(i, j)
		}
		s.InDelta(1.0, rowSum, 1e-10)
	}
}

func (s *UtilsSuite) TestGetMaxIndexRowMiddle() {
	m := mat.NewDense(1, 3, []float64{0.1, 0.8, 0.1})
	s.Equal(1, utils.GetMaxIndexRow(m, 0))
}

func (s *UtilsSuite) TestGetMaxIndexRowFirst() {
	m := mat.NewDense(1, 3, []float64{0.9, 0.05, 0.05})
	s.Equal(0, utils.GetMaxIndexRow(m, 0))
}

func (s *UtilsSuite) TestGetMaxIndexRowLast() {
	m := mat.NewDense(1, 3, []float64{0.1, 0.1, 0.8})
	s.Equal(2, utils.GetMaxIndexRow(m, 0))
}
