package layer

import (
	"testing"

	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
)

type DropoutSuite struct{ suite.Suite }

func TestDropout(t *testing.T) { suite.Run(t, new(DropoutSuite)) }

func (s *DropoutSuite) TestForwardTrainingDropsSome() {
	// Use 1000 elements, p=0.5: expect 350-650 zeros (loose bound, no fixed seed).
	d := NewDropout(0.5)
	data := make([]float64, 1000)
	for i := range data {
		data[i] = 1.0
	}
	out := d.Forward(mat.NewDense(1, 1000, data))
	zeros := 0
	for _, v := range out.RawMatrix().Data {
		if v == 0 {
			zeros++
		}
	}
	s.GreaterOrEqual(zeros, 350, "too few zeros for p=0.5")
	s.LessOrEqual(zeros, 650, "too many zeros for p=0.5")
}

func (s *DropoutSuite) TestForwardTrainingScalesSurvivors() {
	// Survivors must be scaled by 1/(1-p). p=0.5 → scale=2.
	d := NewDropout(0.5)
	data := make([]float64, 100)
	for i := range data {
		data[i] = 1.0
	}
	out := d.Forward(mat.NewDense(1, 100, data))
	for _, v := range out.RawMatrix().Data {
		if v != 0.0 {
			s.InDelta(2.0, v, 1e-10, "non-zero element should equal 1/(1-0.5) = 2")
		}
	}
}

func (s *DropoutSuite) TestForwardEvalPassthrough() {
	// p=0.9 in eval: all elements must pass through unchanged.
	d := NewDropout(0.9)
	d.SetTraining(false)
	data := []float64{1, 2, 3, 4, 5}
	out := d.Forward(mat.NewDense(1, 5, data))
	for i, v := range out.RawMatrix().Data {
		s.InDelta(data[i], v, 1e-10)
	}
}

func (s *DropoutSuite) TestForwardZeroRatePassthrough() {
	// p=0: no dropout, all pass through with no scaling.
	d := NewDropout(0.0)
	data := []float64{1, 2, 3}
	out := d.Forward(mat.NewDense(1, 3, data))
	for i, v := range out.RawMatrix().Data {
		s.InDelta(data[i], v, 1e-10)
	}
}

func (s *DropoutSuite) TestSetTraining() {
	d := NewDropout(0.5)
	s.True(d.training)
	d.SetTraining(false)
	s.False(d.training)
	d.SetTraining(true)
	s.True(d.training)
}

func (s *DropoutSuite) TestBackwardAppliesMask() {
	// p=0: mask = all 1.0 (scale=1), backward is identity
	d := NewDropout(0.0)
	x := mat.NewDense(1, 3, []float64{1, 2, 3})
	d.Forward(x)
	dout := mat.NewDense(1, 3, []float64{4, 5, 6})
	grad := d.Backward(dout)
	for i, v := range grad.RawMatrix().Data {
		s.InDelta(dout.RawMatrix().Data[i], v, 1e-10)
	}
}

func (s *DropoutSuite) TestBackwardNilMaskPassthrough() {
	// No Forward call: mask is nil → Backward returns gradOutput unchanged.
	d := NewDropout(0.5)
	dout := mat.NewDense(1, 3, []float64{1, 2, 3})
	grad := d.Backward(dout)
	for i, v := range grad.RawMatrix().Data {
		s.InDelta(dout.RawMatrix().Data[i], v, 1e-10)
	}
}

func (s *DropoutSuite) TestClearCache() {
	d := NewDropout(0.5)
	d.Forward(mat.NewDense(1, 4, []float64{1, 2, 3, 4}))
	s.NotNil(d.mask)
	d.ClearCache()
	s.Nil(d.mask)
}

func (s *DropoutSuite) TestGetWeightsNil() { s.Nil(NewDropout(0.5).GetWeights()) }
func (s *DropoutSuite) TestGetBiasesNil()  { s.Nil(NewDropout(0.5).GetBiases()) }
