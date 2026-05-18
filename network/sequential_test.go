package network

import (
	"path/filepath"
	"testing"

	"github.com/VigyatGoel/gotorch/layer"
	"github.com/VigyatGoel/gotorch/optimizer"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
)

type SequentialSuite struct{ suite.Suite }

func TestSequential(t *testing.T) { suite.Run(t, new(SequentialSuite)) }

func (s *SequentialSuite) TestForwardOutputShape() {
	model := NewSequential(
		layer.NewLinear(4, 8),
		layer.NewReLU(),
		layer.NewLinear(8, 2),
	)
	x := mat.NewDense(3, 4, nil)
	for i := range x.RawMatrix().Data {
		x.RawMatrix().Data[i] = 1.0
	}
	out := model.Forward(x)
	r, c := out.Dims()
	s.Equal(3, r)
	s.Equal(2, c)
}

func (s *SequentialSuite) TestBackwardUpdatesWeights() {
	model := NewSequential(layer.NewLinear(2, 2))
	model.SetOptimizer(optimizer.NewSGD(0.1))
	x := mat.NewDense(1, 2, []float64{1, 2})
	initialW := mat.DenseCopyOf(model.Layers[0].GetWeights())
	out := model.Forward(x)
	outR, outC := out.Dims()
	dout := mat.NewDense(outR, outC, nil)
	for i := range dout.RawMatrix().Data {
		dout.RawMatrix().Data[i] = 1
	}
	model.Backward(dout)
	newW := model.Layers[0].GetWeights()
	changed := false
	for i, v := range newW.RawMatrix().Data {
		if v != initialW.RawMatrix().Data[i] {
			changed = true
			break
		}
	}
	s.True(changed, "weights must change after backward+optimizer step")
}

func (s *SequentialSuite) TestTrainEvalSwitchesDropout() {
	// p=1.0 in training drops everything; eval passes through unchanged.
	d := layer.NewDropout(1.0)
	model := NewSequential(d)
	x := mat.NewDense(1, 4, []float64{1, 2, 3, 4})

	model.Train()
	outTrain := model.Forward(x)
	for _, v := range outTrain.RawMatrix().Data {
		s.InDelta(0.0, v, 1e-10, "training with p=1: all zeros")
	}

	model.Eval()
	outEval := model.Forward(x)
	for i, v := range outEval.RawMatrix().Data {
		s.InDelta(x.RawMatrix().Data[i], v, 1e-10, "eval: passthrough")
	}
}

func (s *SequentialSuite) TestPredictMatchesForward() {
	model := NewSequential(layer.NewLinear(2, 2))
	x := mat.NewDense(1, 2, []float64{1, 2})
	fwd := model.Forward(x)
	pred := model.Predict(x)
	for i, v := range pred.RawMatrix().Data {
		s.InDelta(fwd.RawMatrix().Data[i], v, 1e-10)
	}
}

func (s *SequentialSuite) TestAdd() {
	model := NewSequential()
	model.Add(layer.NewLinear(2, 3))
	model.Add(layer.NewReLU())
	s.Len(model.Layers, 2)
	out := model.Forward(mat.NewDense(1, 2, []float64{1, 2}))
	_, c := out.Dims()
	s.Equal(3, c)
}

func (s *SequentialSuite) TestSaveLoadWeightsPreserved() {
	model := NewSequential(layer.NewLinear(3, 2))
	model.Layers[0].UpdateWeights(mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6}))
	path := filepath.Join(s.T().TempDir(), "model.gth")
	require.NoError(s.T(), model.Save(path))

	loaded, err := Load(path)
	require.NoError(s.T(), err)
	s.Len(loaded.Layers, 1)

	want := model.Layers[0].GetWeights().RawMatrix().Data
	got := loaded.Layers[0].GetWeights().RawMatrix().Data
	for i, v := range got {
		s.InDelta(want[i], v, 1e-10)
	}
}
