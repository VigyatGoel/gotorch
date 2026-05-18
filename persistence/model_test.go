package persistence_test

import (
	"path/filepath"
	"testing"

	"github.com/VigyatGoel/gotorch/layer"
	"github.com/VigyatGoel/gotorch/optimizer"
	"github.com/VigyatGoel/gotorch/persistence"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
)

// testModel is a minimal ModelInterface implementation for tests.
type testModel struct {
	layers []layer.Layer
	opt    optimizer.Optimizer
}

func (m *testModel) GetLayers() []layer.Layer          { return m.layers }
func (m *testModel) GetOptimizer() optimizer.Optimizer { return m.opt }

type PersistenceSuite struct{ suite.Suite }

func TestPersistence(t *testing.T) { suite.Run(t, new(PersistenceSuite)) }

func (s *PersistenceSuite) TestLinearWeightsRoundtrip() {
	l := layer.NewLinear(3, 2)
	l.UpdateWeights(mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6}))
	l.UpdateBiases(mat.NewDense(1, 2, []float64{7, 8}))

	path := filepath.Join(s.T().TempDir(), "model.gth")
	require.NoError(s.T(), persistence.SaveModel(&testModel{layers: []layer.Layer{l}}, path))

	loaded, err := persistence.LoadModelData(path)
	require.NoError(s.T(), err)
	require.Len(s.T(), loaded.Layers, 1)

	ll, ok := loaded.Layers[0].(*layer.Linear)
	require.True(s.T(), ok)
	for i, v := range ll.GetWeights().RawMatrix().Data {
		s.InDelta(l.GetWeights().RawMatrix().Data[i], v, 1e-10)
	}
	for i, v := range ll.GetBiases().RawMatrix().Data {
		s.InDelta(l.GetBiases().RawMatrix().Data[i], v, 1e-10)
	}
}

func (s *PersistenceSuite) TestAllLayerTypesRoundtrip() {
	layers := []layer.Layer{
		layer.NewLinear(2, 3),
		layer.NewReLU(),
		layer.NewLeakyReLU(0.01),
		layer.NewSigmoid(),
		layer.NewSoftmax(),
		layer.NewSiLU(),
		layer.NewFlatten(),
		layer.NewDropout(0.3),
	}
	path := filepath.Join(s.T().TempDir(), "all.gth")
	require.NoError(s.T(), persistence.SaveModel(&testModel{layers: layers}, path))

	loaded, err := persistence.LoadModelData(path)
	require.NoError(s.T(), err)
	require.Len(s.T(), loaded.Layers, 8)

	_, ok0 := loaded.Layers[0].(*layer.Linear)
	_, ok1 := loaded.Layers[1].(*layer.ReLU)
	_, ok2 := loaded.Layers[2].(*layer.LeakyReLU)
	_, ok3 := loaded.Layers[3].(*layer.Sigmoid)
	_, ok4 := loaded.Layers[4].(*layer.Softmax)
	_, ok5 := loaded.Layers[5].(*layer.SiLU)
	_, ok6 := loaded.Layers[6].(*layer.Flatten)
	d, ok7 := loaded.Layers[7].(*layer.Dropout)

	s.True(ok0)
	s.True(ok1)
	s.True(ok2)
	s.True(ok3)
	s.True(ok4)
	s.True(ok5)
	s.True(ok6)
	s.True(ok7)
	s.InDelta(0.3, d.DropoutRate, 1e-10)
}

func (s *PersistenceSuite) TestLeakyReLUAlphaPreserved() {
	l := layer.NewLeakyReLU(0.05)
	path := filepath.Join(s.T().TempDir(), "lrelu.gth")
	require.NoError(s.T(), persistence.SaveModel(&testModel{layers: []layer.Layer{l}}, path))
	loaded, err := persistence.LoadModelData(path)
	require.NoError(s.T(), err)
	ll := loaded.Layers[0].(*layer.LeakyReLU)
	s.InDelta(0.05, ll.Alpha, 1e-10)
}

func (s *PersistenceSuite) TestWrongExtensionError() {
	path := filepath.Join(s.T().TempDir(), "model.json")
	err := persistence.SaveModel(&testModel{layers: []layer.Layer{}}, path)
	s.Error(err)
}

func (s *PersistenceSuite) TestMissingFileError() {
	_, err := persistence.LoadModelData("/nonexistent/path/model.gth")
	s.Error(err)
}

func (s *PersistenceSuite) TestWithAdamOptimizer() {
	m := &testModel{
		layers: []layer.Layer{layer.NewLinear(2, 2)},
		opt:    optimizer.DefaultAdam(0.001),
	}
	path := filepath.Join(s.T().TempDir(), "adam.gth")
	require.NoError(s.T(), persistence.SaveModel(m, path))
	loaded, err := persistence.LoadModelData(path)
	require.NoError(s.T(), err)
	_, ok := loaded.Optimizer.(*optimizer.Adam)
	s.True(ok)
	s.InDelta(0.001, loaded.Optimizer.GetLearningRate(), 1e-10)
}

func (s *PersistenceSuite) TestNoOptimizerRoundtrip() {
	m := &testModel{layers: []layer.Layer{layer.NewReLU()}}
	path := filepath.Join(s.T().TempDir(), "noopt.gth")
	require.NoError(s.T(), persistence.SaveModel(m, path))
	loaded, err := persistence.LoadModelData(path)
	require.NoError(s.T(), err)
	s.Nil(loaded.Optimizer)
}
