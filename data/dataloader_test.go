package data_test

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/VigyatGoel/gotorch/data"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

func writeCSV(t *testing.T, content string) string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "test*.csv")
	require.NoError(t, err)
	_, err = fmt.Fprint(f, content)
	require.NoError(t, err)
	require.NoError(t, f.Close())
	return f.Name()
}

type DataLoaderSuite struct{ suite.Suite }

func TestDataLoader(t *testing.T) { suite.Run(t, new(DataLoaderSuite)) }

func (s *DataLoaderSuite) TestLoadClassification() {
	path := writeCSV(s.T(), "f1,f2,label\n1.0,2.0,cat\n3.0,4.0,dog\n5.0,6.0,cat\n")
	dl := data.NewDataLoader(path, data.Classification, 0)
	dl.Shuffle = false
	require.NoError(s.T(), dl.Load())

	fr, fc := dl.Features.Dims()
	s.Equal(3, fr)
	s.Equal(2, fc)

	tr, tc := dl.Targets.Dims()
	s.Equal(3, tr)
	s.Equal(2, tc) // 2 classes: cat, dog

	s.Len(dl.ClassNames, 2)
}

func (s *DataLoaderSuite) TestLoadRegression() {
	path := writeCSV(s.T(), "f1,f2,target\n1.0,2.0,3.0\n4.0,5.0,9.0\n7.0,8.0,15.0\n")
	dl := data.NewDataLoader(path, data.Regression, 0)
	dl.Shuffle = false
	require.NoError(s.T(), dl.Load())

	tr, tc := dl.Targets.Dims()
	s.Equal(3, tr)
	s.Equal(1, tc)
}

func (s *DataLoaderSuite) TestLoadMissingFileError() {
	dl := data.NewDataLoader("/nonexistent.csv", data.Regression, 0)
	s.Error(dl.Load())
}

func (s *DataLoaderSuite) TestSplit() {
	csv := "f1,label\n"
	for i := 0; i < 10; i++ {
		csv += fmt.Sprintf("%.1f,a\n", float64(i))
	}
	path := writeCSV(s.T(), csv)
	dl := data.NewDataLoader(path, data.Classification, 0)
	dl.Shuffle = false
	dl.SplitRatio = 0.8
	require.NoError(s.T(), dl.Load())

	trainX, _, testX, _ := dl.Split()
	tR, _ := trainX.Dims()
	eR, _ := testX.Dims()
	s.Equal(8, tR)
	s.Equal(2, eR)
}

func (s *DataLoaderSuite) TestGetBatchesCoversAllRows() {
	csv := "f1,label\n"
	for i := 0; i < 10; i++ {
		csv += fmt.Sprintf("%.1f,a\n", float64(i))
	}
	path := writeCSV(s.T(), csv)
	dl := data.NewDataLoader(path, data.Classification, 3)
	dl.Shuffle = false
	require.NoError(s.T(), dl.Load())

	batches := dl.GetBatches(dl.Features, dl.Targets, 0)
	total := 0
	for _, b := range batches {
		r, _ := b.Features.Dims()
		total += r
	}
	s.Equal(10, total)
}

func (s *DataLoaderSuite) TestGetBatchesLastBatchSmaller() {
	csv := "f1,label\n"
	for i := 0; i < 7; i++ {
		csv += fmt.Sprintf("%.1f,a\n", float64(i))
	}
	path := writeCSV(s.T(), csv)
	dl := data.NewDataLoader(path, data.Classification, 3)
	dl.Shuffle = false
	require.NoError(s.T(), dl.Load())

	batches := dl.GetBatches(dl.Features, dl.Targets, 0)
	s.Len(batches, 3) // 3 + 3 + 1
	lastR, _ := batches[len(batches)-1].Features.Dims()
	s.Equal(1, lastR)
}

func (s *DataLoaderSuite) TestNormalizeFeaturesMeanZeroStdOne() {
	csv := "f1,f2,label\n"
	for i := 0; i < 20; i++ {
		csv += fmt.Sprintf("%.1f,%.1f,a\n", float64(i), float64(i*2))
	}
	path := writeCSV(s.T(), csv)
	dl := data.NewDataLoader(path, data.Classification, 0)
	dl.Shuffle = false
	require.NoError(s.T(), dl.Load())
	dl.NormalizeFeatures()

	rows, cols := dl.Features.Dims()
	raw := dl.Features.RawMatrix().Data
	for j := 0; j < cols; j++ {
		var mean float64
		for i := 0; i < rows; i++ {
			mean += raw[i*cols+j]
		}
		mean /= float64(rows)
		var variance float64
		for i := 0; i < rows; i++ {
			d := raw[i*cols+j] - mean
			variance += d * d
		}
		variance /= float64(rows)
		s.InDelta(0.0, mean, 1e-10, "col %d mean", j)
		s.InDelta(1.0, math.Sqrt(variance), 1e-10, "col %d std", j)
	}
}

func (s *DataLoaderSuite) TestShuffleDeterministic() {
	csv := "f1,label\n"
	for i := 0; i < 10; i++ {
		csv += fmt.Sprintf("%.1f,a\n", float64(i))
	}
	path := writeCSV(s.T(), csv)

	dl1 := data.NewDataLoader(path, data.Classification, 0)
	dl1.Seed = 99
	require.NoError(s.T(), dl1.Load())

	dl2 := data.NewDataLoader(path, data.Classification, 0)
	dl2.Seed = 99
	require.NoError(s.T(), dl2.Load())

	for i, v := range dl1.Features.RawMatrix().Data {
		s.InDelta(v, dl2.Features.RawMatrix().Data[i], 1e-10)
	}
}

func (s *DataLoaderSuite) TestShuffleDifferentSeedsDifferentOrder() {
	// Different seeds must produce different row orderings.
	csv := "f1,label\n"
	for i := 0; i < 20; i++ {
		csv += fmt.Sprintf("%.1f,a\n", float64(i))
	}
	path := writeCSV(s.T(), csv)

	dl1 := data.NewDataLoader(path, data.Classification, 0)
	dl1.Seed = 1
	require.NoError(s.T(), dl1.Load())

	dl2 := data.NewDataLoader(path, data.Classification, 0)
	dl2.Seed = 9999
	require.NoError(s.T(), dl2.Load())

	// At least one element must differ.
	anyDifferent := false
	for i, v := range dl1.Features.RawMatrix().Data {
		if v != dl2.Features.RawMatrix().Data[i] {
			anyDifferent = true
			break
		}
	}
	s.True(anyDifferent, "different seeds should produce different shuffle orders")
}
