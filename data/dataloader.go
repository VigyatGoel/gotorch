package data

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type DataType int

const (
	Classification DataType = iota
	Regression
)

type DataLoader struct {
	FilePath   string
	DataType   DataType
	Shuffle    bool
	Seed       int64
	SplitRatio float64
	BatchSize  int
	Features   *mat.Dense
	Targets    *mat.Dense
	ClassNames []string
}

type Batch struct {
	Features *mat.Dense
	Targets  *mat.Dense
}

func NewDataLoader(filePath string, dataType DataType, batchSize int) *DataLoader {
	return &DataLoader{
		FilePath:   filePath,
		DataType:   dataType,
		Shuffle:    true,
		Seed:       42,
		SplitRatio: 0.8,
		BatchSize:  batchSize,
		ClassNames: []string{},
	}
}

func (dl *DataLoader) Load() error {
	file, err := os.Open(dl.FilePath)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return fmt.Errorf("error reading records: %w", err)
	}
	if len(records) < 2 {
		return fmt.Errorf("no data found")
	}

	records = records[1:]
	rows := len(records)
	cols := len(records[0])
	featuresData := make([]float64, rows*(cols-1))

	for i, record := range records {
		for j := 0; j < cols-1; j++ {
			val, err := strconv.ParseFloat(strings.TrimSpace(record[j]), 64)
			if err != nil {
				return fmt.Errorf("invalid float at row %d col %d: %w", i, j, err)
			}
			featuresData[i*(cols-1)+j] = val
		}
	}
	dl.Features = mat.NewDense(rows, cols-1, featuresData)

	if dl.DataType == Classification {
		classMap := make(map[string]int)
		for _, record := range records {
			label := strings.TrimSpace(record[cols-1])
			if _, ok := classMap[label]; !ok {
				classMap[label] = len(dl.ClassNames)
				dl.ClassNames = append(dl.ClassNames, label)
			}
		}
		targetsData := make([]float64, rows*len(dl.ClassNames))
		for i, record := range records {
			label := strings.TrimSpace(record[cols-1])
			targetsData[i*len(dl.ClassNames)+classMap[label]] = 1.0
		}
		dl.Targets = mat.NewDense(rows, len(dl.ClassNames), targetsData)
	} else {
		targetsData := make([]float64, rows)
		for i, record := range records {
			val, err := strconv.ParseFloat(strings.TrimSpace(record[cols-1]), 64)
			if err != nil {
				return fmt.Errorf("invalid target at row %d: %w", i, err)
			}
			targetsData[i] = val
		}
		dl.Targets = mat.NewDense(rows, 1, targetsData)
	}

	if dl.Shuffle {
		dl.shuffle()
	}
	return nil
}

func (dl *DataLoader) shuffle() {
	r := rand.New(rand.NewSource(dl.Seed))
	featureData := dl.Features.RawMatrix().Data
	targetData := dl.Targets.RawMatrix().Data
	fRows, fCols := dl.Features.Dims()
	_, tCols := dl.Targets.Dims()

	for i := fRows - 1; i > 0; i-- {
		j := r.Intn(i + 1)
		for k := 0; k < fCols; k++ {
			featureData[i*fCols+k], featureData[j*fCols+k] = featureData[j*fCols+k], featureData[i*fCols+k]
		}
		for k := 0; k < tCols; k++ {
			targetData[i*tCols+k], targetData[j*tCols+k] = targetData[j*tCols+k], targetData[i*tCols+k]
		}
	}
}

func (dl *DataLoader) GetClassNames() []string {
	return dl.ClassNames
}

func (dl *DataLoader) GetBatches(features *mat.Dense, targets *mat.Dense, epoch int) []Batch {
	fRows, fCols := features.Dims()
	_, tCols := targets.Dims()

	if dl.BatchSize <= 0 {
		return []Batch{{Features: features, Targets: targets}}
	}

	indices := make([]int, fRows)
	for i := range indices {
		indices[i] = i
	}
	if dl.Shuffle {
		r := rand.New(rand.NewSource(dl.Seed + int64(epoch)))
		r.Shuffle(fRows, func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
	}

	featureData := features.RawMatrix().Data
	targetData := targets.RawMatrix().Data
	numBatches := int(math.Ceil(float64(fRows) / float64(dl.BatchSize)))
	batches := make([]Batch, 0, numBatches)

	for i := 0; i < fRows; i += dl.BatchSize {
		end := i + dl.BatchSize
		if end > fRows {
			end = fRows
		}
		batchSize := end - i
		batchFeatureData := make([]float64, batchSize*fCols)
		batchTargetData := make([]float64, batchSize*tCols)

		for j := i; j < end; j++ {
			src := indices[j]
			dst := j - i
			copy(batchFeatureData[dst*fCols:(dst+1)*fCols], featureData[src*fCols:(src+1)*fCols])
			copy(batchTargetData[dst*tCols:(dst+1)*tCols], targetData[src*tCols:(src+1)*tCols])
		}
		batches = append(batches, Batch{
			Features: mat.NewDense(batchSize, fCols, batchFeatureData),
			Targets:  mat.NewDense(batchSize, tCols, batchTargetData),
		})
	}
	return batches
}

func (dl *DataLoader) NormalizeFeatures() {
	fRows, fCols := dl.Features.Dims()
	if fRows == 0 {
		return
	}
	data := dl.Features.RawMatrix().Data

	means := make([]float64, fCols)
	for i := 0; i < fRows; i++ {
		for j := 0; j < fCols; j++ {
			means[j] += data[i*fCols+j]
		}
	}
	for j := range means {
		means[j] /= float64(fRows)
	}

	stdDevs := make([]float64, fCols)
	for i := 0; i < fRows; i++ {
		for j := 0; j < fCols; j++ {
			diff := data[i*fCols+j] - means[j]
			stdDevs[j] += diff * diff
		}
	}
	for j := range stdDevs {
		stdDevs[j] = math.Sqrt(stdDevs[j] / float64(fRows))
		if stdDevs[j] == 0 {
			stdDevs[j] = 1
		}
	}

	for i := 0; i < fRows; i++ {
		for j := 0; j < fCols; j++ {
			data[i*fCols+j] = (data[i*fCols+j] - means[j]) / stdDevs[j]
		}
	}
}

func (dl *DataLoader) NumFeatures() int {
	_, cols := dl.Features.Dims()
	return cols
}

func (dl *DataLoader) Split() (trainX, trainY, testX, testY *mat.Dense) {
	fRows, fCols := dl.Features.Dims()
	_, tCols := dl.Targets.Dims()
	if fRows == 0 {
		return nil, nil, nil, nil
	}

	split := int(float64(fRows) * dl.SplitRatio)
	if split <= 0 {
		split = 1
	} else if split >= fRows {
		split = fRows - 1
	}

	featureData := dl.Features.RawMatrix().Data
	targetData := dl.Targets.RawMatrix().Data

	trainFData := make([]float64, split*fCols)
	trainTData := make([]float64, split*tCols)
	testFData := make([]float64, (fRows-split)*fCols)
	testTData := make([]float64, (fRows-split)*tCols)

	copy(trainFData, featureData[:split*fCols])
	copy(trainTData, targetData[:split*tCols])
	copy(testFData, featureData[split*fCols:])
	copy(testTData, targetData[split*tCols:])

	trainX = mat.NewDense(split, fCols, trainFData)
	trainY = mat.NewDense(split, tCols, trainTData)
	testX = mat.NewDense(fRows-split, fCols, testFData)
	testY = mat.NewDense(fRows-split, tCols, testTData)
	return
}
