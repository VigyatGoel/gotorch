package data

import (
	"encoding/csv"
	"fmt"
	"io"
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
	FilePath    string
	DataType    DataType
	Shuffle     bool
	Seed        int64
	SplitRatio  float64
	BatchSize   int
	Features    *mat.Dense
	Targets     *mat.Dense
	ClassNames  []string
	ColumnNames []string
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

	header, err := reader.Read()
	if err != nil {
		return fmt.Errorf("error reading header: %w", err)
	}
	dl.ColumnNames = header

	var records [][]string
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("error reading record: %w", err)
		}
		records = append(records, record)
	}

	if len(records) == 0 {
		return fmt.Errorf("no data found")
	}

	cols := len(records[0])
	rows := len(records)
	featuresData := make([]float64, rows*(cols-1))
	targetsData := make([]float64, rows)

	for i, record := range records {
		for j := 0; j < cols-1; j++ {
			val, err := strconv.ParseFloat(strings.TrimSpace(record[j]), 64)
			if err != nil {
				return fmt.Errorf("invalid float at row %d col %d value '%s': %w", i, j, record[j], err)
			}
			featuresData[i*(cols-1)+j] = val
		}
	}

	if dl.DataType == Classification {
		classMap := make(map[string]int)
		for _, record := range records {
			label := strings.TrimSpace(record[cols-1])
			if _, ok := classMap[label]; !ok {
				classMap[label] = len(dl.ClassNames)
				dl.ClassNames = append(dl.ClassNames, label)
			}
		}

		targetsData = make([]float64, rows*len(dl.ClassNames))
		for i, record := range records {
			label := strings.TrimSpace(record[cols-1])
			classIndex := classMap[label]
			targetsData[i*len(dl.ClassNames)+classIndex] = 1.0
		}
		dl.Targets = mat.NewDense(rows, len(dl.ClassNames), targetsData)
	} else {
		for i, record := range records {
			val, err := strconv.ParseFloat(strings.TrimSpace(record[cols-1]), 64)
			if err != nil {
				return fmt.Errorf("invalid target at row %d value '%s': %w", i, record[cols-1], err)
			}
			targetsData[i] = val
		}
		dl.Targets = mat.NewDense(rows, 1, targetsData)
	}

	dl.Features = mat.NewDense(rows, cols-1, featuresData)

	if dl.Shuffle {
		dl.shuffle()
	}

	if dl.SplitRatio <= 0 || dl.SplitRatio >= 1 {
		dl.SplitRatio = 0.8
	}

	return nil
}

func (dl *DataLoader) shuffle() {
	r := rand.New(rand.NewSource(dl.Seed))
	rows, cols := dl.Features.Dims()
	_, targetCols := dl.Targets.Dims()

	for i := rows - 1; i > 0; i-- {
		j := r.Intn(i + 1)
		for k := 0; k < cols; k++ {
			dl.Features.Set(i, k, dl.Features.At(j, k))
			dl.Features.Set(j, k, dl.Features.At(i, k))
		}
		for k := 0; k < targetCols; k++ {
			dl.Targets.Set(i, k, dl.Targets.At(j, k))
			dl.Targets.Set(j, k, dl.Targets.At(i, k))
		}
	}
}

func (dl *DataLoader) GetClassNames() []string {
	return dl.ClassNames
}

func (dl *DataLoader) GetColumnNames() []string {
	return dl.ColumnNames
}

func (dl *DataLoader) GetAll() (*mat.Dense, *mat.Dense) {
	return dl.Features, dl.Targets
}

func (dl *DataLoader) GetBatches(features *mat.Dense, targets *mat.Dense, epoch int) []Batch {
	rows, _ := features.Dims()
	if dl.BatchSize <= 0 {
		return []Batch{{Features: features, Targets: targets}}
	}

	indices := make([]int, rows)
	for i := 0; i < rows; i++ {
		indices[i] = i
	}

	if dl.Shuffle {
		epochSeed := dl.Seed + int64(epoch)
		r := rand.New(rand.NewSource(epochSeed))
		r.Shuffle(rows, func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}

	numBatches := int(math.Ceil(float64(rows) / float64(dl.BatchSize)))
	batches := make([]Batch, 0, numBatches)

	for i := 0; i < rows; i += dl.BatchSize {
		end := i + dl.BatchSize
		if end > rows {
			end = rows
		}

		batchFeatures := mat.NewDense(end-i, features.RawMatrix().Cols, nil)
		batchTargets := mat.NewDense(end-i, targets.RawMatrix().Cols, nil)

		for j := i; j < end; j++ {
			for k := 0; k < features.RawMatrix().Cols; k++ {
				batchFeatures.Set(j-i, k, features.At(indices[j], k))
			}
			for k := 0; k < targets.RawMatrix().Cols; k++ {
				batchTargets.Set(j-i, k, targets.At(indices[j], k))
			}
		}
		batches = append(batches, Batch{Features: batchFeatures, Targets: batchTargets})
	}

	return batches
}

func (dl *DataLoader) NumFeatures() int {
	rows, cols := dl.Features.Dims()
	if rows == 0 || cols == 0 {
		if len(dl.ColumnNames) > 1 {
			return len(dl.ColumnNames) - 1
		}
		return 0
	}
	return cols
}

func (dl *DataLoader) Split() (trainX, trainY, testX, testY *mat.Dense) {
	rows, cols := dl.Features.Dims()
	if rows == 0 {
		return nil, nil, nil, nil
	}

	split := int(float64(rows) * dl.SplitRatio)
	if split <= 0 {
		split = 1
	} else if split >= rows {
		split = rows - 1
	}

	trainX = mat.NewDense(split, cols, nil)
	trainY = mat.NewDense(split, dl.Targets.RawMatrix().Cols, nil)
	testX = mat.NewDense(rows-split, cols, nil)
	testY = mat.NewDense(rows-split, dl.Targets.RawMatrix().Cols, nil)

	for i := 0; i < split; i++ {
		for j := 0; j < cols; j++ {
			trainX.Set(i, j, dl.Features.At(i, j))
		}
		for j := 0; j < dl.Targets.RawMatrix().Cols; j++ {
			trainY.Set(i, j, dl.Targets.At(i, j))
		}
	}

	for i := split; i < rows; i++ {
		for j := 0; j < cols; j++ {
			testX.Set(i-split, j, dl.Features.At(i, j))
		}
		for j := 0; j < dl.Targets.RawMatrix().Cols; j++ {
			testY.Set(i-split, j, dl.Targets.At(i, j))
		}
	}

	return
}

func (dl *DataLoader) NormalizeFeatures() {
	rows, cols := dl.Features.Dims()
	if rows == 0 {
		return
	}

	means := make([]float64, cols)
	stdDevs := make([]float64, cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			means[j] += dl.Features.At(i, j)
		}
	}
	for j := 0; j < cols; j++ {
		means[j] /= float64(rows)
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			stdDevs[j] += math.Pow(dl.Features.At(i, j)-means[j], 2)
		}
	}
	for j := 0; j < cols; j++ {
		stdDevs[j] = math.Sqrt(stdDevs[j] / float64(rows))
		if stdDevs[j] == 0 {
			stdDevs[j] = 1
		}
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			dl.Features.Set(i, j, (dl.Features.At(i, j)-means[j])/stdDevs[j])
		}
	}
}
