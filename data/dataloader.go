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
	Features    [][]float64
	Targets     [][]float64
	ClassNames  []string
	ColumnNames []string
}

type Batch struct {
	Features [][]float64
	Targets  [][]float64
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
	dl.Features = make([][]float64, len(records))
	for i, record := range records {
		dl.Features[i] = make([]float64, cols-1)
		for j := 0; j < cols-1; j++ {
			val, err := strconv.ParseFloat(strings.TrimSpace(record[j]), 64)
			if err != nil {
				return fmt.Errorf("invalid float at row %d col %d value '%s': %w", i, j, record[j], err)
			}
			dl.Features[i][j] = val
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

		dl.Targets = make([][]float64, len(records))
		for i, record := range records {
			label := strings.TrimSpace(record[cols-1])
			classIndex := classMap[label]
			oneHot := make([]float64, len(dl.ClassNames))
			oneHot[classIndex] = 1.0
			dl.Targets[i] = oneHot
		}
	} else {
		dl.Targets = make([][]float64, len(records))
		for i, record := range records {
			val, err := strconv.ParseFloat(strings.TrimSpace(record[cols-1]), 64)
			if err != nil {
				return fmt.Errorf("invalid target at row %d value '%s': %w", i, record[cols-1], err)
			}
			dl.Targets[i] = []float64{val}
		}
	}

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
	n := len(dl.Features)
	for i := n - 1; i > 0; i-- {
		j := r.Intn(i + 1)
		dl.Features[i], dl.Features[j] = dl.Features[j], dl.Features[i]
		dl.Targets[i], dl.Targets[j] = dl.Targets[j], dl.Targets[i]
	}
}

func (dl *DataLoader) GetClassNames() []string {
	return dl.ClassNames
}

func (dl *DataLoader) GetColumnNames() []string {
	return dl.ColumnNames
}

func (dl *DataLoader) GetAll() ([][]float64, [][]float64) {
	return dl.Features, dl.Targets
}

func (dl *DataLoader) GetBatches(features [][]float64, targets [][]float64, epoch int) []Batch {
	if dl.BatchSize <= 0 {
		return []Batch{{Features: features, Targets: targets}}
	}

	n := len(features)
	indices := make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = i
	}

	if dl.Shuffle {
		epochSeed := dl.Seed + int64(epoch)
		r := rand.New(rand.NewSource(epochSeed))
		r.Shuffle(n, func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}

	numBatches := int(math.Ceil(float64(n) / float64(dl.BatchSize)))
	batches := make([]Batch, 0, numBatches)

	for i := 0; i < n; i += dl.BatchSize {
		end := i + dl.BatchSize
		if end > n {
			end = n
		}

		batchIndices := indices[i:end]
		batchFeatures := make([][]float64, len(batchIndices))
		batchTargets := make([][]float64, len(batchIndices))

		for j, idx := range batchIndices {
			batchFeatures[j] = features[idx]
			batchTargets[j] = targets[idx]
		}
		batches = append(batches, Batch{Features: batchFeatures, Targets: batchTargets})
	}

	return batches
}

func (dl *DataLoader) NumFeatures() int {
	if len(dl.Features) == 0 || len(dl.Features[0]) == 0 {
		if len(dl.ColumnNames) > 1 {
			return len(dl.ColumnNames) - 1
		}
		return 0
	}
	return len(dl.Features[0])
}

func (dl *DataLoader) Split() (trainX, trainY, testX, testY [][]float64) {
	if len(dl.Features) == 0 {
		return nil, nil, nil, nil
	}

	split := int(float64(len(dl.Features)) * dl.SplitRatio)
	if split <= 0 {
		split = 1
	} else if split >= len(dl.Features) {
		split = len(dl.Features) - 1
	}

	trainX = make([][]float64, split)
	trainY = make([][]float64, split)
	for i := 0; i < split; i++ {
		trainX[i] = make([]float64, len(dl.Features[i]))
		copy(trainX[i], dl.Features[i])

		trainY[i] = make([]float64, len(dl.Targets[i]))
		copy(trainY[i], dl.Targets[i])
	}

	testX = make([][]float64, len(dl.Features)-split)
	testY = make([][]float64, len(dl.Features)-split)
	for i := 0; i < len(dl.Features)-split; i++ {
		testX[i] = make([]float64, len(dl.Features[i+split]))
		copy(testX[i], dl.Features[i+split])

		testY[i] = make([]float64, len(dl.Targets[i+split]))
		copy(testY[i], dl.Targets[i+split])
	}

	return
}

func (dl *DataLoader) NormalizeFeatures() {
	if len(dl.Features) == 0 {
		return
	}

	numFeatures := len(dl.Features[0])
	means := make([]float64, numFeatures)
	stdDevs := make([]float64, numFeatures)

	for _, row := range dl.Features {
		for j := 0; j < numFeatures; j++ {
			means[j] += row[j]
		}
	}
	for j := 0; j < numFeatures; j++ {
		means[j] /= float64(len(dl.Features))
	}

	for _, row := range dl.Features {
		for j := 0; j < numFeatures; j++ {
			stdDevs[j] += math.Pow(row[j]-means[j], 2)
		}
	}
	for j := 0; j < numFeatures; j++ {
		stdDevs[j] = math.Sqrt(stdDevs[j] / float64(len(dl.Features)))
		if stdDevs[j] == 0 {
			stdDevs[j] = 1 
		}
	}

	for i := range dl.Features {
		for j := 0; j < numFeatures; j++ {
			dl.Features[i][j] = (dl.Features[i][j] - means[j]) / stdDevs[j]
		}
	}
}
