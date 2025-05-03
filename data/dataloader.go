package data

import (
	"encoding/csv"
	"fmt"
	"io"
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
	Features    [][]float64
	Targets     [][]float64
	ClassNames  []string
	ColumnNames []string
}

func NewDataLoader(filePath string, dataType DataType) *DataLoader {
	return &DataLoader{
		FilePath:   filePath,
		DataType:   dataType,
		Shuffle:    true,
		Seed:       42,
		SplitRatio: 0.8,
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
