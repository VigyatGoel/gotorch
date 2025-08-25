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

	"gorgonia.org/tensor"
)

type DataType int

const (
	Classification DataType = iota
	Regression
	ImageClassification // Added for image dataset support
)

type ImageSample struct {
	Path     string
	ClassIdx int
}

type DataLoader struct {
	FilePath          string
	DataType          DataType
	Shuffle           bool
	Seed              int64
	SplitRatio        float64
	BatchSize         int
	Features          *tensor.Dense
	Targets           *tensor.Dense
	ClassNames        []string
	ColumnNames       []string
	ImageDir          string
	ImageSamples      []ImageSample
	ImageWidth        int
	ImageHeight       int
	Grayscale         bool
	trainImageSamples []ImageSample
	testImageSamples  []ImageSample
}

type Batch struct {
	Features *tensor.Dense
	Targets  *tensor.Dense
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
		Grayscale:  true, // Default to grayscale
	}
}

func (dl *DataLoader) Load() error {
	if dl.DataType == ImageClassification {
		if dl.ImageDir == "" {
			return fmt.Errorf("ImageDir must be set for ImageClassification")
		}
		if len(dl.ClassNames) == 0 {
			classNames, err := GetClassNamesFromDir(dl.ImageDir)
			if err != nil {
				return fmt.Errorf("failed to get class names: %w", err)
			}
			dl.ClassNames = classNames
		}
		samples, err := LoadImagePathsAndLabels(dl.ImageDir, dl.ClassNames)
		if err != nil {
			return fmt.Errorf("failed to load image paths: %w", err)
		}
		dl.ImageSamples = samples
		if dl.Shuffle {
			r := rand.New(rand.NewSource(dl.Seed))
			r.Shuffle(len(dl.ImageSamples), func(i, j int) {
				dl.ImageSamples[i], dl.ImageSamples[j] = dl.ImageSamples[j], dl.ImageSamples[i]
			})
		}
		if dl.SplitRatio <= 0 || dl.SplitRatio >= 1 {
			dl.SplitRatio = 0.8
		}

		// Split the samples
		total := len(dl.ImageSamples)
		splitIndex := int(float64(total) * dl.SplitRatio)
		dl.trainImageSamples = dl.ImageSamples[:splitIndex]
		dl.testImageSamples = dl.ImageSamples[splitIndex:]

		return nil
	}

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
		dl.Targets = tensor.New(tensor.WithShape(rows, len(dl.ClassNames)), tensor.WithBacking(targetsData))
	} else {
		for i, record := range records {
			val, err := strconv.ParseFloat(strings.TrimSpace(record[cols-1]), 64)
			if err != nil {
				return fmt.Errorf("invalid target at row %d value '%s': %w", i, record[cols-1], err)
			}
			targetsData[i] = val
		}
		dl.Targets = tensor.New(tensor.WithShape(rows, 1), tensor.WithBacking(targetsData))
	}

	dl.Features = tensor.New(tensor.WithShape(rows, cols-1), tensor.WithBacking(featuresData))

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

	// Get data slices
	featureData := dl.Features.Data().([]float64)
	targetData := dl.Targets.Data().([]float64)

	// Get shapes
	featureShape := dl.Features.Shape()
	targetShape := dl.Targets.Shape()

	rows := featureShape[0]
	featureCols := featureShape[1]
	targetCols := targetShape[1]

	for i := rows - 1; i > 0; i-- {
		j := r.Intn(i + 1)

		// Swap feature rows
		for k := 0; k < featureCols; k++ {
			idxI := i*featureCols + k
			idxJ := j*featureCols + k
			featureData[idxI], featureData[idxJ] = featureData[idxJ], featureData[idxI]
		}

		// Swap target rows
		for k := 0; k < targetCols; k++ {
			idxI := i*targetCols + k
			idxJ := j*targetCols + k
			targetData[idxI], targetData[idxJ] = targetData[idxJ], targetData[idxI]
		}
	}
}

func (dl *DataLoader) GetClassNames() []string {
	return dl.ClassNames
}

func (dl *DataLoader) GetColumnNames() []string {
	return dl.ColumnNames
}

func (dl *DataLoader) GetAll() (*tensor.Dense, *tensor.Dense) {
	return dl.Features, dl.Targets
}

func (dl *DataLoader) GetBatches(features *tensor.Dense, targets *tensor.Dense, epoch int) []Batch {
	shape := features.Shape()
	rows := shape[0]
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

	// Get data slices
	featureData := features.Data().([]float64)
	targetData := targets.Data().([]float64)

	// Get shapes
	featureShape := features.Shape()
	targetShape := targets.Shape()
	featureCols := featureShape[1]
	targetCols := targetShape[1]

	for i := 0; i < rows; i += dl.BatchSize {
		end := i + dl.BatchSize
		if end > rows {
			end = rows
		}

		batchSize := end - i
		batchFeatureData := make([]float64, batchSize*featureCols)
		batchTargetData := make([]float64, batchSize*targetCols)

		for j := i; j < end; j++ {
			srcIdx := indices[j]
			dstIdx := j - i

			// Copy feature data
			for k := 0; k < featureCols; k++ {
				batchFeatureData[dstIdx*featureCols+k] = featureData[srcIdx*featureCols+k]
			}

			// Copy target data
			for k := 0; k < targetCols; k++ {
				batchTargetData[dstIdx*targetCols+k] = targetData[srcIdx*targetCols+k]
			}
		}

		batchFeatures := tensor.New(tensor.WithShape(batchSize, featureCols), tensor.WithBacking(batchFeatureData))
		batchTargets := tensor.New(tensor.WithShape(batchSize, targetCols), tensor.WithBacking(batchTargetData))
		batches = append(batches, Batch{Features: batchFeatures, Targets: batchTargets})
	}

	return batches
}

func (dl *DataLoader) GetTrainImageBatches(epoch int, batchSize int) []Batch {
	return dl.getImageBatches(dl.trainImageSamples, epoch, batchSize)
}

func (dl *DataLoader) GetTestImageBatches(batchSize int) []Batch {
	return dl.getImageBatches(dl.testImageSamples, 0, batchSize)
}

func (dl *DataLoader) getImageBatches(samples []ImageSample, epoch int, batchSize int) []Batch {
	total := len(samples)
	indices := make([]int, total)
	for i := 0; i < total; i++ {
		indices[i] = i
	}
	if dl.Shuffle {
		epochSeed := dl.Seed + int64(epoch)
		r := rand.New(rand.NewSource(epochSeed))
		r.Shuffle(total, func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}
	numBatches := int(math.Ceil(float64(total) / float64(batchSize)))
	batches := make([]Batch, 0, numBatches)
	for i := 0; i < total; i += batchSize {
		end := i + batchSize
		if end > total {
			end = total
		}
		var features []*tensor.Dense
		labels := make([][]float64, 0, end-i)
		for _, idx := range indices[i:end] {
			sample := samples[idx]
			var imgMat *tensor.Dense
			var err error
			if dl.Grayscale {
				imgMat, err = LoadImageGrayscale(sample.Path, dl.ImageWidth, dl.ImageHeight)
			} else {
				imgMat, err = LoadImageRGB(sample.Path, dl.ImageWidth, dl.ImageHeight)
			}
			if err != nil {
				continue
			}
			features = append(features, imgMat)
			label := make([]float64, len(dl.ClassNames))
			label[sample.ClassIdx] = 1.0
			labels = append(labels, label)
		}
		if len(features) == 0 {
			continue
		}

		imgShape := features[0].Shape()
		channels := imgShape[1]
		height := imgShape[2]
		width := imgShape[3]
		currentBatchSize := len(features)

		featuresData := make([]float64, 0, currentBatchSize*channels*height*width)
		for _, f := range features {
			featuresData = append(featuresData, f.Data().([]float64)...)
		}

		batchFeatures := tensor.New(
			tensor.WithShape(currentBatchSize, channels, height, width),
			tensor.WithBacking(featuresData),
		)

		labelsData := make([]float64, 0, len(labels)*len(dl.ClassNames))
		for _, l := range labels {
			labelsData = append(labelsData, l...)
		}
		batchTargets := tensor.New(tensor.WithShape(len(labels), len(dl.ClassNames)), tensor.WithBacking(labelsData))

		batches = append(batches, Batch{Features: batchFeatures, Targets: batchTargets})
	}
	return batches
}

func (dl *DataLoader) NumFeatures() int {
	shape := dl.Features.Shape()
	if len(shape) < 2 {
		if len(dl.ColumnNames) > 1 {
			return len(dl.ColumnNames) - 1
		}
		return 0
	}
	return shape[1]
}

func (dl *DataLoader) Split() (trainX, trainY, testX, testY *tensor.Dense) {
	shape := dl.Features.Shape()
	rows := shape[0]
	if rows == 0 {
		return nil, nil, nil, nil
	}

	cols := shape[1]
	targetShape := dl.Targets.Shape()
	targetCols := targetShape[1]

	split := int(float64(rows) * dl.SplitRatio)
	if split <= 0 {
		split = 1
	} else if split >= rows {
		split = rows - 1
	}

	// Get data slices
	featureData := dl.Features.Data().([]float64)
	targetData := dl.Targets.Data().([]float64)

	// Create training data slices
	trainFeatureData := make([]float64, split*cols)
	trainTargetData := make([]float64, split*targetCols)

	// Create test data slices
	testFeatureData := make([]float64, (rows-split)*cols)
	testTargetData := make([]float64, (rows-split)*targetCols)

	// Copy data
	for i := 0; i < split; i++ {
		// Copy features
		for j := 0; j < cols; j++ {
			trainFeatureData[i*cols+j] = featureData[i*cols+j]
		}
		// Copy targets
		for j := 0; j < targetCols; j++ {
			trainTargetData[i*targetCols+j] = targetData[i*targetCols+j]
		}
	}

	for i := split; i < rows; i++ {
		// Copy features
		for j := 0; j < cols; j++ {
			testFeatureData[(i-split)*cols+j] = featureData[i*cols+j]
		}
		// Copy targets
		for j := 0; j < targetCols; j++ {
			testTargetData[(i-split)*targetCols+j] = targetData[i*targetCols+j]
		}
	}

	trainX = tensor.New(tensor.WithShape(split, cols), tensor.WithBacking(trainFeatureData))
	trainY = tensor.New(tensor.WithShape(split, targetCols), tensor.WithBacking(trainTargetData))
	testX = tensor.New(tensor.WithShape(rows-split, cols), tensor.WithBacking(testFeatureData))
	testY = tensor.New(tensor.WithShape(rows-split, targetCols), tensor.WithBacking(testTargetData))

	return
}

func (dl *DataLoader) NormalizeFeatures() {
	shape := dl.Features.Shape()
	rows := shape[0]
	if rows == 0 {
		return
	}
	cols := shape[1]

	// Get data slice
	featureData := dl.Features.Data().([]float64)

	// Calculate means
	means := make([]float64, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			means[j] += featureData[i*cols+j]
		}
	}
	for j := 0; j < cols; j++ {
		means[j] /= float64(rows)
	}

	// Calculate standard deviations
	stdDevs := make([]float64, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			diff := featureData[i*cols+j] - means[j]
			stdDevs[j] += diff * diff
		}
	}
	for j := 0; j < cols; j++ {
		stdDevs[j] = math.Sqrt(stdDevs[j] / float64(rows))
		if stdDevs[j] == 0 {
			stdDevs[j] = 1
		}
	}

	// Normalize features
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			featureData[i*cols+j] = (featureData[i*cols+j] - means[j]) / stdDevs[j]
		}
	}
}
