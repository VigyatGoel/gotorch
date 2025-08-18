package main

// import (
// 	"fmt"
// 	"log"
// 	"math/rand"
// 	"time"

// 	"github.com/VigyatGoel/gotorch/data"
// 	layer "github.com/VigyatGoel/gotorch/layers"
// 	"github.com/VigyatGoel/gotorch/loss"
// 	"github.com/VigyatGoel/gotorch/network"
// 	"github.com/VigyatGoel/gotorch/optimizer"
// 	"github.com/VigyatGoel/gotorch/utils"
// 	"gonum.org/v1/gonum/mat"
// )

// const (
// 	BatchSize = 32
// )

// func main() {
// 	dl := data.NewDataLoader("", data.ImageClassification, BatchSize)
// 	dl.ImageDir = "examples/mnist"
// 	dl.ImageWidth = 28
// 	dl.ImageHeight = 28
// 	dl.BatchSize = BatchSize
// 	dl.Shuffle = true
// 	dl.Seed = 42
// 	if err := dl.Load(); err != nil {
// 		log.Fatalf("Error loading data: %v", err)
// 	}
// 	classNames := dl.ClassNames
// 	fmt.Println("Classes:", classNames)

// 	testSplit := 0.2
// 	total := len(dl.ImageSamples)
// 	testStart := int(float64(total) * (1 - testSplit))
// 	trainSamples := dl.ImageSamples[:testStart]
// 	testSamples := dl.ImageSamples[testStart:]

// 	model := createCNN(len(classNames))
// 	criterion := loss.NewCrossEntropyLoss()
// 	modelPath := "saved_image_model.gth"
// 	trainAndEvaluateCNN(model, criterion, dl, trainSamples, testSamples, 3, modelPath)
// 	loadAndUseModel(modelPath, dl, testSamples)
// }

// func createCNN(numClasses int) *network.Sequential {
// 	return network.NewSequential(
// 		layer.NewConv2D(3, 8, 3, 1, 1),
// 		layer.NewReLU(),
// 		layer.NewConv2D(8, 16, 3, 1, 1),
// 		layer.NewReLU(),
// 		layer.NewLinear(16*28*28, numClasses),
// 		layer.NewSoftmax(),
// 	)
// }

// func trainAndEvaluateCNN(model *network.Sequential, criterion *loss.CrossEntropyLoss, dl *data.DataLoader, trainSamples, testSamples []data.ImageSample, epochs int, modelPath string) {
// 	startTime := time.Now()
// 	model.SetOptimizer(optimizer.DefaultAdam(0.001))
// 	for epoch := 0; epoch < epochs; epoch++ {
// 		epochLoss := 0.0
// 		batches := getImageBatchesFromSamples(dl, trainSamples, epoch, dl.BatchSize)
// 		batchStartTime := time.Now()
// 		for _, batch := range batches {
// 			model.GetOptimizer().ZeroGrad()
// 			preds := model.Forward(batch.Features)
// 			lossVal := criterion.Forward(preds, batch.Targets)
// 			grad := criterion.Backward()
// 			model.Backward(grad)
// 			epochLoss += lossVal
// 		}
// 		avgEpochLoss := epochLoss / float64(len(batches))
// 		epochTime := time.Since(batchStartTime).Seconds()
// 		fmt.Printf("Epoch [%d/%d] Average Loss: %.4f (%.2f sec)\n", epoch+1, epochs, avgEpochLoss, epochTime)
// 	}
// 	totalTime := time.Since(startTime).Seconds()
// 	fmt.Printf("Training completed in %.2f seconds\n", totalTime)

// 	acc := evaluateCNN(model, dl, testSamples, dl.BatchSize)
// 	fmt.Printf("Test Accuracy: %.2f%%\n", acc*100)

// 	if modelPath != "" {
// 		err := model.Save(modelPath)
// 		if err != nil {
// 			fmt.Printf("Error saving model: %v\n", err)
// 		} else {
// 			fmt.Printf("Model saved successfully to %s\n", modelPath)
// 		}
// 	}
// }

// func getImageBatchesFromSamples(dl *data.DataLoader, samples []data.ImageSample, epoch, batchSize int) []data.Batch {
// 	total := len(samples)
// 	indices := make([]int, total)
// 	for i := 0; i < total; i++ {
// 		indices[i] = i
// 	}
// 	if dl.Shuffle {
// 		epochSeed := dl.Seed + int64(epoch)
// 		r := rand.New(rand.NewSource(epochSeed))
// 		r.Shuffle(total, func(i, j int) {
// 			indices[i], indices[j] = indices[j], indices[i]
// 		})
// 	}
// 	numBatches := (total + batchSize - 1) / batchSize
// 	batches := make([]data.Batch, 0, numBatches)
// 	for i := 0; i < total; i += batchSize {
// 		end := i + batchSize
// 		if end > total {
// 			end = total
// 		}
// 		var features []*mat.Dense
// 		labels := make([][]float64, 0, end-i)
// 		for _, idx := range indices[i:end] {
// 			sample := samples[idx]
// 			imgMat, _, err := data.LoadImageRGB(sample.Path, dl.ImageWidth, dl.ImageHeight)
// 			if err != nil {
// 				continue
// 			}
// 			features = append(features, imgMat)
// 			label := make([]float64, len(dl.ClassNames))
// 			label[sample.ClassIdx] = 1.0
// 			labels = append(labels, label)
// 		}
// 		if len(features) == 0 {
// 			continue
// 		}
// 		rows := len(features)
// 		cols := features[0].RawMatrix().Cols
// 		featuresData := make([]float64, 0, rows*cols)
// 		for _, f := range features {
// 			featuresData = append(featuresData, f.RawRowView(0)...)
// 		}
// 		batchFeatures := mat.NewDense(rows, cols, featuresData)
// 		labelsData := make([]float64, 0, len(labels)*len(dl.ClassNames))
// 		for _, l := range labels {
// 			labelsData = append(labelsData, l...)
// 		}
// 		batchTargets := mat.NewDense(rows, len(dl.ClassNames), labelsData)
// 		batches = append(batches, data.Batch{Features: batchFeatures, Targets: batchTargets})
// 	}
// 	return batches
// }

// func evaluateCNN(model *network.Sequential, dl *data.DataLoader, samples []data.ImageSample, batchSize int) float64 {
// 	correct := 0
// 	total := 0
// 	for i := 0; i < len(samples); i += batchSize {
// 		end := i + batchSize
// 		if end > len(samples) {
// 			end = len(samples)
// 		}
// 		var features []*mat.Dense
// 		labels := make([][]float64, 0, end-i)
// 		for _, sample := range samples[i:end] {
// 			imgMat, _, err := data.LoadImageRGB(sample.Path, dl.ImageWidth, dl.ImageHeight)
// 			if err != nil {
// 				continue
// 			}
// 			features = append(features, imgMat)
// 			label := make([]float64, len(dl.ClassNames))
// 			label[sample.ClassIdx] = 1.0
// 			labels = append(labels, label)
// 		}
// 		if len(features) == 0 {
// 			continue
// 		}
// 		rows := len(features)
// 		cols := features[0].RawMatrix().Cols
// 		featuresData := make([]float64, 0, rows*cols)
// 		for _, f := range features {
// 			featuresData = append(featuresData, f.RawRowView(0)...)
// 		}
// 		batchFeatures := mat.NewDense(rows, cols, featuresData)
// 		preds := model.Forward(batchFeatures)
// 		for j := 0; j < rows; j++ {
// 			predictedClass := utils.GetMaxIndexRow(preds, j)
// 			trueClass := 0
// 			for k := 0; k < len(dl.ClassNames); k++ {
// 				if labels[j][k] == 1.0 {
// 					trueClass = k
// 					break
// 				}
// 			}
// 			if predictedClass == trueClass {
// 				correct++
// 			}
// 			total++
// 		}
// 	}
// 	if total == 0 {
// 		return 0
// 	}
// 	return float64(correct) / float64(total)
// }

// func loadAndUseModel(modelPath string, dl *data.DataLoader, testSamples []data.ImageSample) {
// 	fmt.Printf("\nLoading model from %s\n", modelPath)
// 	loadedModel, err := network.Load(modelPath)
// 	if err != nil {
// 		fmt.Printf("Error loading model: %v\n", err)
// 		return
// 	}
// 	fmt.Println("Model loaded successfully! Evaluating...")
// 	acc := evaluateCNN(loadedModel, dl, testSamples, dl.BatchSize)
// 	fmt.Printf("Loaded model accuracy: %.2f%%\n", acc*100)
// }
