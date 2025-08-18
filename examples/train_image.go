package main

import (
	"fmt"
	"log"
	"time"

	"github.com/VigyatGoel/gotorch/data"
	layer "github.com/VigyatGoel/gotorch/layers"
	"github.com/VigyatGoel/gotorch/loss"
	"github.com/VigyatGoel/gotorch/network"
	"github.com/VigyatGoel/gotorch/optimizer"
	"github.com/VigyatGoel/gotorch/utils"
)

const (
	BatchSize    = 128
	IMAGE_WIDTH  = 28
	IMAGE_HEIGHT = 28
)

func main() {
	classNames, err := data.GetClassNamesFromDir("examples/mnist")
	fmt.Println(classNames)
	if err != nil {
		log.Fatalf("Error reading class names: %v", err)
	}
	dataLoader := &data.DataLoader{
		DataType:    data.ImageClassification,
		ImageDir:    "examples/mnist",
		ClassNames:  classNames,
		BatchSize:   BatchSize,
		Shuffle:     true,
		SplitRatio:  0.8,
		ImageWidth:  IMAGE_WIDTH,
		ImageHeight: IMAGE_HEIGHT,
	}
	err = dataLoader.Load()
	if err != nil {
		log.Fatalf("Error loading image data: %v", err)
	}

	numFeatures := IMAGE_WIDTH * IMAGE_HEIGHT // Grayscale images have 1 channel
	fmt.Printf("Detected %d features from the grayscale image dataset.\n", numFeatures)

	total := len(dataLoader.ImageSamples)
	split := int(float64(total) * dataLoader.SplitRatio)
	trainSamples := dataLoader.ImageSamples[:split]
	testSamples := dataLoader.ImageSamples[split:]

	trainLoader := *dataLoader
	trainLoader.ImageSamples = trainSamples

	testLoader := *dataLoader
	testLoader.ImageSamples = testSamples

	model := network.NewSequential(
		layer.NewLinear(numFeatures, 512),
		layer.NewLinear(512, 256),
		layer.NewReLU(),
		layer.NewLinear(256, 128),
		layer.NewReLU(),
		layer.NewLinear(128, 64),
		layer.NewReLU(),
		layer.NewLinear(64, len(dataLoader.ClassNames)),
		layer.NewSoftmax(),
	)
	criterion := loss.NewCrossEntropyLoss()
	epochs := 5

	fmt.Println("\nTRAINING WITH ADAM (Image Dataset)")
	adamOpt := optimizer.DefaultAdam(0.001)
	model.SetOptimizer(adamOpt)

	startTime := time.Now()

	for epoch := 0; epoch < epochs; epoch++ {
		epochLoss := 0.0
		batchStartTime := time.Now()
		batches := trainLoader.GetImageBatches(epoch, trainLoader.BatchSize)

		for _, batch := range batches {
			model.GetOptimizer().ZeroGrad()
			preds := model.Forward(batch.Features)
			lossVal := criterion.Forward(preds, batch.Targets)
			grad := criterion.Backward()
			model.Backward(grad)
			epochLoss += lossVal
		}

		avgEpochLoss := epochLoss / float64(len(batches))
		epochTime := time.Since(batchStartTime).Seconds()
		fmt.Printf("Epoch [%d/%d] Average Loss: %.4f (%.2f sec)\n", epoch+1, epochs, avgEpochLoss, epochTime)
	}

	totalTime := time.Since(startTime).Seconds()
	fmt.Printf("Training completed in %.2f seconds\n", totalTime)

	testBatches := testLoader.GetImageBatches(0, testLoader.BatchSize)
	correct := 0
	totalTest := 0
	for _, batch := range testBatches {
		preds := model.Predict(batch.Features)
		rows, _ := batch.Features.Dims()
		for i := 0; i < rows; i++ {
			predictedClass := utils.GetMaxIndexRow(preds, i)
			actualClass := utils.GetMaxIndexRow(batch.Targets, i)
			if predictedClass == actualClass {
				correct++
			}
			totalTest++
		}
	}
	accuracy := float64(correct) / float64(totalTest) * 100
	fmt.Printf("Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, totalTest)

	modelPath := "saved_image_model.gth"
	if modelPath != "" {
		err := model.Save(modelPath)
		if err != nil {
			fmt.Printf("Error saving model: %v\n", err)
		} else {
			fmt.Printf("Model saved successfully to %s\n", modelPath)
		}
	}

	fmt.Printf("\nLoading model from %s\n", modelPath)
	loadedModel, err := network.Load(modelPath)
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		return
	}

	fmt.Println("Model loaded successfully! Evaluating...")
	testBatches = testLoader.GetImageBatches(0, testLoader.BatchSize)
	correct = 0
	totalTest = 0
	for _, batch := range testBatches {
		preds := loadedModel.Predict(batch.Features)
		rows, _ := batch.Features.Dims()
		for i := 0; i < rows; i++ {
			predictedClass := utils.GetMaxIndexRow(preds, i)
			actualClass := utils.GetMaxIndexRow(batch.Targets, i)
			if predictedClass == actualClass {
				correct++
			}
			totalTest++
		}
	}
	accuracy = float64(correct) / float64(totalTest) * 100
	fmt.Printf("Loaded model accuracy: %.2f%% (%d/%d)\n", accuracy, correct, totalTest)
}
