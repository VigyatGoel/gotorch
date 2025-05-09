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
	"gonum.org/v1/gonum/mat"
)

const (
	BatchSize = 512
)

func main() {
	dataLoader := data.NewDataLoader("examples/train.csv", data.Classification, BatchSize)
	err := dataLoader.Load()
	if err != nil {
		log.Fatalf("Error loading data: %v", err)
	}

	dataLoader.NormalizeFeatures()

	numFeatures := dataLoader.NumFeatures()
	if numFeatures == 0 {
		log.Fatalf("Could not determine number of features from the data.")
	}
	fmt.Printf("Detected %d features from the dataset.\n", numFeatures)

	x_train, y_train, x_test, y_test := dataLoader.Split()

	model := createModel(numFeatures)
	criterion := loss.NewCrossEntropyLoss()
	epochs := 50

	fmt.Println("\nTRAINING WITH ADAM")
	adamOpt := optimizer.DefaultAdam(0.001)
	model.SetOptimizer(adamOpt)

	modelPath := "saved_model.gth"
	trainAndEvaluate(model, criterion, dataLoader, x_train, y_train, x_test, y_test, epochs, modelPath)

	loadAndUseModel(modelPath, x_test, y_test)
}

func createModel(inputFeatures int) *network.Sequential {
	return network.NewSequential(
		layer.NewLinear(inputFeatures, 1024),
		layer.NewLinear(1024, 512),
		layer.NewSiLU(),
		layer.NewLinear(512, 256),
		layer.NewSiLU(),
		layer.NewLinear(256, 128),
		layer.NewSiLU(),
		layer.NewLinear(128, 64),
		layer.NewSiLU(),
		layer.NewLinear(64, 32),
		layer.NewSiLU(),
		layer.NewLinear(32, 3),
		layer.NewSoftmax(),
	)
}

func trainAndEvaluate(model *network.Sequential, criterion *loss.CrossEntropyLoss,
	dataLoader *data.DataLoader,
	x_train, y_train, x_test, y_test *mat.Dense, epochs int, modelPath string) {

	startTime := time.Now()

	for epoch := 0; epoch < epochs; epoch++ {
		epochLoss := 0.0
		batchStartTime := time.Now()
		batches := dataLoader.GetBatches(x_train, y_train, epoch)

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

	preds := model.Predict(x_test)
	correct := 0
	rows, _ := x_test.Dims()
	for i := 0; i < rows; i++ {
		predictedClass := utils.GetMaxIndexRow(preds, i)
		actualClass := utils.GetMaxIndexRow(y_test, i)
		if predictedClass == actualClass {
			correct++
		}
	}

	accuracy := float64(correct) / float64(rows) * 100
	fmt.Printf("Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, rows)

	if modelPath != "" {
		err := model.Save(modelPath)
		if err != nil {
			fmt.Printf("Error saving model: %v\n", err)
		} else {
			fmt.Printf("Model saved successfully to %s\n", modelPath)
		}
	}
}

func loadAndUseModel(modelPath string, x_test, y_test *mat.Dense) {
	fmt.Printf("\nLoading model from %s\n", modelPath)
	loadedModel, err := network.Load(modelPath)
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		return
	}

	fmt.Println("Model loaded successfully! Evaluating...")

	preds := loadedModel.Predict(x_test)
	correct := 0
	rows, _ := x_test.Dims()
	for i := 0; i < rows; i++ {
		predictedClass := utils.GetMaxIndexRow(preds, i)
		actualClass := utils.GetMaxIndexRow(y_test, i)
		if predictedClass == actualClass {
			correct++
		}
	}

	accuracy := float64(correct) / float64(rows) * 100
	fmt.Printf("Loaded model accuracy: %.2f%% (%d/%d)\n", accuracy, correct, rows)
}
