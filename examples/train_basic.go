package main

import (
	"fmt"
	"log"

	"github.com/VigyatGoel/gotorch/data"
	layer "github.com/VigyatGoel/gotorch/layers"
	"github.com/VigyatGoel/gotorch/loss"
	"github.com/VigyatGoel/gotorch/network"
	"github.com/VigyatGoel/gotorch/optimizer"
)

const (
	BatchSize = 32
)

func main() {
	dataLoader := data.NewDataLoader("examples/iris.csv", data.Classification, BatchSize)
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
	epochs := 20

	fmt.Println("\nTRAINING WITH ADAM")
	adamOpt := optimizer.DefaultAdam(0.001)
	model.SetOptimizer(adamOpt)

	modelPath := "iris_model.gth"
	trainAndEvaluate(model, criterion, dataLoader, x_train, y_train, x_test, y_test, epochs, modelPath)

	loadAndUseModel(modelPath, x_test, y_test)
}

func createModel(inputFeatures int) *network.Sequential {
	return network.NewSequential(
		layer.NewLinear(inputFeatures, 128),
		layer.NewReLU(),
		layer.NewLinear(128, 64),
		layer.NewReLU(),
		layer.NewLinear(64, 32),
		layer.NewReLU(),
		layer.NewLinear(32, 3),
		layer.NewSoftmax(),
	)
}

func trainAndEvaluate(model *network.Sequential, criterion *loss.CrossEntropyLoss,
	dataLoader *data.DataLoader,
	x_train, y_train, x_test, y_test [][]float64, epochs int, modelPath string) {
	for epoch := range epochs {
		epochLoss := 0.0
		batches := dataLoader.GetBatches(x_train, y_train, epoch)

		for _, batch := range batches {
			preds := model.Forward(batch.Features)
			lossVal := criterion.Forward(preds, batch.Targets)
			grad := criterion.Backward()
			model.Backward(grad)
			epochLoss += lossVal
		}

		avgEpochLoss := epochLoss / float64(len(batches))
		fmt.Printf("Epoch [%d/%d] Average Loss: %.4f\n", epoch+1, epochs, avgEpochLoss)
	}

	preds := model.Predict(x_test)
	correct := 0
	for i := range x_test {
		predictedClass := getMaxIndex(preds[i])
		actualClass := getMaxIndex(y_test[i])
		if predictedClass == actualClass {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(x_test)) * 100
	fmt.Printf("Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, len(x_test))

	if modelPath != "" {
		err := model.Save(modelPath)
		if err != nil {
			fmt.Printf("Error saving model: %v\n", err)
		} else {
			fmt.Printf("Model saved successfully to %s\n", modelPath)
		}
	}
}

func loadAndUseModel(modelPath string, x_test, y_test [][]float64) {
	fmt.Printf("\nLoading model from %s\n", modelPath)
	loadedModel, err := network.Load(modelPath)
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		return
	}

	fmt.Println("Model loaded successfully! Evaluating...")

	preds := loadedModel.Predict(x_test)
	correct := 0
	for i := range x_test {
		predictedClass := getMaxIndex(preds[i])
		actualClass := getMaxIndex(y_test[i])
		if predictedClass == actualClass {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(x_test)) * 100
	fmt.Printf("Loaded model accuracy: %.2f%% (%d/%d)\n", accuracy, correct, len(x_test))
}

func getMaxIndex(values []float64) int {
	maxIdx := 0
	maxVal := values[0]

	for i, val := range values {
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}

	return maxIdx
}
