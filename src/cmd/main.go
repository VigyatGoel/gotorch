package main

import (
	"fmt"
	"log"

	"github.com/VigyatGoel/gotorch/src/data"
	layer "github.com/VigyatGoel/gotorch/src/layers"
	"github.com/VigyatGoel/gotorch/src/loss"
	"github.com/VigyatGoel/gotorch/src/network"
)

func main() {
	dataLoader := data.NewDataLoader("src/cmd/iris.csv", data.Classification)

	err := dataLoader.Load()
	if err != nil {
		log.Fatalf("Error loading data: %v", err)
	}

	x_train, y_train, x_test, y_test := dataLoader.Split()

	model := network.NewSequential(
		layer.NewLinear(4, 128),
		layer.NewReLU(),
		layer.NewLinear(128, 64),
		layer.NewReLU(),
		layer.NewLinear(64, 32),
		layer.NewLinear(32, 3),
		layer.NewSoftmax(),
	)

	criterion := loss.NewCrossEntropyLoss()

	epochs := 100
	lr := 0.01

	for epoch := 0; epoch < epochs; epoch++ {
		preds := model.Forward(x_train)

		lossVal := criterion.Forward(preds, y_train)

		grad := criterion.Backward()
		model.Backward(grad, lr)

		fmt.Printf("Epoch [%d/%d] Loss: %.4f\n", epoch+1, epochs, lossVal)
	}

	fmt.Println("\nModel Evaluation:")
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
	fmt.Printf("\nAccuracy: %.2f%% (%d/%d)\n", accuracy, correct, len(x_test))
}

func getMaxIndex(values []float64) int {
	maxIndex := 0
	maxVal := values[0]

	for i, val := range values {
		if val > maxVal {
			maxVal = val
			maxIndex = i
		}
	}

	return maxIndex
}
