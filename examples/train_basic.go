package main

import (
	"fmt"
	"log"

	"github.com/VigyatGoel/gotorch/data"
	"github.com/VigyatGoel/gotorch/layer"
	"github.com/VigyatGoel/gotorch/loss"
	"github.com/VigyatGoel/gotorch/network"
	"github.com/VigyatGoel/gotorch/optimizer"
	"github.com/VigyatGoel/gotorch/utils"
)

func main() {
	dataLoader := data.NewDataLoader("examples/iris.csv", data.Classification, 32)
	err := dataLoader.Load()
	if err != nil {
		log.Fatalf("Error loading data: %v", err)
	}

	dataLoader.NormalizeFeatures()
	numFeatures := dataLoader.NumFeatures()
	numClasses := len(dataLoader.GetClassNames())
	x_train, y_train, x_test, y_test := dataLoader.Split()

	model := network.NewSequential(
		layer.NewLinear(numFeatures, 64),
		layer.NewReLU(),
		layer.NewLinear(64, 32),
		layer.NewReLU(),
		layer.NewLinear(32, numClasses),
		layer.NewSoftmax(),
	)

	criterion := loss.NewCrossEntropyLoss()
	opt := optimizer.DefaultAdam(0.001)
	model.SetOptimizer(opt)
	epochs := 100

	for epoch := 0; epoch < epochs; epoch++ {
		runningLoss := 0.0
		correct, total := 0, 0
		batches := dataLoader.GetBatches(x_train, y_train, epoch)

		for _, batch := range batches {
			preds := model.Forward(batch.Features)
			lossVal := criterion.Forward(preds, batch.Targets)
			model.GetOptimizer().ZeroGrad()
			grad := criterion.Backward()
			model.Backward(grad)

			runningLoss += lossVal
			rows, _ := batch.Features.Dims()
			for i := 0; i < rows; i++ {
				if utils.GetMaxIndexRow(preds, i) == utils.GetMaxIndexRow(batch.Targets, i) {
					correct++
				}
				total++
			}
		}

		trainAcc := 100.0 * float64(correct) / float64(total)
		avgLoss := runningLoss / float64(len(batches))
		fmt.Printf("Epoch [%d/%d] Loss: %.4f Train Acc: %.2f%%\n", epoch+1, epochs, avgLoss, trainAcc)
	}

	preds := model.Predict(x_test)
	correct := 0
	testRows, _ := x_test.Dims()
	for i := 0; i < testRows; i++ {
		if utils.GetMaxIndexRow(preds, i) == utils.GetMaxIndexRow(y_test, i) {
			correct++
		}
	}
	fmt.Printf("Test Accuracy: %.2f%%\n", 100.0*float64(correct)/float64(testRows))
}
