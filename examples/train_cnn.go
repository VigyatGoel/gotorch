package main

import (
	"fmt"

	"github.com/VigyatGoel/gotorch/data"
	"github.com/VigyatGoel/gotorch/layer"
	"github.com/VigyatGoel/gotorch/loss"
	"github.com/VigyatGoel/gotorch/network"
	"github.com/VigyatGoel/gotorch/optimizer"
	"github.com/VigyatGoel/gotorch/utils"
)

func main() {
	// Load data
	classNames, err := data.GetClassNamesFromDir("examples/cifar10/train")
	if err != nil {
		fmt.Println("Error reading class names:", err)
	}

	dataLoader := &data.DataLoader{
		DataType:    data.ImageClassification,
		ImageDir:    "examples/cifar10/train",
		ClassNames:  classNames,
		BatchSize:   32,
		Streaming:   true,
		Shuffle:     true,
		SplitRatio:  0.8,
		ImageWidth:  32,
		ImageHeight: 32,
		Grayscale:   false,
		Prefetch:    4,
	}
	err = dataLoader.Load()
	if err != nil {
		fmt.Println("Error loading data:", err)
	}

	model := network.NewSequential(
		layer.NewConv2D(3, 4, 3, 1, 1),
		layer.NewReLU(),
		layer.NewMaxPool2D(2, 2),

		layer.NewConv2D(4, 8, 3, 1, 1),
		layer.NewReLU(),
		layer.NewMaxPool2D(2, 2),

		layer.NewFlatten(),
		layer.NewLinear(8*8*8, len(classNames)),
		layer.NewSoftmax(),
	)

	// Setup training
	criterion := loss.NewCrossEntropyLoss()
	optimizer := optimizer.DefaultAdam(0.001)
	model.SetOptimizer(optimizer)
	epochs := 20

	// Training loop
	for epoch := range epochs {
		runningLoss := 0.0
		correct, total := 0, 0
		batchCount := 0

		for batch := range dataLoader.TrainBatches(epoch) {
			// Forward pass
			preds := model.Forward(batch.Features)
			lossVal := criterion.Forward(preds, batch.Targets)

			// Backward pass
			model.GetOptimizer().ZeroGrad()
			grad := criterion.Backward()
			model.Backward(grad)

			// Accumulate metrics
			runningLoss += lossVal
			batchCount++

			shape := batch.Features.Shape()
			for i := 0; i < shape[0]; i++ {
				if utils.GetMaxIndexRow(preds, i) == utils.GetMaxIndexRow(batch.Targets, i) {
					correct++
				}
				total++
			}
		}

		trainAcc := 100.0 * float64(correct) / float64(total)
		avgLoss := runningLoss / float64(batchCount)
		fmt.Printf("Epoch [%d/%d] Loss: %.4f Train Acc: %.2f%%\n", epoch+1, epochs, avgLoss, trainAcc)
	}

	// Test evaluation
	correct, total := 0, 0
	for batch := range dataLoader.TestBatches() {
		preds := model.Predict(batch.Features)
		shape := batch.Features.Shape()
		for i := 0; i < shape[0]; i++ {
			if utils.GetMaxIndexRow(preds, i) == utils.GetMaxIndexRow(batch.Targets, i) {
				correct++
			}
			total++
		}
	}

	testAcc := 100.0 * float64(correct) / float64(total)
	fmt.Printf("Test Accuracy: %.2f%%\n", testAcc)
}
