package main

// import (
// 	"fmt"
// 	"log"

// 	"github.com/VigyatGoel/gotorch/data"
// 	"github.com/VigyatGoel/gotorch/layer"
// 	"github.com/VigyatGoel/gotorch/loss"
// 	"github.com/VigyatGoel/gotorch/network"
// 	"github.com/VigyatGoel/gotorch/optimizer"
// 	"github.com/VigyatGoel/gotorch/utils"
// )

// func main() {
// 	// Load CSV data
// 	dataLoader := data.NewDataLoader("examples/iris.csv", data.Classification, 32, false)
// 	err := dataLoader.Load()
// 	if err != nil {
// 		log.Fatalf("Error loading data: %v", err)
// 	}

// 	dataLoader.NormalizeFeatures()
// 	numFeatures := dataLoader.NumFeatures()
// 	numClasses := len(dataLoader.GetClassNames())
// 	x_train, y_train, x_test, y_test := dataLoader.Split()

// 	// Create model with Dropout
// 	model := network.NewSequential(
// 		layer.NewLinear(numFeatures, 64),
// 		layer.NewReLU(),
// 		layer.NewLinear(64, 32),
// 		layer.NewReLU(),
// 		layer.NewDropout(0.3), // 30% dropout
// 		layer.NewLinear(32, numClasses),
// 		layer.NewSoftmax(),
// 	)

// 	// Setup training
// 	criterion := loss.NewCrossEntropyLoss()
// 	optimizer := optimizer.DefaultAdam(0.001)
// 	model.SetOptimizer(optimizer)
// 	epochs := 100

// 	// Training loop
// 	for epoch := 0; epoch < epochs; epoch++ {
// 		model.Train() // Enable dropout
// 		runningLoss := 0.0
// 		correct, total := 0, 0
// 		batches := dataLoader.GetBatches(x_train, y_train, epoch)

// 		for _, batch := range batches {
// 			// Forward pass
// 			preds := model.Forward(batch.Features)
// 			lossVal := criterion.Forward(preds, batch.Targets)

// 			// Backward pass
// 			model.GetOptimizer().ZeroGrad()
// 			grad := criterion.Backward()
// 			model.Backward(grad)

// 			// Accumulate metrics
// 			runningLoss += lossVal

// 			shape := batch.Features.Shape()
// 			for i := 0; i < shape[0]; i++ {
// 				if utils.GetMaxIndexRow(preds, i) == utils.GetMaxIndexRow(batch.Targets, i) {
// 					correct++
// 				}
// 				total++
// 			}
// 		}

// 		trainAcc := 100.0 * float64(correct) / float64(total)
// 		avgLoss := runningLoss / float64(len(batches))
// 		fmt.Printf("Epoch [%d/%d] Loss: %.4f Train Acc: %.2f%%\n", epoch+1, epochs, avgLoss, trainAcc)
// 	}

// 	// Test evaluation
// 	model.Eval() // Disable dropout
// 	preds := model.Predict(x_test)
// 	correct := 0
// 	shape := x_test.Shape()
// 	for i := 0; i < shape[0]; i++ {
// 		if utils.GetMaxIndexRow(preds, i) == utils.GetMaxIndexRow(y_test, i) {
// 			correct++
// 		}
// 	}

// 	testAcc := 100.0 * float64(correct) / float64(shape[0])
// 	fmt.Printf("Test Accuracy: %.2f%%\n", testAcc)
// }
