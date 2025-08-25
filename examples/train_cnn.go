package main

// import (
// 	"fmt"
// 	"log"
// 	"time"

// 	"github.com/VigyatGoel/gotorch/data"
// 	"github.com/VigyatGoel/gotorch/layer"
// 	"github.com/VigyatGoel/gotorch/loss"
// 	"github.com/VigyatGoel/gotorch/network"
// 	"github.com/VigyatGoel/gotorch/optimizer"
// 	"github.com/VigyatGoel/gotorch/utils"
// )

// const (
// 	BatchSize    = 128
// 	IMAGE_WIDTH  = 32
// 	IMAGE_HEIGHT = 32
// )

// func main() {
// 	// Get class names from the MNIST directory
// 	data_path := "examples/cifar10/train"
// 	classNames, err := data.GetClassNamesFromDir(data_path)
// 	if err != nil {
// 		log.Fatalf("Error reading class names: %v", err)
// 	}

// 	// Create and load the data loader

// 	dataLoader := &data.DataLoader{
// 		DataType:    data.ImageClassification,
// 		ImageDir:    data_path,
// 		ClassNames:  classNames,
// 		BatchSize:   BatchSize,
// 		Shuffle:     true,
// 		SplitRatio:  0.8,
// 		ImageWidth:  IMAGE_WIDTH,
// 		ImageHeight: IMAGE_HEIGHT,
// 		Grayscale:   false, // Load as RGB
// 	}
// 	err = dataLoader.Load()
// 	if err != nil {
// 		log.Fatalf("Error loading image data: %v", err)
// 	}

// 	fmt.Printf("Detected %d classes from the image dataset: %v\n", len(classNames), classNames)

// 	// Create CNN model
// 	model := createCNNModel(len(classNames))

// 	criterion := loss.NewCrossEntropyLoss()
// 	epochs := 10

// 	fmt.Println("\nTRAINING CNN WITH ADAM")
// 	adamOpt := optimizer.DefaultAdam(0.001)
// 	model.SetOptimizer(adamOpt)

// 	modelPath := "saved_cnn_model.gth"
// 	trainAndEvaluateCNN(model, criterion, dataLoader, epochs, modelPath)

// 	loadAndUseCNNModel(modelPath, dataLoader)
// }

// func createCNNModel(classCount int) *network.Sequential {
// 	return network.NewSequential(
// 		// First convolutional block
// 		layer.NewConv2D(3, 4, 3, 1, 1), // Input: 32x32x1 -> Output: 32x32x4
// 		layer.NewReLU(),
// 		layer.NewMaxPool2D(2, 2), // Output: 16x16x4

// 		// Second convolutional block
// 		layer.NewConv2D(4, 8, 3, 1, 1), // Input: 16x16x4 -> Output: 16x16x8
// 		layer.NewReLU(),
// 		layer.NewMaxPool2D(2, 2), // Output: 8x8x8

// 		// Flatten the output for the linear layer
// 		layer.NewFlatten(),

// 		// Linear layer: 8*8*8 = 512 features
// 		layer.NewLinear(8*8*8, classCount),

// 		// Softmax to convert logits to probabilities
// 		layer.NewSoftmax(),
// 	)
// }

// func trainAndEvaluateCNN(model *network.Sequential, criterion *loss.CrossEntropyLoss,
// 	dataLoader *data.DataLoader, epochs int, modelPath string) {

// 	startTime := time.Now()

// 	for epoch := 0; epoch < epochs; epoch++ {
// 		epochLoss := 0.0
// 		batchStartTime := time.Now()

// 		// Use the data loader directly for training batches
// 		batches := dataLoader.GetTrainImageBatches(epoch, dataLoader.BatchSize)

// 		for _, batch := range batches {
// 			model.GetOptimizer().ZeroGrad()

// 			// The data loader now provides 4D tensors, so no reshape is needed
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

// 	// Evaluation
// 	evalBatches := dataLoader.GetTestImageBatches(dataLoader.BatchSize) // Use epoch 0 for evaluation
// 	correct := 0
// 	totalTest := 0
// 	for _, batch := range evalBatches {
// 		// The data loader now provides 4D tensors, so no reshape is needed
// 		preds := model.Predict(batch.Features)
// 		shape := batch.Features.Shape()

// 		rows := shape[0]
// 		for i := 0; i < rows; i++ {
// 			predictedClass := utils.GetMaxIndexRow(preds, i)
// 			actualClass := utils.GetMaxIndexRow(batch.Targets, i)
// 			if predictedClass == actualClass {
// 				correct++
// 			}
// 			totalTest++
// 		}
// 	}

// 	accuracy := float64(correct) / float64(totalTest) * 100
// 	fmt.Printf("Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, totalTest)

// 	if modelPath != "" {
// 		err := model.Save(modelPath)
// 		if err != nil {
// 			fmt.Printf("Error saving model: %v\n", err)
// 		} else {
// 			fmt.Printf("Model saved successfully to %s\n", modelPath)
// 		}
// 	}
// }

// func loadAndUseCNNModel(modelPath string, dataLoader *data.DataLoader) {
// 	fmt.Printf("\nLoading model from %s\n", modelPath)
// 	loadedModel, err := network.Load(modelPath)
// 	if err != nil {
// 		fmt.Printf("Error loading model: %v\n", err)
// 		return
// 	}

// 	fmt.Println("Model loaded successfully! Evaluating...")

// 	// Evaluate the loaded model
// 	evalBatches := dataLoader.GetTestImageBatches(dataLoader.BatchSize) // Use epoch 0 for evaluation
// 	correct := 0
// 	totalTest := 0
// 	for _, batch := range evalBatches {
// 		// The data loader now provides 4D tensors, so no reshape is needed
// 		preds := loadedModel.Predict(batch.Features)
// 		shape := batch.Features.Shape()

// 		rows := shape[0]
// 		for i := 0; i < rows; i++ {
// 			predictedClass := utils.GetMaxIndexRow(preds, i)
// 			actualClass := utils.GetMaxIndexRow(batch.Targets, i)
// 			if predictedClass == actualClass {
// 				correct++
// 			}
// 			totalTest++
// 		}
// 	}

// 	accuracy := float64(correct) / float64(totalTest) * 100
// 	fmt.Printf("Loaded model accuracy: %.2f%% (%d/%d)\n", accuracy, correct, totalTest)
// }
