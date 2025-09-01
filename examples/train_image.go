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
// 	IMAGE_WIDTH  = 28
// 	IMAGE_HEIGHT = 28
// )

// func main() {
// 	classNames, err := data.GetClassNamesFromDir("examples/mnist")
// 	fmt.Println(classNames)
// 	if err != nil {
// 		log.Fatalf("Error reading class names: %v", err)
// 	}

// 	// Create and load the data loader
// 	dataLoader := &data.DataLoader{
// 		DataType:    data.ImageClassification,
// 		ImageDir:    "examples/mnist",
// 		ClassNames:  classNames,
// 		BatchSize:   BatchSize,
// 		Shuffle:     true,
// 		SplitRatio:  0.8,
// 		ImageWidth:  IMAGE_WIDTH,
// 		ImageHeight: IMAGE_HEIGHT,
// 	}
// 	err = dataLoader.Load()
// 	if err != nil {
// 		log.Fatalf("Error loading image data: %v", err)
// 	}

// 	numFeatures := IMAGE_WIDTH * IMAGE_HEIGHT // Grayscale images have 1 channel
// 	fmt.Printf("Detected %d features from the grayscale image dataset.\n", numFeatures)

// 	// Use the same data loader for both training and testing
// 	// The GetImageBatches method will handle the splitting internally based on the SplitRatio
// 	model := network.NewSequential(
// 		layer.NewLinear(numFeatures, 512),
// 		layer.NewLinear(512, 256),
// 		layer.NewReLU(),
// 		layer.NewLinear(256, 128),
// 		layer.NewReLU(),
// 		layer.NewLinear(128, 64),
// 		layer.NewReLU(),
// 		layer.NewLinear(64, len(dataLoader.ClassNames)),
// 		layer.NewSoftmax(),
// 	)
// 	criterion := loss.NewCrossEntropyLoss()
// 	epochs := 5

// 	fmt.Println("\nTRAINING WITH ADAM (Image Dataset)")
// 	adamOpt := optimizer.DefaultAdam(0.001)
// 	model.SetOptimizer(adamOpt)

// 	startTime := time.Now()

// 	for epoch := 0; epoch < epochs; epoch++ {
// 		epochLoss := 0.0
// 		batchStartTime := time.Now()
// 		// Use the data loader directly for training batches
// 		batches := dataLoader.GetImageBatches(epoch, dataLoader.BatchSize)

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

// 	// For evaluation, we would ideally have a separate test set
// 	// But for now, we'll use the same data loader to demonstrate
// 	// In a real scenario, you'd want to load a separate test dataset
// 	evalBatches := dataLoader.GetImageBatches(0, dataLoader.BatchSize) // Use epoch 0 for evaluation
// 	correct := 0
// 	totalTest := 0
// 	for _, batch := range evalBatches {
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

// 	modelPath := "saved_image_model.gth"
// 	if modelPath != "" {
// 		err := model.Save(modelPath)
// 		if err != nil {
// 			fmt.Printf("Error saving model: %v\n", err)
// 		} else {
// 			fmt.Printf("Model saved successfully to %s\n", modelPath)
// 		}
// 	}

// 	fmt.Printf("\nLoading model from %s\n", modelPath)
// 	loadedModel, err := network.Load(modelPath)
// 	if err != nil {
// 		fmt.Printf("Error loading model: %v\n", err)
// 		return
// 	}

// 	fmt.Println("Model loaded successfully! Evaluating...")
// 	// Evaluate the loaded model
// 	evalBatches = dataLoader.GetImageBatches(0, dataLoader.BatchSize) // Use epoch 0 for evaluation
// 	correct = 0
// 	totalTest = 0
// 	for _, batch := range evalBatches {
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
// 	accuracy = float64(correct) / float64(totalTest) * 100
// 	fmt.Printf("Loaded model accuracy: %.2f%% (%d/%d)\n", accuracy, correct, totalTest)
// }
