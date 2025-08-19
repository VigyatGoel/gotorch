package data

import (
	"image"
	"image/draw"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"

	"github.com/nfnt/resize"
	"gorgonia.org/tensor"
)

func LoadImage(path string, width, height int) (*tensor.Dense, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var img image.Image
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".jpg", ".jpeg":
		img, err = jpeg.Decode(file)
	case ".png":
		img, err = png.Decode(file)
	default:
		return nil, err
	}
	if err != nil {
		return nil, err
	}

	grayImg := image.NewGray(img.Bounds())
	draw.Draw(grayImg, img.Bounds(), img, image.Point{}, draw.Src)

	resized := resize.Resize(uint(width), uint(height), grayImg, resize.Lanczos3)
	resizedGray, ok := resized.(*image.Gray)
	if !ok {
		b := resized.Bounds()
		gray := image.NewGray(b)
		draw.Draw(gray, b, resized, b.Min, draw.Src)
		resizedGray = gray
	}

	pixels := make([]float64, width*height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			c := resizedGray.GrayAt(x, y)
			pixels[y*width+x] = float64(c.Y) / 255.0
		}
	}
	return tensor.New(tensor.WithShape(1, width*height), tensor.WithBacking(pixels)), nil
}

// LoadImageGrayscale loads an image as grayscale and returns it as a flattened vector
// This function is specifically designed for loading single grayscale images for neural networks
func LoadImageGrayscale(path string, width, height int) (*tensor.Dense, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var img image.Image
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".jpg", ".jpeg":
		img, err = jpeg.Decode(file)
	case ".png":
		img, err = png.Decode(file)
	default:
		return nil, err
	}
	if err != nil {
		return nil, err
	}

	// Convert to grayscale
	grayImg := image.NewGray(img.Bounds())
	draw.Draw(grayImg, img.Bounds(), img, image.Point{}, draw.Src)

	// Resize the image
	resized := resize.Resize(uint(width), uint(height), grayImg, resize.Lanczos3)
	resizedGray, ok := resized.(*image.Gray)
	if !ok {
		b := resized.Bounds()
		gray := image.NewGray(b)
		draw.Draw(gray, b, resized, b.Min, draw.Src)
		resizedGray = gray
	}

	// Convert to normalized float64 values
	pixels := make([]float64, width*height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			c := resizedGray.GrayAt(x, y)
			pixels[y*width+x] = float64(c.Y) / 255.0
		}
	}
	return tensor.New(tensor.WithShape(1, width*height), tensor.WithBacking(pixels)), nil
}

// LoadImageRGB loads an image as a 3D tensor (channels, height, width) and flattens it for tensor.Dense
func LoadImageRGB(path string, width, height int) (*tensor.Dense, [3]int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, [3]int{}, err
	}
	defer file.Close()

	var img image.Image
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".jpg", ".jpeg":
		img, err = jpeg.Decode(file)
	case ".png":
		img, err = png.Decode(file)
	default:
		return nil, [3]int{}, err
	}
	if err != nil {
		return nil, [3]int{}, err
	}

	resized := resize.Resize(uint(width), uint(height), img, resize.Lanczos3)
	channels := 3
	pixels := make([]float64, channels*width*height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			pixels[0*width*height+y*width+x] = float64(r) / 65535.0
			pixels[1*width*height+y*width+x] = float64(g) / 65535.0
			pixels[2*width*height+y*width+x] = float64(b) / 65535.0
		}
	}
	return tensor.New(tensor.WithShape(1, channels*width*height), tensor.WithBacking(pixels)), [3]int{channels, height, width}, nil
}

func GetClassNamesFromDir(root string) ([]string, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	var classNames []string
	for _, entry := range entries {
		if entry.IsDir() {
			classNames = append(classNames, entry.Name())
		}
	}
	return classNames, nil
}

func LoadImagePathsAndLabels(root string, classNames []string) ([]ImageSample, error) {
	classToIdx := make(map[string]int)
	for i, name := range classNames {
		classToIdx[name] = i
	}
	var samples []ImageSample
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		dir := filepath.Base(filepath.Dir(path))
		classIdx, ok := classToIdx[dir]
		if !ok {
			return nil
		}
		samples = append(samples, ImageSample{Path: path, ClassIdx: classIdx})
		return nil
	})
	if err != nil {
		return nil, err
	}
	return samples, nil
}

func LoadImageDataset(root string, classNames []string) ([]*tensor.Dense, [][]float64, error) {
	var features []*tensor.Dense
	var labels [][]float64
	classToIdx := make(map[string]int)
	for i, name := range classNames {
		classToIdx[name] = i
	}

	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		dir := filepath.Base(filepath.Dir(path))
		classIdx, ok := classToIdx[dir]
		if !ok {
			return nil
		}
		imgMat, err := LoadImage(path, 28, 28)
		if err != nil {
			return nil
		}
		features = append(features, imgMat)
		label := make([]float64, len(classNames))
		label[classIdx] = 1.0
		labels = append(labels, label)
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	return features, labels, nil
}
