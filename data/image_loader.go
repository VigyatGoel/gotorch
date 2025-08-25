package data

import (
	"fmt"
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

// LoadImageAs4D loads an image and returns it as a 4D tensor [1, channels, height, width]
// This is the format needed for CNN models
func LoadImageAs4D(path string, width, height int, grayscale bool) (*tensor.Dense, error) {
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
		return nil, fmt.Errorf("unsupported image format: %s", ext)
	}
	if err != nil {
		return nil, err
	}

	// Resize the image
	resized := resize.Resize(uint(width), uint(height), img, resize.Lanczos3)

	var pixels []float64
	var channels int

	if grayscale {
		// Convert to grayscale
		channels = 1
		grayImg := image.NewGray(resized.Bounds())
		draw.Draw(grayImg, resized.Bounds(), resized, image.Point{}, draw.Src)

		pixels = make([]float64, channels*width*height)
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				c := grayImg.GrayAt(x, y)
				// Store in CHW format (channels, height, width)
				pixels[y*width+x] = float64(c.Y) / 255.0
			}
		}
	} else {
		// RGB image
		channels = 3
		pixels = make([]float64, channels*width*height)

		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				r, g, b, _ := resized.At(x, y).RGBA()
				// Store in CHW format (channels, height, width)
				pixels[0*height*width+y*width+x] = float64(r) / 65535.0
				pixels[1*height*width+y*width+x] = float64(g) / 65535.0
				pixels[2*height*width+y*width+x] = float64(b) / 65535.0
			}
		}
	}

	// Return as 4D tensor [1, channels, height, width]
	return tensor.New(
		tensor.WithShape(1, channels, height, width),
		tensor.WithBacking(pixels),
	), nil
}

// LoadImageGrayscale loads a grayscale image as 4D tensor [1, 1, height, width]
func LoadImageGrayscale(path string, width, height int) (*tensor.Dense, error) {
	return LoadImageAs4D(path, width, height, true)
}

// LoadImageRGB loads an RGB image as 4D tensor [1, 3, height, width]
func LoadImageRGB(path string, width, height int) (*tensor.Dense, error) {
	return LoadImageAs4D(path, width, height, false)
}

// LoadImageFlattened loads and flattens an image (for non-CNN models)
func LoadImageFlattened(path string, width, height int) (*tensor.Dense, error) {
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
		return nil, fmt.Errorf("unsupported image format: %s", ext)
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

		// Check if it's an image file
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
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
