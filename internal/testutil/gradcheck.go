package testutil

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

const (
	gradCheckEps = 1e-4
	gradCheckTol = 1e-5
)

// Forwardable is satisfied by gotorch layers that have a differentiable backward pass.
type Forwardable interface {
	Forward(input *mat.Dense) *mat.Dense
	Backward(dout *mat.Dense) *mat.Dense
	ClearCache()
}

// WeightedLayer is satisfied by layers with learnable parameters (e.g. Linear).
type WeightedLayer interface {
	Forwardable
	GetWeights() *mat.Dense
	GetGradients() *mat.Dense
	GetBiases() *mat.Dense
	GetBiasGradients() *mat.Dense
}

// MatClose asserts element-wise closeness within tol using relative error.
func MatClose(t *testing.T, want, got *mat.Dense, tol float64, msg string) {
	t.Helper()
	wr, wc := want.Dims()
	gr, gc := got.Dims()
	if !assert.Equal(t, wr, gr, "%s: row mismatch", msg) || !assert.Equal(t, wc, gc, "%s: col mismatch", msg) {
		return
	}
	for i := 0; i < wr; i++ {
		for j := 0; j < wc; j++ {
			w := want.At(i, j)
			g := got.At(i, j)
			denom := math.Max(math.Abs(w), math.Max(math.Abs(g), 1e-8))
			relErr := math.Abs(w-g) / denom
			assert.Less(t, relErr, tol, "%s [%d,%d]: relErr=%e want=%e got=%e", msg, i, j, relErr, w, g)
		}
	}
}

// GradCheck verifies Backward input gradients via central finite differences.
// Do not use for stochastic layers (Dropout) or passthrough layers (Softmax, Flatten).
func GradCheck(t *testing.T, l Forwardable, input *mat.Dense) {
	t.Helper()
	defer l.ClearCache()
	rows, cols := input.Dims()
	inputData := input.RawMatrix().Data

	// Determine output shape.
	l.ClearCache()
	out := l.Forward(input)
	outRows, outCols := out.Dims()
	onesGrad := onesMatrix(outRows, outCols)

	// Analytical gradient.
	l.ClearCache()
	l.Forward(input)
	analyticalGrad := l.Backward(onesGrad)
	// Snapshot analyticalGrad.RawMatrix().Data here because the subsequent
	// ClearCache() calls in the perturbation loop must not corrupt the
	// analytical gradient. This requires that layers return gradient tensors
	// that do not alias their internal mutable state — all current gotorch
	// layers satisfy this.
	aData := copySlice(analyticalGrad.RawMatrix().Data)

	// Numerical gradient via central differences.
	for i := 0; i < rows*cols; i++ {
		orig := inputData[i]

		inputData[i] = orig + gradCheckEps
		l.ClearCache()
		fPlus := matSum(l.Forward(input))

		inputData[i] = orig - gradCheckEps
		l.ClearCache()
		fMinus := matSum(l.Forward(input))

		inputData[i] = orig

		numerical := (fPlus - fMinus) / (2 * gradCheckEps)
		analytical := aData[i]
		denom := math.Max(math.Abs(analytical), math.Max(math.Abs(numerical), 1e-8))
		relErr := math.Abs(analytical-numerical) / denom
		assert.Less(t, relErr, gradCheckTol,
			"input grad [%d]: analytical=%e numerical=%e relErr=%e", i, analytical, numerical, relErr)
	}
}

// GradCheckWeights verifies weight and bias gradients via central finite differences.
// GetWeights() and GetBiases() must return the layer's actual parameter matrices (not copies).
// In gotorch, Linear.GetWeights() returns l.weightMat directly, so in-place mutation works.
func GradCheckWeights(t *testing.T, l WeightedLayer, input *mat.Dense) {
	t.Helper()
	defer l.ClearCache()

	l.ClearCache()
	out := l.Forward(input)
	outRows, outCols := out.Dims()
	onesGrad := onesMatrix(outRows, outCols)

	// Analytical gradients from one forward+backward.
	l.ClearCache()
	l.Forward(input)
	l.Backward(onesGrad)
	aWData := copySlice(l.GetGradients().RawMatrix().Data)
	aBData := copySlice(l.GetBiasGradients().RawMatrix().Data)

	// Numerical weight gradients (perturb underlying data in-place).
	wData := l.GetWeights().RawMatrix().Data
	for i := range wData {
		orig := wData[i]

		wData[i] = orig + gradCheckEps
		l.ClearCache()
		fPlus := matSum(l.Forward(input))

		wData[i] = orig - gradCheckEps
		l.ClearCache()
		fMinus := matSum(l.Forward(input))

		wData[i] = orig

		numerical := (fPlus - fMinus) / (2 * gradCheckEps)
		analytical := aWData[i]
		denom := math.Max(math.Abs(analytical), math.Max(math.Abs(numerical), 1e-8))
		relErr := math.Abs(analytical-numerical) / denom
		assert.Less(t, relErr, gradCheckTol,
			"weight grad [%d]: analytical=%e numerical=%e relErr=%e", i, analytical, numerical, relErr)
	}

	// Numerical bias gradients.
	bData := l.GetBiases().RawMatrix().Data
	for i := range bData {
		orig := bData[i]

		bData[i] = orig + gradCheckEps
		l.ClearCache()
		fPlus := matSum(l.Forward(input))

		bData[i] = orig - gradCheckEps
		l.ClearCache()
		fMinus := matSum(l.Forward(input))

		bData[i] = orig

		numerical := (fPlus - fMinus) / (2 * gradCheckEps)
		analytical := aBData[i]
		denom := math.Max(math.Abs(analytical), math.Max(math.Abs(numerical), 1e-8))
		relErr := math.Abs(analytical-numerical) / denom
		assert.Less(t, relErr, gradCheckTol,
			"bias grad [%d]: analytical=%e numerical=%e relErr=%e", i, analytical, numerical, relErr)
	}
}

func onesMatrix(rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = 1.0
	}
	return mat.NewDense(rows, cols, data)
}

func matSum(m *mat.Dense) float64 {
	var s float64
	for _, v := range m.RawMatrix().Data {
		s += v
	}
	return s
}

func copySlice(src []float64) []float64 {
	dst := make([]float64, len(src))
	copy(dst, src)
	return dst
}
