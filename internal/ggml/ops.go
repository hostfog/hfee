package ggml

import (
	"math"
	"runtime"
	"sync"
)

// Matrix is a 2D weight tensor stored row-major in its on-disk encoding.
type Matrix struct {
	Type    Type
	Rows    int
	Cols    int
	Data    []byte
	rowSize int
}

func NewMatrix(t Type, rows, cols int, data []byte) *Matrix {
	return &Matrix{Type: t, Rows: rows, Cols: cols, Data: data, rowSize: t.ByteSize(cols)}
}

// Row decodes row r into dst, which must hold Cols elements.
func (m *Matrix) Row(r int, dst []float32) {
	off := r * m.rowSize
	Dequantize(m.Type, dst[:m.Cols], m.Data[off:off+m.rowSize])
}

var workers = runtime.NumCPU()

// Parallel splits [0, n) into chunks and runs fn on each concurrently.
func Parallel(n int, fn func(start, end int)) {
	if n == 0 {
		return
	}
	chunks := min(workers, n)
	if chunks == 1 {
		fn(0, n)
		return
	}

	var wg sync.WaitGroup
	step := (n + chunks - 1) / chunks
	for start := 0; start < n; start += step {
		end := min(start+step, n)
		wg.Add(1)
		go func() {
			defer wg.Done()
			fn(start, end)
		}()
	}
	wg.Wait()
}

// MatMul computes out[t] = W · x[t] for n input rows.
// x is n×Cols and out is n×Rows, both row-major.
func MatMul(out []float32, w *Matrix, x []float32, n int) {
	if dot := intDots[w.Type]; dot != nil {
		matMulInt(out, w, x, n, dot)
		return
	}

	Parallel(w.Rows, func(start, end int) {
		row := make([]float32, w.Cols)
		for r := start; r < end; r++ {
			w.Row(r, row)
			for t := 0; t < n; t++ {
				out[t*w.Rows+r] = Dot(row, x[t*w.Cols:(t+1)*w.Cols])
			}
		}
	})
}

// matMulInt quantises the inputs to int8 once, then runs the integer kernel
// for every weight row. Each row is read once for all n inputs.
func matMulInt(out []float32, w *Matrix, x []float32, n int, dot func([]byte, *Q8, []float32) float32) {
	inputs := make([]*Q8, n)
	for t := range inputs {
		inputs[t] = NewQ8(w.Cols)
		inputs[t].Quantize(x[t*w.Cols : (t+1)*w.Cols])
	}

	Parallel(w.Rows, func(start, end int) {
		scratch := make([]float32, w.Cols/16)
		for r := start; r < end; r++ {
			row := w.Data[r*w.rowSize : (r+1)*w.rowSize]
			for t, in := range inputs {
				out[t*w.Rows+r] = dot(row, in, scratch)
			}
		}
	})
}

// Dot returns the dot product of a and b (len(b) >= len(a)).
func Dot(a, b []float32) float32 {
	return dotF32(a, b[:len(a)])
}

// Axpy adds alpha·x to y.
func Axpy(y []float32, alpha float32, x []float32) {
	axpyF32(y, alpha, x[:len(y)])
}

func axpyGeneric(y []float32, alpha float32, x []float32) {
	for i := range y {
		y[i] += alpha * x[i]
	}
}

func dotGeneric(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return s0 + s1 + s2 + s3
}

// RMSNorm writes x normalised and scaled by weight into dst.
func RMSNorm(dst, x, weight []float32, eps float32) {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	scale := float32(1 / math.Sqrt(ss/float64(len(x))+float64(eps)))
	for i, v := range x {
		dst[i] = v * scale * weight[i]
	}
}

func Softmax(x []float32) {
	maxv := x[0]
	for _, v := range x[1:] {
		maxv = max(maxv, v)
	}
	var sum float32
	for i, v := range x {
		e := float32(math.Exp(float64(v - maxv)))
		x[i] = e
		sum += e
	}
	for i := range x {
		x[i] /= sum
	}
}

func Add(dst, src []float32) {
	src = src[:len(dst)]
	for i := range dst {
		dst[i] += src[i]
	}
}

// SwiGLU computes gate = silu(gate) * up in place.
func SwiGLU(gate, up []float32) {
	up = up[:len(gate)]
	for i, g := range gate {
		gate[i] = g / (1 + float32(math.Exp(float64(-g)))) * up[i]
	}
}
