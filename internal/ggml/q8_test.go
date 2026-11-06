package ggml

import (
	"math"
	"math/rand"
	"testing"
)

func randomRow(rng *rand.Rand, typ Type, n int) []byte {
	row := make([]byte, typ.ByteSize(n))
	rng.Read(row)
	fixScales(typ, row)
	return row
}

func randomVec(rng *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	return x
}

func close(a, b, tol float32) bool {
	return math.Abs(float64(a-b)) <= float64(tol)*(1+math.Abs(float64(b)))
}

// The integer path quantises activations to int8, so it is compared with
// the float result loosely; the AVX2 kernels must match the portable ones closely.
func TestIntDotsMatchFloat(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const n = 1024

	for typ, dot := range intDots {
		row := randomRow(rng, typ, n)
		x := randomVec(rng, n)

		w := make([]float32, n)
		Dequantize(typ, w, row)
		want := dotGeneric(w, x)

		q := NewQ8(n)
		q.Quantize(x)
		got := dot(row, q, make([]float32, n/16))

		// relative to the size of the terms, since the sum itself can be near zero
		var scale float64
		for i := range w {
			scale += math.Abs(float64(w[i] * x[i]))
		}
		if math.Abs(float64(got-want)) > 0.01*scale {
			t.Errorf("%v: integer dot = %v, float dot = %v", typ, got, want)
		}
	}
}

func TestKernelsMatchPortable(t *testing.T) {
	if SIMD() == "none" {
		t.Skip("no SIMD on this machine")
	}
	rng := rand.New(rand.NewSource(3))
	const n = 2048

	q := NewQ8(n)
	q.Quantize(randomVec(rng, n))
	fac32 := randomVec(rng, n/32)
	fac16 := randomVec(rng, n/16)

	tests := []struct {
		name           string
		fast, portable func([]byte, []int8, []float32) float32
		typ            Type
		fac            []float32
	}{
		{"Q8_0", kernelQ8_0, genericQ8_0, Q8_0, q.D},
		{"Q4_0", kernelNib32, genericNib32, Q4_0, fac32},
		{"Q4_K", kernelNib256, genericNib256, Q4_K, fac32},
		{"Q6_K", kernelQ6, genericQ6, Q6_K, fac16},
		{"Q5_0", kernelNib5, genericNib5, Q5_0, fac32},
		{"Q5_K", kernelNib5K, genericNib5K, Q5_K, fac32},
	}
	for _, tt := range tests {
		row := randomRow(rng, tt.typ, n)
		got, want := tt.fast(row, q.Q, tt.fac), tt.portable(row, q.Q, tt.fac)
		if !close(got, want, 1e-4) {
			t.Errorf("%s: SIMD = %v, portable = %v", tt.name, got, want)
		}
	}

	for _, size := range []int{1, 7, 8, 33, 100, 1027} {
		a, b := randomVec(rng, size), randomVec(rng, size)
		if got, want := dotF32(a, b), dotGeneric(a, b); !close(got, want, 1e-4) {
			t.Errorf("dot(%d) = %v, want %v", size, got, want)
		}

		y1, y2 := randomVec(rng, size), make([]float32, size)
		copy(y2, y1)
		axpyF32(y1, 0.5, b)
		axpyGeneric(y2, 0.5, b)
		for i := range y1 {
			if !close(y1[i], y2[i], 1e-6) {
				t.Fatalf("axpy(%d)[%d] = %v, want %v", size, i, y1[i], y2[i])
			}
		}
	}
}

func BenchmarkKernels(b *testing.B) {
	rng := rand.New(rand.NewSource(4))
	const n = 4096
	q := NewQ8(n)
	q.Quantize(randomVec(rng, n))
	scratch := make([]float32, n/16)

	for _, typ := range []Type{Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K} {
		row := randomRow(rng, typ, n)
		b.Run(typ.String(), func(b *testing.B) {
			for range b.N {
				intDots[typ](row, q, scratch)
			}
		})
	}
}
