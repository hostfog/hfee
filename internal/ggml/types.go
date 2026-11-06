// Package ggml implements the GGML tensor types used by GGUF files:
// dequantization kernels and the matrix operations the model needs.
package ggml

import "fmt"

type Type uint32

const (
	F32  Type = 0
	F16  Type = 1
	Q4_0 Type = 2
	Q4_1 Type = 3
	Q5_0 Type = 6
	Q5_1 Type = 7
	Q8_0 Type = 8
	Q2_K Type = 10
	Q3_K Type = 11
	Q4_K Type = 12
	Q5_K Type = 13
	Q6_K Type = 14
	BF16 Type = 30
)

const qk = 256 // super-block size for k-quants

type traits struct {
	name    string
	block   int // elements per block
	size    int // bytes per block
	dequant func(dst []float32, src []byte)
}

var typeTraits = [...]traits{
	F32:  {"F32", 1, 4, dequantF32},
	F16:  {"F16", 1, 2, dequantF16},
	BF16: {"BF16", 1, 2, dequantBF16},
	Q4_0: {"Q4_0", 32, 18, dequantQ4_0},
	Q4_1: {"Q4_1", 32, 20, dequantQ4_1},
	Q5_0: {"Q5_0", 32, 22, dequantQ5_0},
	Q5_1: {"Q5_1", 32, 24, dequantQ5_1},
	Q8_0: {"Q8_0", 32, 34, dequantQ8_0},
	Q2_K: {"Q2_K", qk, 84, dequantQ2K},
	Q3_K: {"Q3_K", qk, 110, dequantQ3K},
	Q4_K: {"Q4_K", qk, 144, dequantQ4K},
	Q5_K: {"Q5_K", qk, 176, dequantQ5K},
	Q6_K: {"Q6_K", qk, 210, dequantQ6K},
}

func (t Type) String() string {
	if t.Supported() {
		return typeTraits[t].name
	}
	return fmt.Sprintf("type(%d)", uint32(t))
}

func (t Type) Supported() bool {
	return int(t) < len(typeTraits) && typeTraits[t].dequant != nil
}

// ByteSize returns the number of bytes needed to store n elements of type t.
func (t Type) ByteSize(n int) int {
	tr := typeTraits[t]
	return n / tr.block * tr.size
}

// BlockSize returns how many elements one block of type t holds.
func (t Type) BlockSize() int {
	return typeTraits[t].block
}

// Dequantize decodes len(dst) elements from src.
func Dequantize(t Type, dst []float32, src []byte) {
	typeTraits[t].dequant(dst, src)
}
