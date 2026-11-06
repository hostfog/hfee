//go:build !amd64

package ggml

var (
	kernelQ8_0   = genericQ8_0
	kernelNib32  = genericNib32
	kernelNib256 = genericNib256
	kernelQ6     = genericQ6
	kernelNib5   = genericNib5
	kernelNib5K  = genericNib5K
	dotF32       = dotGeneric
	axpyF32      = axpyGeneric
)

// SIMD reports which vector instructions the kernels use.
func SIMD() string { return "none" }
