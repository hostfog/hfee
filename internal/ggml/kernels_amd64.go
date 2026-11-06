package ggml

// AVX2 kernels, implemented in kernels_amd64.s.

//go:noescape
func q8_0AVX2(row []byte, q []int8, d []float32) float32

//go:noescape
func nib32AVX2(row []byte, q []int8, fac []float32) float32

//go:noescape
func nib256AVX2(row []byte, q []int8, fac []float32) float32

//go:noescape
func q6AVX2(row []byte, q []int8, fac []float32) float32

//go:noescape
func nib5AVX2(row []byte, q []int8, fac []float32) float32

//go:noescape
func nib5KAVX2(row []byte, q []int8, fac []float32) float32

//go:noescape
func dotAVX2(a, b []float32) float32

//go:noescape
func axpyAVX2(y []float32, alpha float32, x []float32)

func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
func xgetbv() (eax uint32)

var hasAVX2 = func() bool {
	if max, _, _, _ := cpuid(0, 0); max < 7 {
		return false
	}
	_, _, c1, _ := cpuid(1, 0)
	const fma, osxsave, avx, f16c = 1 << 12, 1 << 27, 1 << 28, 1 << 29
	if c1&(fma|osxsave|avx|f16c) != fma|osxsave|avx|f16c {
		return false
	}
	// the OS must save YMM registers on context switches
	if xgetbv()&6 != 6 {
		return false
	}
	_, b7, _, _ := cpuid(7, 0)
	return b7&(1<<5) != 0
}()

var (
	kernelQ8_0   = pick(q8_0AVX2, genericQ8_0)
	kernelNib32  = pick(nib32AVX2, genericNib32)
	kernelNib256 = pick(nib256AVX2, genericNib256)
	kernelQ6     = pick(q6AVX2, genericQ6)
	kernelNib5   = pick(nib5AVX2, genericNib5)
	kernelNib5K  = pick(nib5KAVX2, genericNib5K)
	dotF32       = pick(dotAVX2, dotGeneric)
	axpyF32      = pick(axpyAVX2, axpyGeneric)
)

func pick[F any](fast, portable F) F {
	if hasAVX2 {
		return fast
	}
	return portable
}

// SIMD reports which vector instructions the kernels use.
func SIMD() string {
	if hasAVX2 {
		return "AVX2"
	}
	return "none"
}
