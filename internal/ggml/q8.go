package ggml

import (
	"encoding/binary"
	"math"
)

// Q8 is a vector quantised to int8 in blocks of 32, the form the integer
// kernels multiply weights against. Sum16/Sum32 hold the integer sums of
// each 16/32-element run, used to apply weight offsets and minimums.
type Q8 struct {
	Q     []int8
	D     []float32 // scale per 32 elements
	Sum16 []int32
	Sum32 []int32
}

func NewQ8(n int) *Q8 {
	return &Q8{
		Q:     make([]int8, n),
		D:     make([]float32, n/32),
		Sum16: make([]int32, n/16),
		Sum32: make([]int32, n/32),
	}
}

// Quantize fills q from x; len(x) must equal len(q.Q).
func (q *Q8) Quantize(x []float32) {
	for b := range q.D {
		xb := x[b*32 : b*32+32]
		var amax float32
		for _, v := range xb {
			amax = max(amax, float32(math.Abs(float64(v))))
		}

		d := amax / 127
		inv := float32(0)
		if d > 0 {
			inv = 1 / d
		}
		q.D[b] = d

		qb := q.Q[b*32 : b*32+32]
		var s0, s1 int32
		for i, v := range xb {
			r := int8(math.Round(float64(v * inv)))
			qb[i] = r
			if i < 16 {
				s0 += int32(r)
			} else {
				s1 += int32(r)
			}
		}
		q.Sum16[2*b], q.Sum16[2*b+1] = s0, s1
		q.Sum32[b] = s0 + s1
	}
}

// intDots are the weight types with integer kernels.
var intDots = map[Type]func(row []byte, a *Q8, scratch []float32) float32{
	Q8_0: intDotQ8_0,
	Q4_0: intDotQ4_0,
	Q5_0: intDotQ5_0,
	Q4_K: intDotQ4K,
	Q5_K: intDotQ5K,
	Q6_K: intDotQ6K,
}

func intDotQ8_0(row []byte, a *Q8, _ []float32) float32 {
	return kernelQ8_0(row, a.Q, a.D)
}

func intDotQ4_0(row []byte, a *Q8, fac []float32) float32 {
	nb := len(a.D)
	fac = fac[:nb]
	var corr float32
	for b := range fac {
		fac[b] = f16(row[b*18:]) * a.D[b]
		corr += fac[b] * float32(a.Sum32[b])
	}
	// weights are nibbles minus 8
	return kernelNib32(row, a.Q, fac) - 8*corr
}

func intDotQ5_0(row []byte, a *Q8, fac []float32) float32 {
	nb := len(a.D)
	fac = fac[:nb]
	var corr float32
	for b := range fac {
		fac[b] = f16(row[b*22:]) * a.D[b]
		corr += fac[b] * float32(a.Sum32[b])
	}
	// weights are 5-bit values minus 16
	return kernelNib5(row, a.Q, fac) - 16*corr
}

// kScales fills fac with d·scale·d_a per 32-value sub-block of Q4_K/Q5_K rows
// and returns the minimum correction. size is the super-block size in bytes.
func kScales(row []byte, size int, a *Q8, fac []float32) float32 {
	var corr float32
	for s := 0; s < len(fac)/8; s++ {
		blk := row[s*size:]
		d, dmin := f16(blk), f16(blk[2:])
		scales := blk[4:16]
		for j := 0; j < 8; j++ {
			sc, m := scaleMinK4(j, scales)
			i := s*8 + j
			fac[i] = d * sc * a.D[i]
			corr += dmin * m * a.D[i] * float32(a.Sum32[i])
		}
	}
	return corr
}

func intDotQ5K(row []byte, a *Q8, fac []float32) float32 {
	fac = fac[:len(a.D)]
	corr := kScales(row, 176, a, fac)
	return kernelNib5K(row, a.Q, fac) - corr
}

func intDotQ4K(row []byte, a *Q8, fac []float32) float32 {
	fac = fac[:len(a.D)]
	corr := kScales(row, 144, a, fac)
	return kernelNib256(row, a.Q, fac) - corr
}

func intDotQ6K(row []byte, a *Q8, fac []float32) float32 {
	ng := len(a.Sum16)
	fac = fac[:ng]
	var corr float32
	for s := 0; s < ng/16; s++ {
		blk := row[s*210:]
		d := f16(blk[208:])
		sc := blk[192:208]
		for g := 0; g < 16; g++ {
			i := s*16 + g
			fac[i] = d * int8f[sc[g]] * a.D[i/2]
			corr += fac[i] * float32(a.Sum16[i])
		}
	}
	// weights are 6-bit values minus 32
	return kernelQ6(row, a.Q, fac) - 32*corr
}

// Portable kernels. The amd64 build replaces them with AVX2 versions when
// the CPU supports it; tests compare the two.

// genericQ8_0 returns Σ_b d_w·d_a·Σ w·q over blocks of 34 bytes (f16 scale + 32 int8).
func genericQ8_0(row []byte, q []int8, d []float32) float32 {
	var acc float32
	for b := range d {
		blk := row[b*34 : b*34+34]
		qb := q[b*32 : b*32+32]
		var s int32
		for i := 0; i < 32; i++ {
			s += int32(int8(blk[2+i])) * int32(qb[i])
		}
		acc += f16(blk) * d[b] * float32(s)
	}
	return acc
}

// genericNib32 returns Σ_b fac_b·Σ nibble·q for Q4_0 blocks (18 bytes, nibbles at offset 2).
func genericNib32(row []byte, q []int8, fac []float32) float32 {
	var acc float32
	for b := range fac {
		qs := row[b*18+2 : b*18+18]
		qb := q[b*32 : b*32+32]
		var s int32
		for j := 0; j < 16; j++ {
			s += int32(qs[j]&0x0f)*int32(qb[j]) + int32(qs[j]>>4)*int32(qb[j+16])
		}
		acc += fac[b] * float32(s)
	}
	return acc
}

// genericNib256 returns Σ_i fac_i·Σ nibble·q over the 32-element sub-blocks of Q4_K super-blocks.
func genericNib256(row []byte, q []int8, fac []float32) float32 {
	var acc float32
	for s := 0; s < len(fac)/8; s++ {
		qs := row[s*144+16 : s*144+144]
		for c := 0; c < 4; c++ {
			chunk := qs[c*32 : c*32+32]
			lo := q[s*256+c*64 : s*256+c*64+32]
			hi := q[s*256+c*64+32 : s*256+c*64+64]
			var sl, sh int32
			for l := 0; l < 32; l++ {
				sl += int32(chunk[l]&0x0f) * int32(lo[l])
				sh += int32(chunk[l]>>4) * int32(hi[l])
			}
			acc += fac[s*8+2*c]*float32(sl) + fac[s*8+2*c+1]*float32(sh)
		}
	}
	return acc
}

// genericQ6 returns Σ_g fac_g·Σ v·q over 16-element groups, v being the unsigned 6-bit Q6_K values.
func genericQ6(row []byte, q []int8, fac []float32) float32 {
	var acc float32
	for s := 0; s < len(fac)/16; s++ {
		blk := row[s*210:]
		for n := 0; n < 2; n++ {
			ql := blk[n*64 : n*64+64]
			qh := blk[128+n*32 : 128+n*32+32]
			base := s*256 + n*128
			var sums [8]int32 // groups of 16 within this 128 chunk
			for l := 0; l < 32; l++ {
				h := qh[l]
				v1 := int32(ql[l]&0x0f | (h&3)<<4)
				v2 := int32(ql[l+32]&0x0f | (h>>2&3)<<4)
				v3 := int32(ql[l]>>4 | (h>>4&3)<<4)
				v4 := int32(ql[l+32]>>4 | (h>>6&3)<<4)
				g := l / 16
				sums[g] += v1 * int32(q[base+l])
				sums[g+2] += v2 * int32(q[base+l+32])
				sums[g+4] += v3 * int32(q[base+l+64])
				sums[g+6] += v4 * int32(q[base+l+96])
			}
			for g, sum := range sums {
				acc += fac[s*16+n*8+g] * float32(sum)
			}
		}
	}
	return acc
}

// genericNib5 returns Σ_b fac_b·Σ v·q for Q5_0 blocks (22 bytes: d, qh, 16 bytes of nibbles), v = nibble | bit<<4.
func genericNib5(row []byte, q []int8, fac []float32) float32 {
	var acc float32
	for b := range fac {
		blk := row[b*22 : b*22+22]
		qh := binary.LittleEndian.Uint32(blk[2:6])
		qs := blk[6:22]
		qb := q[b*32 : b*32+32]
		var s int32
		for j := 0; j < 16; j++ {
			lo := int32(qs[j]&0x0f) | int32(qh>>j&1)<<4
			hi := int32(qs[j]>>4) | int32(qh>>(j+16)&1)<<4
			s += lo*int32(qb[j]) + hi*int32(qb[j+16])
		}
		acc += fac[b] * float32(s)
	}
	return acc
}

// genericNib5K returns Σ_i fac_i·Σ v·q over the sub-blocks of Q5_K super-blocks.
func genericNib5K(row []byte, q []int8, fac []float32) float32 {
	var acc float32
	for s := 0; s < len(fac)/8; s++ {
		blk := row[s*176:]
		qh := blk[16:48]
		qs := blk[48:176]
		for c := 0; c < 4; c++ {
			chunk := qs[c*32 : c*32+32]
			lo := q[s*256+c*64 : s*256+c*64+32]
			hi := q[s*256+c*64+32 : s*256+c*64+64]
			var sl, sh int32
			for l := 0; l < 32; l++ {
				h := qh[l] >> (2 * c)
				sl += int32(chunk[l]&0x0f|(h&1)<<4) * int32(lo[l])
				sh += int32(chunk[l]>>4|(h&2)<<3) * int32(hi[l])
			}
			acc += fac[s*8+2*c]*float32(sl) + fac[s*8+2*c+1]*float32(sh)
		}
	}
	return acc
}
