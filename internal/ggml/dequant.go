package ggml

import (
	"encoding/binary"
	"math"
)

// half holds every float16 value converted to float32; int8f every signed byte.
var (
	half  [1 << 16]float32
	int8f [256]float32
)

func init() {
	for i := range half {
		half[i] = HalfToFloat(uint16(i))
	}
	for i := range int8f {
		int8f[i] = float32(int8(i))
	}
}

func f16(b []byte) float32 {
	return half[binary.LittleEndian.Uint16(b)]
}

// HalfToFloat converts an IEEE 754 half-precision value to float32.
func HalfToFloat(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff

	switch {
	case exp == 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	case exp != 0:
		return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
	case mant == 0:
		return math.Float32frombits(sign)
	}

	// subnormal: normalise the mantissa
	e := uint32(113)
	for mant&0x400 == 0 {
		mant <<= 1
		e--
	}
	return math.Float32frombits(sign | e<<23 | (mant&0x3ff)<<13)
}

func dequantF32(dst []float32, src []byte) {
	for i := range dst {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
	}
}

func dequantF16(dst []float32, src []byte) {
	for i := range dst {
		dst[i] = f16(src[i*2:])
	}
}

func dequantBF16(dst []float32, src []byte) {
	for i := range dst {
		dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(src[i*2:])) << 16)
	}
}

func dequantQ4_0(dst []float32, src []byte) {
	for b := 0; b < len(dst)/32; b++ {
		blk := src[b*18:]
		y := dst[b*32:]
		d := f16(blk)
		qs := blk[2:18]
		for j := 0; j < 16; j++ {
			y[j] = float32(int(qs[j]&0x0f)-8) * d
			y[j+16] = float32(int(qs[j]>>4)-8) * d
		}
	}
}

func dequantQ4_1(dst []float32, src []byte) {
	for b := 0; b < len(dst)/32; b++ {
		blk := src[b*20:]
		y := dst[b*32:]
		d, m := f16(blk), f16(blk[2:])
		qs := blk[4:20]
		for j := 0; j < 16; j++ {
			y[j] = float32(qs[j]&0x0f)*d + m
			y[j+16] = float32(qs[j]>>4)*d + m
		}
	}
}

func dequantQ5_0(dst []float32, src []byte) {
	for b := 0; b < len(dst)/32; b++ {
		blk := src[b*22:]
		y := dst[b*32:]
		d := f16(blk)
		qh := binary.LittleEndian.Uint32(blk[2:])
		qs := blk[6:22]
		for j := 0; j < 16; j++ {
			h0 := byte((qh>>j)<<4) & 0x10
			h1 := byte(qh>>(j+12)) & 0x10
			y[j] = float32(int(qs[j]&0x0f|h0)-16) * d
			y[j+16] = float32(int(qs[j]>>4|h1)-16) * d
		}
	}
}

func dequantQ5_1(dst []float32, src []byte) {
	for b := 0; b < len(dst)/32; b++ {
		blk := src[b*24:]
		y := dst[b*32:]
		d, m := f16(blk), f16(blk[2:])
		qh := binary.LittleEndian.Uint32(blk[4:])
		qs := blk[8:24]
		for j := 0; j < 16; j++ {
			h0 := byte((qh>>j)<<4) & 0x10
			h1 := byte(qh>>(j+12)) & 0x10
			y[j] = float32(qs[j]&0x0f|h0)*d + m
			y[j+16] = float32(qs[j]>>4|h1)*d + m
		}
	}
}

func dequantQ8_0(dst []float32, src []byte) {
	for b := 0; b < len(dst)/32; b++ {
		blk := src[b*34:]
		y := dst[b*32:]
		d := f16(blk)
		for j := 0; j < 32; j++ {
			y[j] = float32(int8(blk[2+j])) * d
		}
	}
}

// scaleMinK4 unpacks the 6-bit scale and min for sub-block j of a Q4_K/Q5_K block.
func scaleMinK4(j int, q []byte) (float32, float32) {
	if j < 4 {
		return float32(q[j] & 63), float32(q[j+4] & 63)
	}
	d := q[j+4]&0x0f | (q[j-4]>>6)<<4
	m := q[j+4]>>4 | (q[j]>>6)<<4
	return float32(d), float32(m)
}

func dequantQ2K(dst []float32, src []byte) {
	for b := 0; b < len(dst)/qk; b++ {
		blk := src[b*84:]
		y := dst[b*qk:]
		scales := blk[:16]
		q := blk[16:80]
		d, dmin := f16(blk[80:]), f16(blk[82:])

		is, out := 0, 0
		for n := 0; n < qk; n += 128 {
			for shift := uint(0); shift < 8; shift += 2 {
				for half := 0; half < 2; half++ {
					sc := scales[is]
					is++
					dl := d * float32(sc&0x0f)
					ml := dmin * float32(sc>>4)
					for l := 0; l < 16; l++ {
						y[out] = dl*float32((q[half*16+l]>>shift)&3) - ml
						out++
					}
				}
			}
			q = q[32:]
		}
	}
}

func dequantQ3K(dst []float32, src []byte) {
	const kmask1, kmask2 = 0x03030303, 0x0f0f0f0f
	var aux [4]uint32
	var scales [16]int8

	for b := 0; b < len(dst)/qk; b++ {
		blk := src[b*110:]
		y := dst[b*qk:]
		hm := blk[:32]
		q := blk[32:96]
		raw := blk[96:108]
		d := f16(blk[108:])

		aux[0] = binary.LittleEndian.Uint32(raw[0:])
		aux[1] = binary.LittleEndian.Uint32(raw[4:])
		tmp := binary.LittleEndian.Uint32(raw[8:])
		aux[2] = (aux[0]>>4)&kmask2 | ((tmp>>4)&kmask1)<<4
		aux[3] = (aux[1]>>4)&kmask2 | ((tmp>>6)&kmask1)<<4
		aux[0] = aux[0]&kmask2 | (tmp&kmask1)<<4
		aux[1] = aux[1]&kmask2 | ((tmp>>2)&kmask1)<<4
		for i := 0; i < 16; i++ {
			scales[i] = int8(aux[i/4] >> (8 * (i % 4)))
		}

		is, out := 0, 0
		m := byte(1)
		for n := 0; n < qk; n += 128 {
			for shift := uint(0); shift < 8; shift += 2 {
				for half := 0; half < 2; half++ {
					dl := d * float32(int(scales[is])-32)
					is++
					for l := 0; l < 16; l++ {
						k := half*16 + l
						v := int((q[k] >> shift) & 3)
						if hm[k]&m == 0 {
							v -= 4
						}
						y[out] = dl * float32(v)
						out++
					}
				}
				m <<= 1
			}
			q = q[32:]
		}
	}
}

func dequantQ4K(dst []float32, src []byte) {
	for b := 0; b < len(dst)/qk; b++ {
		blk := src[b*144:]
		y := dst[b*qk:]
		d, dmin := f16(blk), f16(blk[2:])
		scales := blk[4:16]
		q := blk[16:144]

		for j, is := 0, 0; j < qk; j, is = j+64, is+2 {
			sc1, m1 := scaleMinK4(is, scales)
			sc2, m2 := scaleMinK4(is+1, scales)
			d1, min1 := d*sc1, dmin*m1
			d2, min2 := d*sc2, dmin*m2
			for l := 0; l < 32; l++ {
				y[j+l] = d1*float32(q[l]&0x0f) - min1
				y[j+32+l] = d2*float32(q[l]>>4) - min2
			}
			q = q[32:]
		}
	}
}

func dequantQ5K(dst []float32, src []byte) {
	for b := 0; b < len(dst)/qk; b++ {
		blk := src[b*176:]
		y := dst[b*qk:]
		d, dmin := f16(blk), f16(blk[2:])
		scales := blk[4:16]
		qh := blk[16:48]
		ql := blk[48:176]

		u1, u2 := byte(1), byte(2)
		for j, is := 0, 0; j < qk; j, is = j+64, is+2 {
			sc1, m1 := scaleMinK4(is, scales)
			sc2, m2 := scaleMinK4(is+1, scales)
			d1, min1 := d*sc1, dmin*m1
			d2, min2 := d*sc2, dmin*m2
			for l := 0; l < 32; l++ {
				lo, hi := float32(ql[l]&0x0f), float32(ql[l]>>4)
				if qh[l]&u1 != 0 {
					lo += 16
				}
				if qh[l]&u2 != 0 {
					hi += 16
				}
				y[j+l] = d1*lo - min1
				y[j+32+l] = d2*hi - min2
			}
			ql = ql[32:]
			u1 <<= 2
			u2 <<= 2
		}
	}
}

func dequantQ6K(dst []float32, src []byte) {
	for b := 0; b < len(dst)/qk; b++ {
		blk := src[b*210:]
		y := dst[b*qk:]
		ql := blk[:128]
		qh := blk[128:192]
		sc := blk[192:208]
		d := f16(blk[208:])

		for n := 0; n < qk; n += 128 {
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int(ql[l]&0x0f|(qh[l]&3)<<4) - 32
				q2 := int(ql[l+32]&0x0f|((qh[l]>>2)&3)<<4) - 32
				q3 := int(ql[l]>>4|((qh[l]>>4)&3)<<4) - 32
				q4 := int(ql[l+32]>>4|((qh[l]>>6)&3)<<4) - 32
				y[n+l] = d * float32(int8(sc[is])) * float32(q1)
				y[n+l+32] = d * float32(int8(sc[is+2])) * float32(q2)
				y[n+l+64] = d * float32(int8(sc[is+4])) * float32(q3)
				y[n+l+96] = d * float32(int8(sc[is+6])) * float32(q4)
			}
			ql = ql[64:]
			qh = qh[32:]
			sc = sc[8:]
		}
	}
}
