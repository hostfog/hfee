package ggml

import "testing"

func TestHalfToFloat(t *testing.T) {
	tests := map[uint16]float32{
		0x0000: 0,
		0x3c00: 1,
		0xc000: -2,
		0x3800: 0.5,
		0x0001: 5.9604645e-8, // smallest subnormal
		0x7bff: 65504,
	}
	for in, want := range tests {
		if got := HalfToFloat(in); got != want {
			t.Errorf("HalfToFloat(%#04x) = %v, want %v", in, got, want)
		}
	}
}

// fixScales overwrites each block's float16 scales with 1.0 so random bytes
// never produce NaN or Inf.
func fixScales(typ Type, row []byte) {
	tr := typeTraits[typ]
	offsets := map[Type][]int{Q4_0: {0}, Q5_0: {0}, Q8_0: {0}, Q4_K: {0, 2}, Q5_K: {0, 2}, Q6_K: {208}}[typ]
	for b := 0; b < len(row); b += tr.size {
		for _, o := range offsets {
			row[b+o], row[b+o+1] = 0x00, 0x3c
		}
	}
}
