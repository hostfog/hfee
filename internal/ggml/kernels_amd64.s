#include "textflag.h"

// Integer kernels multiply unsigned (or sign-adjusted) weight bytes with
// int8 activations using VPMADDUBSW, widen the pair sums to int32 with
// VPMADDWD against ones, then scale in float32.

// ONES16 sets reg to sixteen int16 ones.
#define ONES16(reg) \
	VPCMPEQW reg, reg, reg \
	VPSRLW   $15, reg, reg

// BYTES sets reg to imm repeated in every byte.
#define BYTES(imm, reg, xreg) \
	MOVL         imm, AX \
	VMOVD        AX, xreg \
	VPBROADCASTD xreg, reg

// HSUM reduces Y0 to a float32 in X0.
#define HSUM \
	VEXTRACTF128 $1, Y0, X1 \
	VADDPS       X1, X0, X0 \
	VHADDPS      X0, X0, X0 \
	VHADDPS      X0, X0, X0

// func q8_0AVX2(row []byte, q []int8, d []float32) float32
TEXT ·q8_0AVX2(SB), NOSPLIT, $0-76
	MOVQ row_base+0(FP), SI
	MOVQ q_base+24(FP), DI
	MOVQ d_base+48(FP), DX
	MOVQ d_len+56(FP), CX
	VXORPS Y0, Y0, Y0
	ONES16(Y15)
	TESTQ CX, CX
	JZ    q8done

q8loop:
	MOVWLZX      (SI), AX
	VMOVD        AX, X1
	VCVTPH2PS    X1, X1
	VMULSS       (DX), X1, X1
	VBROADCASTSS X1, Y1
	VMOVDQU      2(SI), Y2
	VMOVDQU      (DI), Y3
	VPSIGNB      Y2, Y2, Y4       // |w|
	VPSIGNB      Y2, Y3, Y5       // q with the sign of w
	VPMADDUBSW   Y5, Y4, Y6
	VPMADDWD     Y15, Y6, Y6
	VCVTDQ2PS    Y6, Y6
	VFMADD231PS  Y1, Y6, Y0
	ADDQ         $34, SI
	ADDQ         $32, DI
	ADDQ         $4, DX
	DECQ         CX
	JNZ          q8loop

q8done:
	HSUM
	VZEROUPPER
	MOVSS X0, ret+72(FP)
	RET

// func nib32AVX2(row []byte, q []int8, fac []float32) float32
// Q4_0: 18-byte blocks, 16 bytes of nibbles at offset 2 (low = first 16 values).
TEXT ·nib32AVX2(SB), NOSPLIT, $0-76
	MOVQ row_base+0(FP), SI
	ADDQ $2, SI
	MOVQ q_base+24(FP), DI
	MOVQ fac_base+48(FP), DX
	MOVQ fac_len+56(FP), CX
	VXORPS Y0, Y0, Y0
	ONES16(Y15)
	BYTES($0x0f0f0f0f, Y14, X14)
	TESTQ CX, CX
	JZ    n32done

n32loop:
	VMOVDQU      (SI), X2
	VPSRLW       $4, X2, X3
	VINSERTI128  $1, X3, Y2, Y2
	VPAND        Y14, Y2, Y2
	VMOVDQU      (DI), Y3
	VPMADDUBSW   Y3, Y2, Y6
	VPMADDWD     Y15, Y6, Y6
	VCVTDQ2PS    Y6, Y6
	VBROADCASTSS (DX), Y1
	VFMADD231PS  Y1, Y6, Y0
	ADDQ         $18, SI
	ADDQ         $32, DI
	ADDQ         $4, DX
	DECQ         CX
	JNZ          n32loop

n32done:
	HSUM
	VZEROUPPER
	MOVSS X0, ret+72(FP)
	RET

// func nib256AVX2(row []byte, q []int8, fac []float32) float32
// Q4_K: 144-byte super-blocks, 128 bytes of nibbles at offset 16 in four
// 32-byte chunks; low nibbles are one 32-value sub-block, high nibbles the next.
TEXT ·nib256AVX2(SB), NOSPLIT, $0-76
	MOVQ row_base+0(FP), SI
	MOVQ q_base+24(FP), DI
	MOVQ fac_base+48(FP), DX
	MOVQ fac_len+56(FP), CX
	SHRQ $3, CX
	VXORPS Y0, Y0, Y0
	VXORPS Y9, Y9, Y9
	ONES16(Y15)
	BYTES($0x0f0f0f0f, Y14, X14)
	TESTQ CX, CX
	JZ    n256done

n256super:
	LEAQ 16(SI), R8
	MOVQ $4, R9

n256chunk:
	VMOVDQU      (R8), Y2
	VPAND        Y14, Y2, Y3
	VPSRLW       $4, Y2, Y4
	VPAND        Y14, Y4, Y4
	VPMADDUBSW   (DI), Y3, Y5
	VPMADDWD     Y15, Y5, Y5
	VCVTDQ2PS    Y5, Y5
	VBROADCASTSS (DX), Y1
	VFMADD231PS  Y1, Y5, Y0
	VPMADDUBSW   32(DI), Y4, Y6
	VPMADDWD     Y15, Y6, Y6
	VCVTDQ2PS    Y6, Y6
	VBROADCASTSS 4(DX), Y1
	VFMADD231PS  Y1, Y6, Y9
	ADDQ         $32, R8
	ADDQ         $64, DI
	ADDQ         $8, DX
	DECQ         R9
	JNZ          n256chunk

	ADDQ $144, SI
	DECQ CX
	JNZ  n256super

n256done:
	VADDPS Y9, Y0, Y0
	HSUM
	VZEROUPPER
	MOVSS X0, ret+72(FP)
	RET

// Q6DOT multiplies the 32 6-bit values in vals with 32 activations at
// off(DI), scaling the lower and upper 16 by the floats at foff(DX) and foff+4(DX).
#define Q6DOT(vals, off, foff, acc) \
	VPMADDUBSW   off(DI), vals, vals \
	VPMADDWD     Y15, vals, vals \
	VCVTDQ2PS    vals, vals \
	VBROADCASTSS foff(DX), X8 \
	VBROADCASTSS foff+4(DX), X9 \
	VINSERTF128  $1, X9, Y8, Y8 \
	VFMADD231PS  Y8, vals, acc

// func q6AVX2(row []byte, q []int8, fac []float32) float32
// Q6_K: 210-byte super-blocks: ql[128], qh[64], scales[16], d.
// Each 128-value half uses 64 bytes of ql and 32 of qh:
//   v1 = ql[l]&15    | (qh[l]&0x03)<<4
//   v2 = ql[l+32]&15 | (qh[l]&0x0c)<<2
//   v3 = ql[l]>>4    | (qh[l]&0x30)
//   v4 = ql[l+32]>>4 | (qh[l]&0xc0)>>2
TEXT ·q6AVX2(SB), NOSPLIT, $0-76
	MOVQ row_base+0(FP), SI
	MOVQ q_base+24(FP), DI
	MOVQ fac_base+48(FP), DX
	MOVQ fac_len+56(FP), CX
	SHRQ $4, CX
	VXORPS Y0, Y0, Y0
	VXORPS Y7, Y7, Y7
	ONES16(Y15)
	BYTES($0x0f0f0f0f, Y14, X14)
	BYTES($0x03030303, Y13, X13)
	BYTES($0x0c0c0c0c, Y12, X12)
	BYTES($0x30303030, Y11, X11)
	BYTES($0xc0c0c0c0, Y10, X10)
	TESTQ CX, CX
	JZ    q6done

q6super:
	MOVQ SI, R8
	LEAQ 128(SI), R10
	MOVQ $2, R9

q6half:
	VMOVDQU (R8), Y2
	VMOVDQU 32(R8), Y3
	VMOVDQU (R10), Y4

	VPAND  Y14, Y2, Y5
	VPAND  Y13, Y4, Y6
	VPSLLW $4, Y6, Y6
	VPOR   Y6, Y5, Y5
	Q6DOT(Y5, 0, 0, Y0)

	VPAND  Y14, Y3, Y5
	VPAND  Y12, Y4, Y6
	VPSLLW $2, Y6, Y6
	VPOR   Y6, Y5, Y5
	Q6DOT(Y5, 32, 8, Y7)

	VPSRLW $4, Y2, Y5
	VPAND  Y14, Y5, Y5
	VPAND  Y11, Y4, Y6
	VPOR   Y6, Y5, Y5
	Q6DOT(Y5, 64, 16, Y0)

	VPSRLW $4, Y3, Y5
	VPAND  Y14, Y5, Y5
	VPAND  Y10, Y4, Y6
	VPSRLW $2, Y6, Y6
	VPOR   Y6, Y5, Y5
	Q6DOT(Y5, 96, 24, Y7)

	ADDQ $64, R8
	ADDQ $32, R10
	ADDQ $128, DI
	ADDQ $32, DX
	DECQ R9
	JNZ  q6half

	ADDQ $210, SI
	DECQ CX
	JNZ  q6super

q6done:
	VADDPS Y7, Y0, Y0
	HSUM
	VZEROUPPER
	MOVSS X0, ret+72(FP)
	RET

// func dotAVX2(a, b []float32) float32
TEXT ·dotAVX2(SB), NOSPLIT, $0-52
	MOVQ a_base+0(FP), SI
	MOVQ b_base+24(FP), DI
	MOVQ a_len+8(FP), CX
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

dot32:
	CMPQ        CX, $32
	JL          dot8
	VMOVUPS     (SI), Y4
	VMOVUPS     32(SI), Y5
	VMOVUPS     64(SI), Y6
	VMOVUPS     96(SI), Y8
	VFMADD231PS (DI), Y4, Y0
	VFMADD231PS 32(DI), Y5, Y1
	VFMADD231PS 64(DI), Y6, Y2
	VFMADD231PS 96(DI), Y8, Y3
	ADDQ        $128, SI
	ADDQ        $128, DI
	SUBQ        $32, CX
	JMP         dot32

dot8:
	CMPQ        CX, $8
	JL          dotreduce
	VMOVUPS     (SI), Y4
	VFMADD231PS (DI), Y4, Y0
	ADDQ        $32, SI
	ADDQ        $32, DI
	SUBQ        $8, CX
	JMP         dot8

dotreduce:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0
	HSUM

dottail:
	TESTQ       CX, CX
	JZ          dotdone
	VMOVSS      (SI), X4
	VFMADD231SS (DI), X4, X0
	ADDQ        $4, SI
	ADDQ        $4, DI
	DECQ        CX
	JMP         dottail

dotdone:
	VZEROUPPER
	MOVSS X0, ret+48(FP)
	RET

// func axpyAVX2(y []float32, alpha float32, x []float32)
TEXT ·axpyAVX2(SB), NOSPLIT, $0-56
	MOVQ         y_base+0(FP), DI
	MOVQ         y_len+8(FP), CX
	VBROADCASTSS alpha+24(FP), Y0
	MOVQ         x_base+32(FP), SI

axpy8:
	CMPQ        CX, $8
	JL          axpytail
	VMOVUPS     (DI), Y1
	VFMADD231PS (SI), Y0, Y1
	VMOVUPS     Y1, (DI)
	ADDQ        $32, SI
	ADDQ        $32, DI
	SUBQ        $8, CX
	JMP         axpy8

axpytail:
	TESTQ       CX, CX
	JZ          axpydone
	VMOVSS      (DI), X1
	VFMADD231SS (SI), X0, X1
	VMOVSS      X1, (DI)
	ADDQ        $4, SI
	ADDQ        $4, DI
	DECQ        CX
	JMP         axpytail

axpydone:
	VZEROUPPER
	RET

// func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL eaxArg+0(FP), AX
	MOVL ecxArg+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbv() (eax uint32)
TEXT ·xgetbv(SB), NOSPLIT, $0-4
	MOVL   $0, CX
	XGETBV
	MOVL   AX, eax+0(FP)
	RET

// Byte i of q5spread holds the qh byte that carries bit i of a 32-bit mask;
// q5bits has every bit except bit i%8 set, so OR + compare finds bit i.
DATA q5spread<>+0(SB)/8, $0x0000000000000000
DATA q5spread<>+8(SB)/8, $0x0101010101010101
DATA q5spread<>+16(SB)/8, $0x0202020202020202
DATA q5spread<>+24(SB)/8, $0x0303030303030303
GLOBL q5spread<>(SB), RODATA|NOPTR, $32

DATA q5bits<>+0(SB)/8, $0x7fbfdfeff7fbfdfe
DATA q5bits<>+8(SB)/8, $0x7fbfdfeff7fbfdfe
DATA q5bits<>+16(SB)/8, $0x7fbfdfeff7fbfdfe
DATA q5bits<>+24(SB)/8, $0x7fbfdfeff7fbfdfe
GLOBL q5bits<>(SB), RODATA|NOPTR, $32

// func nib5AVX2(row []byte, q []int8, fac []float32) float32
// Q5_0: 22-byte blocks: d, qh (32 high bits), 16 bytes of nibbles.
TEXT ·nib5AVX2(SB), NOSPLIT, $0-76
	MOVQ row_base+0(FP), SI
	MOVQ q_base+24(FP), DI
	MOVQ fac_base+48(FP), DX
	MOVQ fac_len+56(FP), CX
	VXORPS Y0, Y0, Y0
	ONES16(Y15)
	BYTES($0x0f0f0f0f, Y14, X14)
	BYTES($0x10101010, Y13, X13)
	VMOVDQU q5spread<>(SB), Y12
	VMOVDQU q5bits<>(SB), Y11
	VPCMPEQB Y10, Y10, Y10
	TESTQ CX, CX
	JZ    n5done

n5loop:
	VPBROADCASTD 2(SI), Y4         // qh in every dword
	VPSHUFB      Y12, Y4, Y4
	VPOR         Y11, Y4, Y4
	VPCMPEQB     Y10, Y4, Y4       // 0xff where the high bit is set
	VPAND        Y13, Y4, Y4       // -> 16

	VMOVDQU      6(SI), X2
	VPSRLW       $4, X2, X3
	VINSERTI128  $1, X3, Y2, Y2
	VPAND        Y14, Y2, Y2
	VPOR         Y4, Y2, Y2

	VPMADDUBSW   (DI), Y2, Y6
	VPMADDWD     Y15, Y6, Y6
	VCVTDQ2PS    Y6, Y6
	VBROADCASTSS (DX), Y1
	VFMADD231PS  Y1, Y6, Y0
	ADDQ         $22, SI
	ADDQ         $32, DI
	ADDQ         $4, DX
	DECQ         CX
	JNZ          n5loop

n5done:
	HSUM
	VZEROUPPER
	MOVSS X0, ret+72(FP)
	RET

// func nib5KAVX2(row []byte, q []int8, fac []float32) float32
// Q5_K: 176-byte super-blocks: d, dmin, scales[12], qh[32], qs[128].
// Chunk c uses bit 2c of qh for its low nibbles and bit 2c+1 for its high ones.
TEXT ·nib5KAVX2(SB), NOSPLIT, $0-76
	MOVQ row_base+0(FP), SI
	MOVQ q_base+24(FP), DI
	MOVQ fac_base+48(FP), DX
	MOVQ fac_len+56(FP), CX
	SHRQ $3, CX
	VXORPS Y0, Y0, Y0
	VXORPS Y9, Y9, Y9
	ONES16(Y15)
	BYTES($0x0f0f0f0f, Y14, X14)
	BYTES($0x10101010, Y13, X13)
	TESTQ CX, CX
	JZ    n5kdone

n5ksuper:
	VMOVDQU 16(SI), Y8                 // qh
	LEAQ    48(SI), R8
	BYTES($0x01010101, Y12, X12)       // bit for low nibbles
	BYTES($0x02020202, Y11, X11)       // bit for high nibbles
	MOVQ    $4, R9

n5kchunk:
	VMOVDQU    (R8), Y2
	VPAND      Y14, Y2, Y3
	VPSRLW     $4, Y2, Y4
	VPAND      Y14, Y4, Y4

	VPAND      Y12, Y8, Y5
	VPCMPEQB   Y12, Y5, Y5
	VPAND      Y13, Y5, Y5
	VPOR       Y5, Y3, Y3
	VPAND      Y11, Y8, Y6
	VPCMPEQB   Y11, Y6, Y6
	VPAND      Y13, Y6, Y6
	VPOR       Y6, Y4, Y4

	VPMADDUBSW   (DI), Y3, Y5
	VPMADDWD     Y15, Y5, Y5
	VCVTDQ2PS    Y5, Y5
	VBROADCASTSS (DX), Y1
	VFMADD231PS  Y1, Y5, Y0
	VPMADDUBSW   32(DI), Y4, Y6
	VPMADDWD     Y15, Y6, Y6
	VCVTDQ2PS    Y6, Y6
	VBROADCASTSS 4(DX), Y1
	VFMADD231PS  Y1, Y6, Y9

	VPSLLW $2, Y12, Y12
	VPSLLW $2, Y11, Y11
	ADDQ   $32, R8
	ADDQ   $64, DI
	ADDQ   $8, DX
	DECQ   R9
	JNZ    n5kchunk

	ADDQ $176, SI
	DECQ CX
	JNZ  n5ksuper

n5kdone:
	VADDPS Y9, Y0, Y0
	HSUM
	VZEROUPPER
	MOVSS X0, ret+72(FP)
	RET
