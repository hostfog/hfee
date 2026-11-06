// Package gguf reads model files in the GGUF format.
package gguf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/hostfog/hfee/internal/ggml"
)

const magic = 0x46554747 // "GGUF"

type Tensor struct {
	Name string
	Type ggml.Type
	Dims []int // Dims[0] is the innermost (contiguous) dimension
	Data []byte
}

func (t *Tensor) Elements() int {
	n := 1
	for _, d := range t.Dims {
		n *= d
	}
	return n
}

type File struct {
	Version uint32
	KV      map[string]any
	Tensors map[string]*Tensor
	data    []byte
	release func() error
}

// Open maps a GGUF file into memory and parses its header.
func Open(path string) (*File, error) {
	data, release, err := mapFile(path)
	if err != nil {
		return nil, err
	}
	f, err := parse(data)
	if err != nil {
		release()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	f.release = release
	return f, nil
}

func (f *File) Close() error {
	if f.release == nil {
		return nil
	}
	return f.release()
}

func parse(data []byte) (*File, error) {
	r := &reader{buf: data}
	if r.u32() != magic {
		return nil, errors.New("not a GGUF file")
	}

	f := &File{Version: r.u32(), KV: map[string]any{}, Tensors: map[string]*Tensor{}, data: data}
	if f.Version < 2 || f.Version > 3 {
		return nil, fmt.Errorf("unsupported GGUF version %d", f.Version)
	}

	nTensors, nKV := r.u64(), r.u64()
	for i := uint64(0); i < nKV && r.err == nil; i++ {
		key := r.str()
		f.KV[key] = r.value(r.u32())
	}

	type info struct {
		t      *Tensor
		offset uint64
	}
	infos := make([]info, 0, nTensors)
	for i := uint64(0); i < nTensors && r.err == nil; i++ {
		t := &Tensor{Name: r.str()}
		nDims := r.u32()
		for range nDims {
			t.Dims = append(t.Dims, int(r.u64()))
		}
		t.Type = ggml.Type(r.u32())
		infos = append(infos, info{t, r.u64()})
	}
	if r.err != nil {
		return nil, r.err
	}

	align := uint64(32)
	if a, ok := f.Uint("general.alignment"); ok && a > 0 {
		align = a
	}
	start := (uint64(r.pos) + align - 1) / align * align

	for _, in := range infos {
		t := in.t
		if !t.Type.Supported() {
			return nil, fmt.Errorf("tensor %s: unsupported type %v", t.Name, t.Type)
		}
		begin := start + in.offset
		end := begin + uint64(t.Type.ByteSize(t.Elements()))
		if end > uint64(len(data)) {
			return nil, fmt.Errorf("tensor %s: data out of range (truncated file?)", t.Name)
		}
		t.Data = data[begin:end]
		f.Tensors[t.Name] = t
	}
	return f, nil
}

func (f *File) String(key string) string {
	s, _ := f.KV[key].(string)
	return s
}

func (f *File) Uint(key string) (uint64, bool) {
	switch v := f.KV[key].(type) {
	case uint8:
		return uint64(v), true
	case uint16:
		return uint64(v), true
	case uint32:
		return uint64(v), true
	case uint64:
		return v, true
	case int8:
		return uint64(v), v >= 0
	case int16:
		return uint64(v), v >= 0
	case int32:
		return uint64(v), v >= 0
	case int64:
		return uint64(v), v >= 0
	}
	return 0, false
}

func (f *File) Int(key string, def int) int {
	if v, ok := f.Uint(key); ok {
		return int(v)
	}
	return def
}

func (f *File) Float(key string, def float32) float32 {
	switch v := f.KV[key].(type) {
	case float32:
		return v
	case float64:
		return float32(v)
	}
	return def
}

func (f *File) Bool(key string, def bool) bool {
	if v, ok := f.KV[key].(bool); ok {
		return v
	}
	return def
}

func (f *File) Strings(key string) []string {
	v, _ := f.KV[key].([]string)
	return v
}

func (f *File) Floats(key string) []float32 {
	v, _ := f.KV[key].([]float32)
	return v
}

func (f *File) Ints(key string) []int32 {
	v, _ := f.KV[key].([]int32)
	return v
}

// reader is a bounds-checked little-endian cursor over the file bytes.
type reader struct {
	buf []byte
	pos int
	err error
}

func (r *reader) next(n int) []byte {
	if r.err != nil {
		return make([]byte, n)
	}
	if n < 0 || r.pos+n > len(r.buf) {
		r.err = errors.New("unexpected end of header")
		return make([]byte, max(n, 0))
	}
	b := r.buf[r.pos : r.pos+n]
	r.pos += n
	return b
}

func (r *reader) u8() uint8   { return r.next(1)[0] }
func (r *reader) u16() uint16 { return binary.LittleEndian.Uint16(r.next(2)) }
func (r *reader) u32() uint32 { return binary.LittleEndian.Uint32(r.next(4)) }
func (r *reader) u64() uint64 { return binary.LittleEndian.Uint64(r.next(8)) }

func (r *reader) str() string {
	n := r.u64()
	if n > uint64(len(r.buf)) {
		r.err = errors.New("string length out of range")
		return ""
	}
	return string(r.next(int(n)))
}

const (
	typeU8 = iota
	typeI8
	typeU16
	typeI16
	typeU32
	typeI32
	typeF32
	typeBool
	typeString
	typeArray
	typeU64
	typeI64
	typeF64
)

func (r *reader) value(t uint32) any {
	switch t {
	case typeU8:
		return r.u8()
	case typeI8:
		return int8(r.u8())
	case typeU16:
		return r.u16()
	case typeI16:
		return int16(r.u16())
	case typeU32:
		return r.u32()
	case typeI32:
		return int32(r.u32())
	case typeF32:
		return math.Float32frombits(r.u32())
	case typeBool:
		return r.u8() != 0
	case typeString:
		return r.str()
	case typeU64:
		return r.u64()
	case typeI64:
		return int64(r.u64())
	case typeF64:
		return math.Float64frombits(r.u64())
	case typeArray:
		return r.array()
	}
	r.err = fmt.Errorf("unknown value type %d", t)
	return nil
}

func (r *reader) array() any {
	t, n := r.u32(), r.u64()
	if n > uint64(len(r.buf)) {
		r.err = errors.New("array length out of range")
		return nil
	}

	switch t {
	case typeString:
		out := make([]string, n)
		for i := range out {
			out[i] = r.str()
		}
		return out
	case typeF32:
		out := make([]float32, n)
		for i := range out {
			out[i] = math.Float32frombits(r.u32())
		}
		return out
	case typeI32:
		out := make([]int32, n)
		for i := range out {
			out[i] = int32(r.u32())
		}
		return out
	}

	out := make([]any, n)
	for i := range out {
		out[i] = r.value(t)
	}
	return out
}
