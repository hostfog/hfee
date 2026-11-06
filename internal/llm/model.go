// Package llm loads GGUF language models and runs inference on the CPU.
package llm

import (
	"fmt"
	"math"

	"github.com/hostfog/hfee/internal/ggml"
	"github.com/hostfog/hfee/internal/gguf"
	"github.com/hostfog/hfee/internal/tokenizer"
)

type Config struct {
	Arch     string
	Dim      int
	Hidden   int
	Layers   int
	Heads    int
	KVHeads  int
	HeadDim  int
	RopeDim  int
	CtxTrain int
	Eps      float32
	RopeBase float64
	RopeNeox bool // rotate pairs (i, i+d/2) instead of (2i, 2i+1)
}

type layer struct {
	attnNorm, ffnNorm []float32
	wq, wk, wv, wo    *ggml.Matrix
	bq, bk, bv        []float32 // optional (qwen2)
	qNorm, kNorm      []float32 // optional per-head norms (qwen3)
	gate, up, down    *ggml.Matrix
}

type Model struct {
	Config
	Name      string
	Tokenizer *tokenizer.Tokenizer
	Template  Template

	file    *gguf.File
	embed   *ggml.Matrix
	outNorm []float32
	output  *ggml.Matrix
	layers  []layer
	invFreq []float64
}

// supported architectures and whether they use NeoX-style rotary embeddings
var architectures = map[string]bool{
	"llama":   false,
	"mistral": false,
	"qwen2":   true,
	"qwen3":   true,
}

func Load(path string) (*Model, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}

	m, err := build(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	return m, nil
}

func (m *Model) Close() error {
	return m.file.Close()
}

func build(f *gguf.File) (*Model, error) {
	arch := f.String("general.architecture")
	neox, ok := architectures[arch]
	if !ok {
		return nil, fmt.Errorf("unsupported model architecture %q", arch)
	}

	key := func(k string) string { return arch + "." + k }
	c := Config{
		Arch:     arch,
		Dim:      f.Int(key("embedding_length"), 0),
		Hidden:   f.Int(key("feed_forward_length"), 0),
		Layers:   f.Int(key("block_count"), 0),
		Heads:    f.Int(key("attention.head_count"), 0),
		CtxTrain: f.Int(key("context_length"), 2048),
		Eps:      f.Float(key("attention.layer_norm_rms_epsilon"), 1e-5),
		RopeBase: float64(f.Float(key("rope.freq_base"), 10000)),
		RopeNeox: neox,
	}
	if c.Dim == 0 || c.Layers == 0 || c.Heads == 0 {
		return nil, fmt.Errorf("model metadata is incomplete")
	}
	c.KVHeads = f.Int(key("attention.head_count_kv"), c.Heads)
	c.HeadDim = f.Int(key("attention.key_length"), c.Dim/c.Heads)
	c.RopeDim = f.Int(key("rope.dimension_count"), c.HeadDim)

	tok, err := tokenizer.FromGGUF(f)
	if err != nil {
		return nil, err
	}

	m := &Model{
		Config:    c,
		Name:      f.String("general.name"),
		Tokenizer: tok,
		Template:  detectTemplate(f.String("tokenizer.chat_template"), tok),
		file:      f,
	}

	l := loader{f: f}
	m.embed = l.matrix("token_embd.weight")
	m.outNorm = l.vector("output_norm.weight")
	m.output = l.optionalMatrix("output.weight")
	if m.output == nil {
		m.output = m.embed // tied embeddings
	}

	m.layers = make([]layer, c.Layers)
	for i := range m.layers {
		name := func(s string) string { return fmt.Sprintf("blk.%d.%s", i, s) }
		m.layers[i] = layer{
			attnNorm: l.vector(name("attn_norm.weight")),
			ffnNorm:  l.vector(name("ffn_norm.weight")),
			wq:       l.matrix(name("attn_q.weight")),
			wk:       l.matrix(name("attn_k.weight")),
			wv:       l.matrix(name("attn_v.weight")),
			wo:       l.matrix(name("attn_output.weight")),
			bq:       l.optionalVector(name("attn_q.bias")),
			bk:       l.optionalVector(name("attn_k.bias")),
			bv:       l.optionalVector(name("attn_v.bias")),
			qNorm:    l.optionalVector(name("attn_q_norm.weight")),
			kNorm:    l.optionalVector(name("attn_k_norm.weight")),
			gate:     l.matrix(name("ffn_gate.weight")),
			up:       l.matrix(name("ffn_up.weight")),
			down:     l.matrix(name("ffn_down.weight")),
		}
	}
	if l.err != nil {
		return nil, l.err
	}
	if c.Hidden == 0 {
		m.Hidden = m.layers[0].gate.Rows
	}

	// rotary frequencies, optionally scaled per dimension (llama 3.1+)
	factors := l.optionalVector("rope_freqs.weight")
	m.invFreq = make([]float64, c.RopeDim/2)
	for i := range m.invFreq {
		m.invFreq[i] = 1 / math.Pow(c.RopeBase, float64(2*i)/float64(c.RopeDim))
		if i < len(factors) {
			m.invFreq[i] /= float64(factors[i])
		}
	}
	return m, l.err
}

// loader collects the first missing-tensor error instead of failing at each call.
type loader struct {
	f   *gguf.File
	err error
}

func (l *loader) tensor(name string, required bool) *gguf.Tensor {
	t, ok := l.f.Tensors[name]
	if !ok && required && l.err == nil {
		l.err = fmt.Errorf("model is missing tensor %s", name)
	}
	return t
}

func (l *loader) matrix(name string) *ggml.Matrix {
	return l.toMatrix(l.tensor(name, true))
}

func (l *loader) optionalMatrix(name string) *ggml.Matrix {
	return l.toMatrix(l.tensor(name, false))
}

func (l *loader) toMatrix(t *gguf.Tensor) *ggml.Matrix {
	if t == nil {
		return nil
	}
	if len(t.Dims) != 2 && l.err == nil {
		l.err = fmt.Errorf("tensor %s: expected 2 dimensions, got %d", t.Name, len(t.Dims))
		return nil
	}
	return ggml.NewMatrix(t.Type, t.Dims[1], t.Dims[0], t.Data)
}

func (l *loader) vector(name string) []float32 {
	return l.toVector(l.tensor(name, true))
}

func (l *loader) optionalVector(name string) []float32 {
	return l.toVector(l.tensor(name, false))
}

func (l *loader) toVector(t *gguf.Tensor) []float32 {
	if t == nil {
		return nil
	}
	v := make([]float32, t.Elements())
	ggml.Dequantize(t.Type, v, t.Data)
	return v
}
