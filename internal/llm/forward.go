package llm

import (
	"math"

	"github.com/hostfog/hfee/internal/ggml"
)

// State holds the key/value cache for one sequence.
type State struct {
	m      *Model
	ctx    int
	keys   [][]float32 // per layer: ctx × kvDim
	values [][]float32
	tokens []int
}

func (m *Model) NewState(ctx int) *State {
	kvDim := m.KVHeads * m.HeadDim
	s := &State{m: m, ctx: ctx, keys: make([][]float32, m.Layers), values: make([][]float32, m.Layers)}
	for i := range s.keys {
		s.keys[i] = make([]float32, ctx*kvDim)
		s.values[i] = make([]float32, ctx*kvDim)
	}
	return s
}

func (s *State) Context() int  { return s.ctx }
func (s *State) Tokens() []int { return s.tokens }

// Truncate drops cached tokens after position n.
func (s *State) Truncate(n int) {
	s.tokens = s.tokens[:min(n, len(s.tokens))]
}

// Forward appends tokens to the sequence and returns the logits for the last one.
func (s *State) Forward(tokens []int) []float32 {
	return s.m.Forward([]Batch{{State: s, Tokens: tokens, Logits: true}})[0]
}

// Batch is one sequence's share of a forward pass.
type Batch struct {
	State  *State
	Tokens []int
	Logits bool // whether the logits for the last token are needed
}

// Forward runs several sequences through the model together, so each weight
// matrix is read once for all of them. It returns the last-token logits for
// every batch that asked for them (nil for the others).
func (m *Model) Forward(batches []Batch) [][]float32 {
	dim, qDim, kvDim := m.Dim, m.Heads*m.HeadDim, m.KVHeads*m.HeadDim

	// flatten: token i belongs to seq[i] at position pos[i]
	var tokens, seq, pos []int
	for b, bt := range batches {
		start := len(bt.State.tokens)
		for k, t := range bt.Tokens {
			tokens = append(tokens, t)
			seq = append(seq, b)
			pos = append(pos, start+k)
		}
	}
	n := len(tokens)

	x := make([]float32, n*dim)
	xb := make([]float32, n*max(dim, qDim))
	q := make([]float32, n*qDim)
	k := make([]float32, n*kvDim)
	v := make([]float32, n*kvDim)
	att := make([]float32, n*qDim)
	gate := make([]float32, n*m.Hidden)
	up := make([]float32, n*m.Hidden)

	for i, t := range tokens {
		m.embed.Row(t, x[i*dim:(i+1)*dim])
	}

	for l := range m.layers {
		ly := &m.layers[l]

		normRows(xb, x, ly.attnNorm, n, dim, m.Eps)
		ggml.MatMul(q, ly.wq, xb, n)
		ggml.MatMul(k, ly.wk, xb, n)
		ggml.MatMul(v, ly.wv, xb, n)
		addBias(q, ly.bq, n)
		addBias(k, ly.bk, n)
		addBias(v, ly.bv, n)

		for i := 0; i < n; i++ {
			qi, ki := q[i*qDim:(i+1)*qDim], k[i*kvDim:(i+1)*kvDim]
			if ly.qNorm != nil {
				normRows(qi, qi, ly.qNorm, m.Heads, m.HeadDim, m.Eps)
				normRows(ki, ki, ly.kNorm, m.KVHeads, m.HeadDim, m.Eps)
			}
			m.rope(qi, m.Heads, pos[i])
			m.rope(ki, m.KVHeads, pos[i])

			st := batches[seq[i]].State
			copy(st.keys[l][pos[i]*kvDim:], ki)
			copy(st.values[l][pos[i]*kvDim:], v[i*kvDim:(i+1)*kvDim])
		}

		m.attention(att, q, l, batches, seq, pos)

		ggml.MatMul(xb[:n*dim], ly.wo, att, n)
		ggml.Add(x, xb[:n*dim])

		normRows(xb, x, ly.ffnNorm, n, dim, m.Eps)
		ggml.MatMul(gate, ly.gate, xb, n)
		ggml.MatMul(up, ly.up, xb, n)
		ggml.SwiGLU(gate, up)
		ggml.MatMul(xb[:n*dim], ly.down, gate, n)
		ggml.Add(x, xb[:n*dim])
	}

	// logits only for the last token of the batches that want them
	var last []float32
	var want []int
	end := 0
	for b, bt := range batches {
		end += len(bt.Tokens)
		bt.State.tokens = append(bt.State.tokens, bt.Tokens...)
		if bt.Logits {
			row := x[(end-1)*dim : end*dim]
			ggml.RMSNorm(row, row, m.outNorm, m.Eps)
			last = append(last, row...)
			want = append(want, b)
		}
	}

	out := make([][]float32, len(batches))
	if len(want) > 0 {
		vocab := m.output.Rows
		logits := make([]float32, len(want)*vocab)
		ggml.MatMul(logits, m.output, last, len(want))
		for i, b := range want {
			out[b] = logits[i*vocab : (i+1)*vocab]
		}
	}
	return out
}

// attention computes causal multi-head attention for every token, each
// against its own sequence's cache.
func (m *Model) attention(out, q []float32, l int, batches []Batch, seq, pos []int) {
	hd, qDim, kvDim := m.HeadDim, m.Heads*m.HeadDim, m.KVHeads*m.HeadDim
	group := m.Heads / m.KVHeads
	scale := float32(1 / math.Sqrt(float64(hd)))

	longest := 0
	for _, p := range pos {
		longest = max(longest, p+1)
	}

	ggml.Parallel(len(pos)*m.Heads, func(start, end int) {
		scores := make([]float32, longest)
		for task := start; task < end; task++ {
			i, h := task/m.Heads, task%m.Heads
			st := batches[seq[i]].State
			keys, values := st.keys[l], st.values[l]
			span := pos[i] + 1
			qh := q[i*qDim+h*hd : i*qDim+(h+1)*hd]
			kvOff := (h / group) * hd

			for j := 0; j < span; j++ {
				scores[j] = ggml.Dot(qh, keys[j*kvDim+kvOff:j*kvDim+kvOff+hd]) * scale
			}
			ggml.Softmax(scores[:span])

			o := out[i*qDim+h*hd : i*qDim+(h+1)*hd]
			clear(o)
			for j := 0; j < span; j++ {
				ggml.Axpy(o, scores[j], values[j*kvDim+kvOff:j*kvDim+kvOff+hd])
			}
		}
	})
}

// rope applies rotary position embeddings to each head in x.
func (m *Model) rope(x []float32, heads, pos int) {
	half := m.RopeDim / 2
	for h := 0; h < heads; h++ {
		v := x[h*m.HeadDim : (h+1)*m.HeadDim]
		for i := 0; i < half; i++ {
			sin, cos := math.Sincos(float64(pos) * m.invFreq[i])
			a, b := 2*i, 2*i+1
			if m.RopeNeox {
				a, b = i, i+half
			}
			x0, x1 := float64(v[a]), float64(v[b])
			v[a] = float32(x0*cos - x1*sin)
			v[b] = float32(x0*sin + x1*cos)
		}
	}
}

func normRows(dst, x, weight []float32, n, width int, eps float32) {
	for i := 0; i < n; i++ {
		ggml.RMSNorm(dst[i*width:(i+1)*width], x[i*width:(i+1)*width], weight, eps)
	}
}

func addBias(x, bias []float32, n int) {
	if bias == nil {
		return
	}
	for t := 0; t < n; t++ {
		ggml.Add(x[t*len(bias):(t+1)*len(bias)], bias)
	}
}
