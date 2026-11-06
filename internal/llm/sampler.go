package llm

import (
	"container/heap"
	"math"
	"math/rand"
	"sort"
)

type Options struct {
	Temperature   float32  `json:"temperature"`
	TopK          int      `json:"top_k"`
	TopP          float32  `json:"top_p"`
	MinP          float32  `json:"min_p"`
	RepeatPenalty float32  `json:"repeat_penalty"`
	RepeatLastN   int      `json:"repeat_last_n"`
	Seed          int64    `json:"seed"`
	NumPredict    int      `json:"num_predict"`
	NumCtx        int      `json:"num_ctx"`
	Stop          []string `json:"stop"`
}

func DefaultOptions() Options {
	return Options{
		Temperature:   0.8,
		TopK:          40,
		TopP:          0.9,
		RepeatPenalty: 1.1,
		RepeatLastN:   64,
		Seed:          -1,
		NumPredict:    -1,
		NumCtx:        4096,
	}
}

type sampler struct {
	opts Options
	rng  *rand.Rand
}

func newSampler(opts Options) *sampler {
	seed := opts.Seed
	if seed < 0 {
		seed = rand.Int63()
	}
	return &sampler{opts: opts, rng: rand.New(rand.NewSource(seed))}
}

type candidate struct {
	id    int
	logit float32
}

func (s *sampler) sample(logits []float32, history []int) int {
	o := s.opts

	if o.RepeatPenalty != 1 && o.RepeatLastN != 0 {
		recent := history
		if o.RepeatLastN > 0 && len(recent) > o.RepeatLastN {
			recent = recent[len(recent)-o.RepeatLastN:]
		}
		seen := map[int]bool{}
		for _, id := range recent {
			if seen[id] || id >= len(logits) {
				continue
			}
			seen[id] = true
			if logits[id] > 0 {
				logits[id] /= o.RepeatPenalty
			} else {
				logits[id] *= o.RepeatPenalty
			}
		}
	}

	if o.Temperature <= 0 {
		best := 0
		for i, v := range logits {
			if v > logits[best] {
				best = i
			}
		}
		return best
	}

	cands := topCandidates(logits, o.TopK)

	probs := make([]float64, len(cands))
	var sum float64
	for i, c := range cands {
		probs[i] = math.Exp(float64((c.logit - cands[0].logit) / o.Temperature))
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}

	keep := len(probs)
	if o.TopP > 0 && o.TopP < 1 {
		var cum float64
		for i, p := range probs {
			cum += p
			if cum >= float64(o.TopP) {
				keep = i + 1
				break
			}
		}
	}
	if o.MinP > 0 {
		for i := 1; i < keep; i++ {
			if probs[i] < float64(o.MinP)*probs[0] {
				keep = i
				break
			}
		}
	}

	var total float64
	for _, p := range probs[:keep] {
		total += p
	}
	r := s.rng.Float64() * total
	for i, p := range probs[:keep] {
		r -= p
		if r <= 0 {
			return cands[i].id
		}
	}
	return cands[keep-1].id
}

// topCandidates returns the k highest logits in descending order (all of them if k <= 0).
func topCandidates(logits []float32, k int) []candidate {
	if k <= 0 || k > len(logits) {
		k = len(logits)
	}

	h := make(minHeap, 0, k)
	for i, v := range logits {
		if len(h) < k {
			heap.Push(&h, candidate{i, v})
		} else if v > h[0].logit {
			h[0] = candidate{i, v}
			heap.Fix(&h, 0)
		}
	}

	out := []candidate(h)
	sort.Slice(out, func(i, j int) bool { return out[i].logit > out[j].logit })
	return out
}

type minHeap []candidate

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].logit < h[j].logit }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(candidate)) }
func (h *minHeap) Pop() any {
	old := *h
	c := old[len(old)-1]
	*h = old[:len(old)-1]
	return c
}
