package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

const batchSize = 256 // tokens processed per forward pass

var ErrClosed = errors.New("model was unloaded")

type Stats struct {
	PromptTokens   int
	PromptDuration time.Duration
	EvalTokens     int
	EvalDuration   time.Duration
	DoneReason     string
}

// Runner serves generation requests for one model. Up to `parallel`
// sequences run together: each step feeds one token from every sequence
// that is generating, plus a chunk of any prompt still being read, through
// a single forward pass.
type Runner struct {
	Model *Model

	numCtx   int
	slots    []*slot
	incoming chan *request
	quit     chan struct{}
	done     chan struct{}
}

// slot is one sequence's cache. It is kept after a request finishes so a
// follow-up message in the same chat only processes the new tokens.
type slot struct {
	state   *State
	req     *request
	lastUse time.Time
}

type request struct {
	ctx    context.Context
	tokens []int
	opts   Options
	out    chan string
	result chan result

	// owned by the loop
	feed      []int // tokens to run next step
	smp       *sampler
	history   []int
	stop      stopper
	dec       utf8Buffer
	started   bool
	stats     Stats
	start     time.Time
	evalStart time.Time
}

type result struct {
	stats Stats
	err   error
}

func NewRunner(m *Model, parallel, numCtx int) *Runner {
	r := &Runner{
		Model:    m,
		numCtx:   min(max(numCtx, 256), m.CtxTrain),
		slots:    make([]*slot, max(parallel, 1)),
		incoming: make(chan *request),
		quit:     make(chan struct{}),
		done:     make(chan struct{}),
	}
	for i := range r.slots {
		r.slots[i] = &slot{}
	}
	go r.loop()
	return r
}

// Close stops the runner, failing requests in progress, and frees the model.
func (r *Runner) Close() error {
	close(r.quit)
	<-r.done
	return r.Model.Close()
}

// Generate continues prompt and streams the response text to emit.
func (r *Runner) Generate(ctx context.Context, prompt string, opts Options, emit func(string) error) (Stats, error) {
	ids := r.Model.Tokenizer.Encode(prompt, true)
	if len(ids) >= r.numCtx {
		return Stats{}, fmt.Errorf("prompt is %d tokens but the context window is %d", len(ids), r.numCtx)
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	req := &request{
		ctx:    ctx,
		tokens: ids,
		opts:   opts,
		out:    make(chan string, 1<<14),
		result: make(chan result, 1),
	}

	select {
	case r.incoming <- req:
	case <-ctx.Done():
		return Stats{}, ctx.Err()
	case <-r.done:
		return Stats{}, ErrClosed
	}

	var emitErr error
	forward := func(text string) {
		if emitErr == nil {
			if emitErr = emit(text); emitErr != nil {
				cancel()
			}
		}
	}

	for {
		select {
		case text := <-req.out:
			forward(text)
		case res := <-req.result:
			for len(req.out) > 0 {
				forward(<-req.out)
			}
			if emitErr != nil {
				return res.stats, emitErr
			}
			return res.stats, res.err
		}
	}
}

func (r *Runner) loop() {
	defer close(r.done)
	for {
		if !r.busy() {
			select {
			case req := <-r.incoming:
				r.admit(req)
			case <-r.quit:
				return
			}
		}

	admit:
		for r.free() != nil {
			select {
			case req := <-r.incoming:
				r.admit(req)
			default:
				break admit
			}
		}

		select {
		case <-r.quit:
			for _, s := range r.slots {
				if s.req != nil {
					r.finish(s, "", ErrClosed)
				}
			}
			return
		default:
		}

		r.step()
	}
}

func (r *Runner) busy() bool {
	for _, s := range r.slots {
		if s.req != nil {
			return true
		}
	}
	return false
}

func (r *Runner) free() *slot {
	for _, s := range r.slots {
		if s.req == nil {
			return s
		}
	}
	return nil
}

// admit puts req in the free slot whose cache shares the longest prefix
// with it, or else the least recently used one.
func (r *Runner) admit(req *request) {
	var best *slot
	bestKeep := -1
	for _, s := range r.slots {
		if s.req != nil {
			continue
		}
		keep := 0
		if s.state != nil {
			keep = commonPrefix(s.state.Tokens(), req.tokens)
		}
		if keep > bestKeep || (keep == bestKeep && s.lastUse.Before(best.lastUse)) {
			best, bestKeep = s, keep
		}
	}

	if best.state == nil {
		best.state = r.Model.NewState(r.numCtx)
	}
	keep := min(bestKeep, len(req.tokens)-1) // always run at least one token to get logits
	best.state.Truncate(keep)

	req.feed = req.tokens[keep:]
	req.smp = newSampler(req.opts)
	req.history = append([]int(nil), req.tokens...)
	req.stop = stopper{stops: req.opts.Stop}
	req.stats.PromptTokens = len(req.feed)
	req.start = time.Now()
	best.req = req
}

// step runs one forward pass over every active sequence.
func (r *Runner) step() {
	var batches []Batch
	var owners []*slot
	budget := batchSize

	// generating sequences first (one token each), then prompt chunks
	for _, phase := range []bool{true, false} {
		for _, s := range r.slots {
			req := s.req
			if req == nil || req.evalStart.IsZero() != !phase {
				continue
			}
			if err := req.ctx.Err(); err != nil {
				r.finish(s, "", err)
				continue
			}
			if len(s.state.Tokens()) >= r.numCtx {
				r.finish(s, "length", nil)
				continue
			}
			if budget == 0 {
				continue
			}

			n := min(len(req.feed), budget)
			budget -= n
			batches = append(batches, Batch{State: s.state, Tokens: req.feed[:n], Logits: n == len(req.feed)})
			owners = append(owners, s)
		}
	}
	if len(batches) == 0 {
		return
	}

	logits := r.Model.Forward(batches)

	for i, s := range owners {
		req := s.req
		req.feed = req.feed[len(batches[i].Tokens):]
		if logits[i] == nil {
			continue // prompt not finished yet
		}
		if req.evalStart.IsZero() {
			req.stats.PromptDuration = time.Since(req.start)
			req.evalStart = time.Now()
		}
		r.sample(s, logits[i])
	}
}

// sample picks the next token for the request in s and streams its text.
func (r *Runner) sample(s *slot, logits []float32) {
	req := s.req
	tok := r.Model.Tokenizer

	id := req.smp.sample(logits, req.history)
	if tok.IsEnd(id) {
		r.finish(s, "stop", nil)
		return
	}
	req.history = append(req.history, id)
	req.stats.EvalTokens++

	text := req.dec.add(tok.Piece(id))
	if !req.started {
		text = strings.TrimLeft(text, " ")
		req.started = text != ""
	}

	out, hit := req.stop.push(text)
	if !send(req, out) {
		r.finish(s, "", errors.New("client is not reading the response"))
		return
	}
	switch {
	case hit:
		r.finish(s, "stop", nil)
	case req.opts.NumPredict >= 0 && req.stats.EvalTokens >= req.opts.NumPredict:
		r.finish(s, "length", nil)
	default:
		req.feed = []int{id}
	}
}

func (r *Runner) finish(s *slot, reason string, err error) {
	req := s.req
	if err == nil {
		send(req, req.stop.flush()+req.dec.flush())
	}
	req.stats.DoneReason = reason
	if !req.evalStart.IsZero() {
		req.stats.EvalDuration = time.Since(req.evalStart)
	}
	req.result <- result{req.stats, err}
	s.req = nil
	s.lastUse = time.Now()
}

func send(req *request, text string) bool {
	if text == "" {
		return true
	}
	select {
	case req.out <- text:
		return true
	default:
		return false
	}
}
