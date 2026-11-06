package server

import (
	"cmp"
	"encoding/json"
	"net/http"
	"sync"
	"time"

	"github.com/hostfog/hfee/internal/llm"
	"github.com/hostfog/hfee/internal/store"
)

type generateRequest struct {
	Model     string          `json:"model"`
	Prompt    string          `json:"prompt"`
	System    string          `json:"system"`
	Raw       bool            `json:"raw"`
	Stream    *bool           `json:"stream"`
	Options   json.RawMessage `json:"options"`
	KeepAlive json.RawMessage `json:"keep_alive"`
}

type chatRequest struct {
	Model     string          `json:"model"`
	Messages  []llm.Message   `json:"messages"`
	Stream    *bool           `json:"stream"`
	Options   json.RawMessage `json:"options"`
	KeepAlive json.RawMessage `json:"keep_alive"`
}

type metrics struct {
	DoneReason         string `json:"done_reason,omitempty"`
	TotalDuration      int64  `json:"total_duration,omitempty"`
	LoadDuration       int64  `json:"load_duration,omitempty"`
	PromptEvalCount    int    `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64  `json:"prompt_eval_duration,omitempty"`
	EvalCount          int    `json:"eval_count,omitempty"`
	EvalDuration       int64  `json:"eval_duration,omitempty"`
}

func parseOptions(raw json.RawMessage) (llm.Options, error) {
	opts := defaultOptions()
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &opts); err != nil {
			return opts, err
		}
	}
	return opts, nil
}

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	var req generateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		badRequest(w, err.Error())
		return
	}
	if req.Model == "" {
		badRequest(w, "model is required")
		return
	}
	opts, err := parseOptions(req.Options)
	if err != nil {
		badRequest(w, "invalid options: "+err.Error())
		return
	}

	// An empty prompt just loads (or with keep_alive 0, unloads) the model.
	if req.Prompt == "" {
		_, err := s.use(req.Model, s.keepAliveFor(req.KeepAlive), func(*llm.Runner) error { return nil })
		if err != nil {
			writeError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"model": req.Model, "created_at": now(), "response": "", "done": true})
		return
	}

	s.stream(w, r, req.Model, req.Stream, req.KeepAlive, func(rn *llm.Runner) string {
		if req.Raw {
			return req.Prompt
		}
		var msgs []llm.Message
		if req.System != "" {
			msgs = append(msgs, llm.Message{Role: "system", Content: req.System})
		}
		msgs = append(msgs, llm.Message{Role: "user", Content: req.Prompt})
		return rn.Model.Template.Format(msgs)
	}, opts, func(text string, done bool) any {
		return map[string]any{"model": req.Model, "created_at": now(), "response": text, "done": done}
	})
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		badRequest(w, err.Error())
		return
	}
	if req.Model == "" {
		badRequest(w, "model is required")
		return
	}
	opts, err := parseOptions(req.Options)
	if err != nil {
		badRequest(w, "invalid options: "+err.Error())
		return
	}

	s.stream(w, r, req.Model, req.Stream, req.KeepAlive, func(rn *llm.Runner) string {
		return rn.Model.Template.Format(req.Messages)
	}, opts, func(text string, done bool) any {
		return map[string]any{
			"model":      req.Model,
			"created_at": now(),
			"message":    llm.Message{Role: "assistant", Content: text},
			"done":       done,
		}
	})
}

// stream runs a generation and writes NDJSON chunks, or a single object when
// streaming is disabled. chunk builds the per-token body; metrics are merged
// into the final one.
func (s *Server) stream(
	w http.ResponseWriter, r *http.Request,
	model string, streamFlag *bool, keepAlive json.RawMessage,
	prompt func(*llm.Runner) string, opts llm.Options,
	chunk func(text string, done bool) any,
) {
	streaming := streamFlag == nil || *streamFlag
	start := time.Now()
	out := newNDJSON(w)

	var full []byte
	var stats llm.Stats
	load, err := s.use(model, s.keepAliveFor(keepAlive), func(rn *llm.Runner) error {
		var err error
		stats, err = rn.Generate(r.Context(), prompt(rn), opts, func(text string) error {
			if !streaming {
				full = append(full, text...)
				return nil
			}
			return out.write(chunk(text, false))
		})
		return err
	})

	if err != nil {
		if r.Context().Err() != nil {
			return // client went away
		}
		if out.started {
			out.write(map[string]string{"error": err.Error()})
		} else {
			writeError(w, err)
		}
		return
	}

	final := merge(chunk(string(full), true), metrics{
		DoneReason:         stats.DoneReason,
		TotalDuration:      int64(time.Since(start)),
		LoadDuration:       int64(load),
		PromptEvalCount:    stats.PromptTokens,
		PromptEvalDuration: int64(stats.PromptDuration),
		EvalCount:          stats.EvalTokens,
		EvalDuration:       int64(stats.EvalDuration),
	})
	if streaming {
		out.write(final)
	} else {
		writeJSON(w, http.StatusOK, final)
	}
}

func (s *Server) handlePull(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model  string `json:"model"`
		Name   string `json:"name"`
		Stream *bool  `json:"stream"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		badRequest(w, err.Error())
		return
	}
	name := cmp.Or(req.Model, req.Name)
	if name == "" {
		badRequest(w, "model is required")
		return
	}

	streaming := req.Stream == nil || *req.Stream
	out := newNDJSON(w)
	progress := func(status string, completed, total int64) {
		if streaming {
			out.write(map[string]any{"status": status, "completed": completed, "total": total})
		}
	}

	if _, err := s.store.Pull(r.Context(), name, progress); err != nil {
		if out.started {
			out.write(map[string]string{"error": err.Error()})
		} else {
			writeError(w, err)
		}
		return
	}

	if streaming {
		out.write(map[string]string{"status": "success"})
	} else {
		writeJSON(w, http.StatusOK, map[string]string{"status": "success"})
	}
}

func (s *Server) handleTags(w http.ResponseWriter, r *http.Request) {
	models, err := s.store.List()
	if err != nil {
		writeError(w, err)
		return
	}

	list := make([]map[string]any, 0, len(models))
	for _, m := range models {
		list = append(list, map[string]any{
			"name":        m.Name,
			"model":       m.Name,
			"modified_at": m.Modified,
			"size":        m.Size,
			"digest":      m.Digest,
			"details":     m.Details,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": list})
}

func (s *Server) handleShow(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
		Name  string `json:"name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		badRequest(w, err.Error())
		return
	}

	m, err := s.store.Get(cmp.Or(req.Model, req.Name))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, m)
}

func (s *Server) handleDelete(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
		Name  string `json:"name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		badRequest(w, err.Error())
		return
	}
	name := store.Normalize(cmp.Or(req.Model, req.Name))

	s.run.Lock()
	s.mu.Lock()
	inUse := s.loaded != nil && s.loaded.name == name
	s.mu.Unlock()
	if inUse {
		s.unload()
	}
	s.run.Unlock()

	if err := s.store.Delete(name); err != nil {
		writeError(w, err)
		return
	}
	w.WriteHeader(http.StatusOK)
}

func (s *Server) handlePs(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()

	list := []map[string]any{}
	if m := s.loaded; m != nil {
		entry := map[string]any{"name": m.name, "model": m.name, "size": m.size}
		if !m.expires.IsZero() {
			entry["expires_at"] = m.expires
		}
		list = append(list, entry)
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": list})
}

// ndjson writes newline-delimited JSON and flushes after each line.
type ndjson struct {
	w       http.ResponseWriter
	mu      sync.Mutex
	started bool
}

func newNDJSON(w http.ResponseWriter) *ndjson { return &ndjson{w: w} }

func (n *ndjson) write(v any) error {
	n.mu.Lock()
	defer n.mu.Unlock()
	if !n.started {
		n.w.Header().Set("Content-Type", "application/x-ndjson")
		n.started = true
	}
	if err := json.NewEncoder(n.w).Encode(v); err != nil {
		return err
	}
	http.NewResponseController(n.w).Flush()
	return nil
}

// merge adds the fields of extra into base (both must marshal to JSON objects).
func merge(base, extra any) map[string]any {
	out := map[string]any{}
	for _, v := range []any{base, extra} {
		b, _ := json.Marshal(v)
		json.Unmarshal(b, &out)
	}
	return out
}

func now() time.Time { return time.Now().UTC() }
