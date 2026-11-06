package server

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/hostfog/hfee/internal/llm"
)

type openAIMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// text accepts both a plain string and the array-of-parts form.
func (m openAIMessage) text() string {
	var s string
	if json.Unmarshal(m.Content, &s) == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	json.Unmarshal(m.Content, &parts)
	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

type chatCompletionRequest struct {
	Model               string          `json:"model"`
	Messages            []openAIMessage `json:"messages"`
	Stream              bool            `json:"stream"`
	Temperature         *float32        `json:"temperature"`
	TopP                *float32        `json:"top_p"`
	MaxTokens           *int            `json:"max_tokens"`
	MaxCompletionTokens *int            `json:"max_completion_tokens"`
	Seed                *int64          `json:"seed"`
	Stop                json.RawMessage `json:"stop"`
}

func (req chatCompletionRequest) options() llm.Options {
	opts := defaultOptions()
	if req.Temperature != nil {
		opts.Temperature = *req.Temperature
	}
	if req.TopP != nil {
		opts.TopP = *req.TopP
	}
	if req.MaxTokens != nil {
		opts.NumPredict = *req.MaxTokens
	}
	if req.MaxCompletionTokens != nil {
		opts.NumPredict = *req.MaxCompletionTokens
	}
	if req.Seed != nil {
		opts.Seed = *req.Seed
	}

	var one string
	if json.Unmarshal(req.Stop, &one) == nil && one != "" {
		opts.Stop = []string{one}
	} else {
		json.Unmarshal(req.Stop, &opts.Stop)
	}
	return opts
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req chatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		openAIError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Model == "" {
		openAIError(w, http.StatusBadRequest, "model is required")
		return
	}

	msgs := make([]llm.Message, len(req.Messages))
	for i, m := range req.Messages {
		role := m.Role
		if role == "developer" {
			role = "system"
		}
		msgs[i] = llm.Message{Role: role, Content: m.text()}
	}

	id := fmt.Sprintf("chatcmpl-%d", rand.Int63())
	created := time.Now().Unix()
	chunk := func(delta map[string]string, finish any) map[string]any {
		return map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created, "model": req.Model,
			"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": finish}},
		}
	}

	var sse *eventStream
	if req.Stream {
		sse = newEventStream(w)
	}

	var full strings.Builder
	var stats llm.Stats
	_, err := s.use(req.Model, s.keepAlive, func(rn *llm.Runner) error {
		var err error
		if sse != nil {
			sse.send(chunk(map[string]string{"role": "assistant", "content": ""}, nil))
		}
		stats, err = rn.Generate(r.Context(), rn.Model.Template.Format(msgs), req.options(), func(text string) error {
			if sse != nil {
				return sse.send(chunk(map[string]string{"content": text}, nil))
			}
			full.WriteString(text)
			return nil
		})
		return err
	})

	if err != nil {
		if r.Context().Err() != nil {
			return
		}
		if sse != nil && sse.started {
			sse.send(map[string]any{"error": map[string]string{"message": err.Error()}})
			return
		}
		openAIError(w, http.StatusInternalServerError, err.Error())
		return
	}

	usage := map[string]int{
		"prompt_tokens":     stats.PromptTokens,
		"completion_tokens": stats.EvalTokens,
		"total_tokens":      stats.PromptTokens + stats.EvalTokens,
	}

	if sse != nil {
		final := chunk(map[string]string{}, stats.DoneReason)
		final["usage"] = usage
		sse.send(final)
		sse.done()
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "object": "chat.completion", "created": created, "model": req.Model,
		"choices": []map[string]any{{
			"index":         0,
			"message":       map[string]string{"role": "assistant", "content": full.String()},
			"finish_reason": stats.DoneReason,
		}},
		"usage": usage,
	})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	models, err := s.store.List()
	if err != nil {
		openAIError(w, http.StatusInternalServerError, err.Error())
		return
	}

	data := make([]map[string]any, 0, len(models))
	for _, m := range models {
		data = append(data, map[string]any{
			"id": m.Name, "object": "model", "created": m.Modified.Unix(), "owned_by": "hfee",
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

func openAIError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": map[string]string{"message": msg, "type": "invalid_request_error"}})
}

type eventStream struct {
	w       http.ResponseWriter
	started bool
}

func newEventStream(w http.ResponseWriter) *eventStream { return &eventStream{w: w} }

func (e *eventStream) send(v any) error {
	if !e.started {
		e.w.Header().Set("Content-Type", "text/event-stream")
		e.w.Header().Set("Cache-Control", "no-cache")
		e.started = true
	}
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(e.w, "data: %s\n\n", b); err != nil {
		return err
	}
	return http.NewResponseController(e.w).Flush()
}

func (e *eventStream) done() {
	fmt.Fprint(e.w, "data: [DONE]\n\n")
	http.NewResponseController(e.w).Flush()
}
