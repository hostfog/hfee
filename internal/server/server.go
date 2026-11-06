// Package server exposes models over an Ollama-compatible HTTP API and an
// OpenAI-compatible /v1 API.
package server

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/hostfog/hfee/internal/llm"
	"github.com/hostfog/hfee/internal/store"
)

const Version = "0.1.0"

const defaultKeepAlive = 5 * time.Minute

type Server struct {
	store     *store.Store
	keepAlive time.Duration

	// Requests share the loaded model (read lock); loading, unloading and
	// switching models wait for them to finish (write lock).
	run sync.RWMutex

	parallel int
	numCtx   int

	mu     sync.Mutex // guards loaded
	loaded *loadedModel
}

type loadedModel struct {
	name    string
	size    int64
	runner  *llm.Runner
	expires time.Time
	timer   *time.Timer
}

func New(st *store.Store) *Server {
	s := &Server{store: st, keepAlive: defaultKeepAlive, parallel: 4, numCtx: defaultOptions().NumCtx}
	if n, err := strconv.Atoi(os.Getenv("HFEE_NUM_PARALLEL")); err == nil && n > 0 {
		s.parallel = n
	}
	if v := os.Getenv("HFEE_KEEP_ALIVE"); v != "" {
		if d, ok := parseKeepAlive([]byte(strconv.Quote(v))); ok {
			s.keepAlive = d
		}
	}
	return s
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("HFee is running"))
	})
	mux.HandleFunc("GET /api/version", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"version": Version})
	})
	mux.HandleFunc("POST /api/generate", s.handleGenerate)
	mux.HandleFunc("POST /api/chat", s.handleChat)
	mux.HandleFunc("POST /api/pull", s.handlePull)
	mux.HandleFunc("GET /api/tags", s.handleTags)
	mux.HandleFunc("POST /api/show", s.handleShow)
	mux.HandleFunc("DELETE /api/delete", s.handleDelete)
	mux.HandleFunc("GET /api/ps", s.handlePs)

	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("GET /v1/models", s.handleModels)

	return cors(mux)
}

// use loads name if needed and runs fn with it. Requests for the loaded
// model run concurrently; a different model waits until they finish.
func (s *Server) use(name string, keepAlive time.Duration, fn func(*llm.Runner) error) (loadTime time.Duration, err error) {
	name = store.Normalize(name)
	start := time.Now()

	var runner *llm.Runner
	for {
		s.run.RLock()
		if cur := s.current(); cur != nil && cur.name == name {
			runner = cur.runner
			break
		}
		s.run.RUnlock()

		s.run.Lock()
		_, err := s.load(name)
		s.run.Unlock()
		if err != nil {
			return 0, err
		}
	}
	loadTime = time.Since(start)

	err = fn(runner)
	s.run.RUnlock()
	s.expire(keepAlive)
	return loadTime, err
}

func (s *Server) current() *loadedModel {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.loaded
}

// load returns the runner for name, replacing any other loaded model. Caller holds s.run for writing.
func (s *Server) load(name string) (*llm.Runner, error) {
	if cur := s.current(); cur != nil && cur.name == name {
		return cur.runner, nil
	}
	s.unload()

	info, err := s.store.Get(name)
	if err != nil {
		return nil, err
	}

	log.Printf("loading %s", name)
	m, err := llm.Load(s.store.BlobPath(info.Digest))
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	s.loaded = &loadedModel{name: name, size: info.Size, runner: llm.NewRunner(m, s.parallel, s.numCtx)}
	s.mu.Unlock()
	return s.loaded.runner, nil
}

// unload frees the current model. Caller holds s.run for writing.
func (s *Server) unload() {
	s.mu.Lock()
	cur := s.loaded
	s.loaded = nil
	s.mu.Unlock()

	if cur == nil {
		return
	}
	if cur.timer != nil {
		cur.timer.Stop()
	}
	log.Printf("unloading %s", cur.name)
	cur.runner.Close()
}

// expire schedules the loaded model to be freed after d.
func (s *Server) expire(d time.Duration) {
	if d == 0 {
		s.run.Lock()
		s.unload()
		s.run.Unlock()
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	cur := s.loaded
	if cur == nil {
		return
	}
	if cur.timer != nil {
		cur.timer.Stop()
		cur.timer = nil
	}
	if d < 0 {
		cur.expires = time.Time{}
		return
	}

	cur.expires = time.Now().Add(d)
	cur.timer = time.AfterFunc(d, func() {
		s.run.Lock()
		defer s.run.Unlock()
		s.mu.Lock()
		stale := s.loaded != cur || time.Now().Before(cur.expires)
		s.mu.Unlock()
		if !stale {
			s.unload()
		}
	})
}

// parseKeepAlive accepts a duration string ("10m"), seconds as a number, or -1 for forever.
func parseKeepAlive(raw json.RawMessage) (time.Duration, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	var n float64
	if json.Unmarshal(raw, &n) == nil {
		if n < 0 {
			return -1, true
		}
		return time.Duration(n * float64(time.Second)), true
	}
	var str string
	if json.Unmarshal(raw, &str) == nil {
		if n, err := strconv.ParseFloat(str, 64); err == nil {
			return parseKeepAlive(json.RawMessage(strconv.FormatFloat(n, 'f', -1, 64)))
		}
		if d, err := time.ParseDuration(str); err == nil {
			if d < 0 {
				return -1, true
			}
			return d, true
		}
	}
	return 0, false
}

// defaultOptions applies server-wide settings from the environment.
func defaultOptions() llm.Options {
	opts := llm.DefaultOptions()
	if n, err := strconv.Atoi(os.Getenv("HFEE_NUM_CTX")); err == nil && n > 0 {
		opts.NumCtx = n
	}
	return opts
}

func (s *Server) keepAliveFor(raw json.RawMessage) time.Duration {
	if d, ok := parseKeepAlive(raw); ok {
		return d
	}
	return s.keepAlive
}

func cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, err error) {
	status := http.StatusInternalServerError
	if errors.Is(err, store.ErrNotFound) {
		status = http.StatusNotFound
	}
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func badRequest(w http.ResponseWriter, msg string) {
	writeJSON(w, http.StatusBadRequest, map[string]string{"error": msg})
}
