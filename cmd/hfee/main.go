// Command hfee runs large language models locally.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"text/tabwriter"
	"time"

	"github.com/hostfog/hfee/internal/server"
	"github.com/hostfog/hfee/internal/store"
)

const usage = `HFee runs large language models on your own machine.

Usage:
  hfee <command> [arguments]

Commands:
  serve                 Start the API server
  run <model> [prompt]  Chat with a model (pulls it if needed)
  pull <model>          Download a model
  create <name> <file>  Add a local .gguf file as a model
  list                  List downloaded models
  show <model>          Show model details
  ps                    List loaded models
  stop <model>          Unload a model from memory
  rm <model>            Delete a model
  version               Print the version

Models:
  llama3.2:1b, qwen2.5:0.5b ...      from the Ollama library
  hf.co/<user>/<repo>[:<quant>]      a GGUF file from Hugging Face

Environment:
  HFEE_HOST          server address (default ` + defaultHost + `)
  HFEE_MODELS        model directory (default ~/.hfee/models)
  HFEE_KEEP_ALIVE    how long models stay loaded (default 5m)
  HFEE_NUM_CTX       context window per conversation (default 4096)
  HFEE_NUM_PARALLEL  conversations answered at once (default 4)
`

func main() {
	log.SetFlags(log.LstdFlags)
	if len(os.Args) < 2 {
		fmt.Print(usage)
		os.Exit(1)
	}

	cmd, args := os.Args[1], os.Args[2:]
	var err error
	switch cmd {
	case "serve":
		err = serve()
	case "run":
		err = withServer(func() error { return run(args) })
	case "pull":
		err = withServer(func() error { return needArgs(args, 1, "pull <model>", func() error { return pull(args[0]) }) })
	case "create":
		err = needArgs(args, 2, "create <name> <file.gguf>", func() error { return create(args[0], args[1]) })
	case "list", "ls":
		err = withServer(list)
	case "show":
		err = withServer(func() error { return needArgs(args, 1, "show <model>", func() error { return show(args[0]) }) })
	case "ps":
		err = withServer(ps)
	case "stop":
		err = withServer(func() error { return needArgs(args, 1, "stop <model>", func() error { return stop(args[0]) }) })
	case "rm":
		err = withServer(func() error { return needArgs(args, 1, "rm <model>", func() error { return remove(args) }) })
	case "version", "--version", "-v":
		fmt.Println("hfee version", server.Version)
	case "help", "--help", "-h":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n%s", cmd, usage)
		os.Exit(1)
	}

	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}

func needArgs(args []string, n int, form string, fn func() error) error {
	if len(args) < n {
		return fmt.Errorf("usage: hfee %s", form)
	}
	return fn()
}

func withServer(fn func() error) error {
	if err := ensureServer(); err != nil {
		return err
	}
	return fn()
}

func openStore() (*store.Store, error) {
	return store.Open(store.DefaultDir())
}

func serve() error {
	st, err := openStore()
	if err != nil {
		return err
	}

	srv := &http.Server{Addr: host(), Handler: server.New(st).Handler()}
	errc := make(chan error, 1)
	go func() {
		log.Printf("HFee %s listening on %s", server.Version, host())
		errc <- srv.ListenAndServe()
	}()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	select {
	case err := <-errc:
		return err
	case <-sig:
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return srv.Shutdown(ctx)
	}
}

func pull(name string) error {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	bar := newProgressBar()
	err := stream(ctx, "/api/pull", map[string]string{"model": name}, func(line []byte) error {
		var p struct {
			Status    string `json:"status"`
			Completed int64  `json:"completed"`
			Total     int64  `json:"total"`
		}
		if err := json.Unmarshal(line, &p); err != nil {
			return err
		}
		bar.update(p.Status, p.Completed, p.Total)
		return nil
	})
	bar.finish()
	return err
}

func create(name, file string) error {
	st, err := openStore()
	if err != nil {
		return err
	}
	fmt.Println("copying model file")
	m, err := st.Import(name, file)
	if err != nil {
		return err
	}
	fmt.Printf("created %s (%s, %s)\n", m.Name, m.Details.ParameterSize, m.Details.Quantization)
	return nil
}

type modelInfo struct {
	Name     string        `json:"name"`
	Digest   string        `json:"digest"`
	Size     int64         `json:"size"`
	Modified time.Time     `json:"modified_at"`
	Source   string        `json:"source"`
	Details  store.Details `json:"details"`
}

func list() error {
	var resp struct {
		Models []modelInfo `json:"models"`
	}
	if err := get(context.Background(), http.MethodGet, "/api/tags", nil, &resp); err != nil {
		return err
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(w, "NAME\tID\tSIZE\tMODIFIED")
	for _, m := range resp.Models {
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", m.Name, shortID(m.Digest), humanBytes(m.Size), since(m.Modified))
	}
	return w.Flush()
}

func show(name string) error {
	var m modelInfo
	if err := get(context.Background(), http.MethodPost, "/api/show", map[string]string{"model": name}, &m); err != nil {
		return err
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintf(w, "  architecture\t%s\n", m.Details.Family)
	fmt.Fprintf(w, "  parameters\t%s\n", m.Details.ParameterSize)
	fmt.Fprintf(w, "  quantization\t%s\n", m.Details.Quantization)
	fmt.Fprintf(w, "  context length\t%d\n", m.Details.ContextLength)
	fmt.Fprintf(w, "  size\t%s\n", humanBytes(m.Size))
	fmt.Fprintf(w, "  source\t%s\n", m.Source)
	return w.Flush()
}

func ps() error {
	var resp struct {
		Models []struct {
			Name    string    `json:"name"`
			Size    int64     `json:"size"`
			Expires time.Time `json:"expires_at"`
		} `json:"models"`
	}
	if err := get(context.Background(), http.MethodGet, "/api/ps", nil, &resp); err != nil {
		return err
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(w, "NAME\tSIZE\tUNTIL")
	for _, m := range resp.Models {
		until := "forever"
		if !m.Expires.IsZero() {
			until = time.Until(m.Expires).Round(time.Second).String() + " from now"
		}
		fmt.Fprintf(w, "%s\t%s\t%s\n", m.Name, humanBytes(m.Size), until)
	}
	return w.Flush()
}

func stop(name string) error {
	return get(context.Background(), http.MethodPost, "/api/generate",
		map[string]any{"model": name, "keep_alive": 0}, nil)
}

func remove(names []string) error {
	for _, name := range names {
		if err := get(context.Background(), http.MethodDelete, "/api/delete", map[string]string{"model": name}, nil); err != nil {
			return err
		}
		fmt.Println("deleted", store.Normalize(name))
	}
	return nil
}

func shortID(digest string) string {
	d := strings.TrimPrefix(digest, "sha256:")
	return d[:min(12, len(d))]
}

func humanBytes(n int64) string {
	const unit = 1000
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for v := n / unit; v >= unit; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
}

func since(t time.Time) string {
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return plural(int(d.Minutes()), "minute")
	case d < 24*time.Hour:
		return plural(int(d.Hours()), "hour")
	default:
		return plural(int(d.Hours()/24), "day")
	}
}

func plural(n int, unit string) string {
	if n == 1 {
		return "1 " + unit + " ago"
	}
	return fmt.Sprintf("%d %ss ago", n, unit)
}
