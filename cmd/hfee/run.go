package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"time"
)

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func run(args []string) error {
	verbose := false
	var rest []string
	for _, a := range args {
		if a == "--verbose" {
			verbose = true
		} else {
			rest = append(rest, a)
		}
	}
	if len(rest) == 0 {
		return errors.New("usage: hfee run <model> [prompt]")
	}
	model := rest[0]

	err := get(context.Background(), http.MethodPost, "/api/show", map[string]string{"model": model}, nil)
	if isNotFound(err) {
		err = pull(model)
	}
	if err != nil {
		return err
	}

	prompt := strings.Join(rest[1:], " ")
	if !isTerminal(os.Stdin) {
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}
		prompt = strings.TrimSpace(strings.TrimSpace(string(data)) + "\n\n" + prompt)
	}

	if prompt != "" {
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
		defer cancel()
		_, err := chat(ctx, model, []message{{Role: "user", Content: prompt}}, verbose)
		return err
	}

	// load the model up front so the first reply isn't delayed
	if err := get(context.Background(), http.MethodPost, "/api/generate", map[string]string{"model": model}, nil); err != nil {
		return err
	}
	return repl(model, verbose)
}

func repl(model string, verbose bool) error {
	var (
		mu      sync.Mutex
		cancel  context.CancelFunc
		history []message
		system  string
	)

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt)
	defer signal.Stop(sig)
	go func() {
		for range sig {
			mu.Lock()
			if cancel != nil {
				cancel()
			} else {
				fmt.Print("\nUse Ctrl + d or /bye to exit.\n>>> ")
			}
			mu.Unlock()
		}
	}()

	in := bufio.NewReader(os.Stdin)
	fmt.Println(`Type a message, or /? for help. Use """ to begin a multi-line message.`)

	for {
		fmt.Print(">>> ")
		line, err := readMessage(in)
		if err == io.EOF {
			fmt.Println()
			return nil
		}
		if err != nil {
			return err
		}

		switch {
		case line == "":
			continue
		case line == "/bye" || line == "/exit":
			return nil
		case line == "/?" || line == "/help":
			fmt.Println(`  /clear          Forget the conversation
  /system <text>  Set the system message
  /bye            Exit
  """             Start or end a multi-line message`)
			continue
		case line == "/clear":
			history = nil
			fmt.Println("Cleared session context")
			continue
		case strings.HasPrefix(line, "/system"):
			system = strings.TrimSpace(strings.TrimPrefix(line, "/system"))
			fmt.Println("Set system message")
			continue
		case strings.HasPrefix(line, "/"):
			fmt.Printf("Unknown command %s. Type /? for help.\n", strings.Fields(line)[0])
			continue
		}

		history = append(history, message{Role: "user", Content: line})
		msgs := history
		if system != "" {
			msgs = append([]message{{Role: "system", Content: system}}, history...)
		}

		ctx, c := context.WithCancel(context.Background())
		mu.Lock()
		cancel = c
		mu.Unlock()

		reply, err := chat(ctx, model, msgs, verbose)

		mu.Lock()
		cancel = nil
		mu.Unlock()
		c()

		if err != nil && ctx.Err() == nil {
			fmt.Fprintln(os.Stderr, "Error:", err)
			history = history[:len(history)-1]
			continue
		}
		history = append(history, message{Role: "assistant", Content: reply})
	}
}

// readMessage reads one line, or several between """ markers.
func readMessage(in *bufio.Reader) (string, error) {
	line, err := in.ReadString('\n')
	if err != nil && (err != io.EOF || line == "") {
		return "", err
	}
	line = strings.TrimSpace(line)
	if !strings.HasPrefix(line, `"""`) {
		return line, nil
	}

	var b strings.Builder
	b.WriteString(strings.TrimPrefix(line, `"""`))
	for !strings.HasSuffix(strings.TrimSpace(b.String()), `"""`) || b.Len() == 0 {
		fmt.Print("... ")
		next, err := in.ReadString('\n')
		if err != nil && next == "" {
			return "", err
		}
		b.WriteString("\n" + strings.TrimRight(next, "\r\n"))
	}
	return strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(b.String()), `"""`)), nil
}

// chat streams a reply to stdout and returns its full text.
func chat(ctx context.Context, model string, msgs []message, verbose bool) (string, error) {
	var reply strings.Builder
	var final struct {
		TotalDuration      int64 `json:"total_duration"`
		LoadDuration       int64 `json:"load_duration"`
		PromptEvalCount    int   `json:"prompt_eval_count"`
		PromptEvalDuration int64 `json:"prompt_eval_duration"`
		EvalCount          int   `json:"eval_count"`
		EvalDuration       int64 `json:"eval_duration"`
	}

	err := stream(ctx, "/api/chat", map[string]any{"model": model, "messages": msgs}, func(line []byte) error {
		var chunk struct {
			Message message `json:"message"`
			Done    bool    `json:"done"`
		}
		if err := json.Unmarshal(line, &chunk); err != nil {
			return err
		}
		if chunk.Done {
			return json.Unmarshal(line, &final)
		}
		fmt.Print(chunk.Message.Content)
		reply.WriteString(chunk.Message.Content)
		return nil
	})
	fmt.Print("\n\n")

	if err == nil && verbose {
		rate := func(n int, d int64) string {
			if d == 0 {
				return "-"
			}
			return fmt.Sprintf("%.2f tokens/s", float64(n)/time.Duration(d).Seconds())
		}
		fmt.Printf("total duration:       %v\n", time.Duration(final.TotalDuration))
		fmt.Printf("load duration:        %v\n", time.Duration(final.LoadDuration))
		fmt.Printf("prompt eval count:    %d token(s)\n", final.PromptEvalCount)
		fmt.Printf("prompt eval rate:     %s\n", rate(final.PromptEvalCount, final.PromptEvalDuration))
		fmt.Printf("eval count:           %d token(s)\n", final.EvalCount)
		fmt.Printf("eval rate:            %s\n\n", rate(final.EvalCount, final.EvalDuration))
	}
	return reply.String(), err
}

func isTerminal(f *os.File) bool {
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}

type progressBar struct {
	status string
	start  time.Time
	shown  bool
}

func newProgressBar() *progressBar { return &progressBar{} }

func (p *progressBar) update(status string, done, total int64) {
	if status != p.status {
		if p.shown {
			fmt.Println()
		}
		p.status, p.start, p.shown = status, time.Now(), true
	}
	if total <= 0 {
		fmt.Printf("\r%s ", status)
		return
	}

	const width = 30
	frac := float64(done) / float64(total)
	filled := int(frac * width)
	speed := ""
	if secs := time.Since(p.start).Seconds(); secs > 0.5 {
		speed = humanBytes(int64(float64(done)/secs)) + "/s"
	}
	fmt.Printf("\r%s %3d%% ▕%s%s▏ %s/%s %s   ", status, int(frac*100),
		strings.Repeat("█", filled), strings.Repeat(" ", width-filled),
		humanBytes(done), humanBytes(total), speed)
}

func (p *progressBar) finish() {
	if p.shown {
		fmt.Println()
	}
}
