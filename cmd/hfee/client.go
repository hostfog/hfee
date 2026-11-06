package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const defaultHost = "127.0.0.1:11500"

func host() string {
	if h := os.Getenv("HFEE_HOST"); h != "" {
		return h
	}
	return defaultHost
}

func baseURL() string {
	h := host()
	if !strings.Contains(h, "://") {
		h = "http://" + h
	}
	return strings.TrimSuffix(h, "/")
}

type apiError struct {
	Status  int
	Message string
}

func (e *apiError) Error() string { return e.Message }

func isNotFound(err error) bool {
	var ae *apiError
	return errors.As(err, &ae) && ae.Status == http.StatusNotFound
}

func call(ctx context.Context, method, path string, body any) (*http.Response, error) {
	var r io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		r = bytes.NewReader(b)
	}

	req, err := http.NewRequestWithContext(ctx, method, baseURL()+path, r)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		var e struct {
			Error string `json:"error"`
		}
		json.NewDecoder(resp.Body).Decode(&e)
		if e.Error == "" {
			e.Error = resp.Status
		}
		return nil, &apiError{Status: resp.StatusCode, Message: e.Error}
	}
	return resp, nil
}

// get performs a request and decodes a single JSON response into out.
func get(ctx context.Context, method, path string, body, out any) error {
	resp, err := call(ctx, method, path, body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if out == nil {
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// stream performs a request and calls fn for each NDJSON line.
func stream(ctx context.Context, path string, body any, fn func(line []byte) error) error {
	resp, err := call(ctx, http.MethodPost, path, body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for sc.Scan() {
		var e struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(sc.Bytes(), &e) == nil && e.Error != "" {
			return errors.New(e.Error)
		}
		if err := fn(sc.Bytes()); err != nil {
			return err
		}
	}
	return sc.Err()
}

// ensureServer starts `hfee serve` in the background if nothing is listening.
func ensureServer() error {
	ping := func() bool {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		return get(ctx, http.MethodGet, "/api/version", nil, nil) == nil
	}
	if ping() {
		return nil
	}

	exe, err := os.Executable()
	if err != nil {
		return err
	}
	home, _ := os.UserHomeDir()
	logDir := filepath.Join(home, ".hfee")
	os.MkdirAll(logDir, 0o755)
	logFile, err := os.OpenFile(filepath.Join(logDir, "server.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer logFile.Close()

	cmd := exec.Command(exe, "serve")
	cmd.Stdout, cmd.Stderr = logFile, logFile
	detach(cmd)
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("could not start server: %w", err)
	}
	cmd.Process.Release()

	for range 50 {
		time.Sleep(200 * time.Millisecond)
		if ping() {
			return nil
		}
	}
	return fmt.Errorf("server did not start, see %s", logFile.Name())
}
