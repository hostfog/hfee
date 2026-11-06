package store

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"strings"
	"time"
)

// Progress reports download state. Total is 0 when unknown.
type Progress func(status string, completed, total int64)

type source struct {
	url    string
	digest string // "sha256:<hex>", empty if unknown
	size   int64
}

// Pull downloads a model. Names are resolved as:
//
//	hf.co/<user>/<repo>[:<quant>]   a GGUF file from a Hugging Face repository
//	[<namespace>/]<model>[:<tag>]   the Ollama model library
func (s *Store) Pull(ctx context.Context, name string, progress Progress) (Model, error) {
	name = Normalize(name)
	progress("pulling manifest", 0, 0)

	var src source
	var err error
	if repo, ok := strings.CutPrefix(name, "hf.co/"); ok {
		src, err = resolveHuggingFace(ctx, repo)
	} else if repo, ok := strings.CutPrefix(name, "huggingface.co/"); ok {
		src, err = resolveHuggingFace(ctx, repo)
	} else {
		src, err = resolveLibrary(ctx, name)
	}
	if err != nil {
		return Model{}, err
	}

	if src.digest != "" {
		if _, err := os.Stat(s.BlobPath(src.digest)); err == nil {
			progress("using existing weights", src.size, src.size)
			return s.register(name, src.digest, src.url)
		}
	}

	digest, err := s.download(ctx, src, progress)
	if err != nil {
		return Model{}, err
	}

	progress("writing manifest", 0, 0)
	return s.register(name, digest, src.url)
}

// Import copies a local GGUF file into the store under name.
func (s *Store) Import(name, file string) (Model, error) {
	in, err := os.Open(file)
	if err != nil {
		return Model{}, err
	}
	defer in.Close()

	tmp, err := os.CreateTemp(s.BlobPath(""), "import-*")
	if err != nil {
		return Model{}, err
	}
	defer os.Remove(tmp.Name())

	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(tmp, h), in); err != nil {
		tmp.Close()
		return Model{}, err
	}
	if err := tmp.Close(); err != nil {
		return Model{}, err
	}

	digest := "sha256:" + hex.EncodeToString(h.Sum(nil))
	if err := os.Rename(tmp.Name(), s.BlobPath(digest)); err != nil {
		return Model{}, err
	}
	return s.register(name, digest, file)
}

func resolveLibrary(ctx context.Context, name string) (source, error) {
	host := "registry.ollama.ai"
	repo, tag, _ := strings.Cut(name, ":")
	if first, rest, ok := strings.Cut(repo, "/"); ok && strings.Contains(first, ".") {
		host, repo = first, rest
	}
	if !strings.Contains(repo, "/") {
		repo = "library/" + repo
	}

	var manifest struct {
		Layers []struct {
			MediaType string `json:"mediaType"`
			Digest    string `json:"digest"`
			Size      int64  `json:"size"`
		} `json:"layers"`
	}
	base := fmt.Sprintf("https://%s/v2/%s", host, repo)
	err := getJSON(ctx, base+"/manifests/"+tag, &manifest,
		"Accept", "application/vnd.docker.distribution.manifest.v2+json")
	if err != nil {
		return source{}, fmt.Errorf("model %s not found in library: %w", name, err)
	}

	for _, l := range manifest.Layers {
		if l.MediaType == "application/vnd.ollama.image.model" {
			return source{url: base + "/blobs/" + l.Digest, digest: l.Digest, size: l.Size}, nil
		}
	}
	return source{}, fmt.Errorf("model %s has no weights layer", name)
}

// preferred quantisations when a Hugging Face name has no tag
var defaultQuants = []string{"q4_k_m", "q4_0", "q8_0"}

func resolveHuggingFace(ctx context.Context, name string) (source, error) {
	repo, quant, _ := strings.Cut(name, ":")

	var info struct {
		Siblings []struct {
			File string `json:"rfilename"`
			Size int64  `json:"size"`
			LFS  *struct {
				SHA256 string `json:"sha256"`
				Size   int64  `json:"size"`
			} `json:"lfs"`
		} `json:"siblings"`
	}
	if err := getJSON(ctx, "https://huggingface.co/api/models/"+repo+"?blobs=true", &info); err != nil {
		return source{}, fmt.Errorf("repository %s: %w", repo, err)
	}

	type file struct {
		name   string
		size   int64
		digest string
	}
	var files []file
	for _, f := range info.Siblings {
		lower := strings.ToLower(f.File)
		if !strings.HasSuffix(lower, ".gguf") || strings.Contains(lower, "-of-0") {
			continue // skip non-GGUF and split files
		}
		gf := file{name: f.File, size: f.Size}
		if f.LFS != nil {
			gf.digest, gf.size = "sha256:"+f.LFS.SHA256, f.LFS.Size
		}
		files = append(files, gf)
	}
	if len(files) == 0 {
		return source{}, fmt.Errorf("repository %s has no single-file GGUF models", repo)
	}

	wanted := []string{quant}
	if quant == "latest" {
		wanted = defaultQuants
	}
	for _, q := range wanted {
		for _, f := range files {
			if strings.Contains(strings.ToLower(path.Base(f.name)), q) {
				url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repo, f.name)
				return source{url: url, digest: f.digest, size: f.size}, nil
			}
		}
	}

	names := make([]string, len(files))
	for i, f := range files {
		names[i] = path.Base(f.name)
	}
	return source{}, fmt.Errorf("no %s file in %s (available: %s)", quant, repo, strings.Join(names, ", "))
}

// download fetches src into the blob store, resuming a partial download if present.
func (s *Store) download(ctx context.Context, src source, progress Progress) (string, error) {
	partial := s.BlobPath("partial-" + strings.TrimPrefix(src.digest, "sha256:"))
	if src.digest == "" {
		partial = s.BlobPath(fmt.Sprintf("partial-%x", sha256.Sum256([]byte(src.url))))
	}

	f, err := os.OpenFile(partial, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	offset, err := io.Copy(h, f)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, src.url, nil)
	if err != nil {
		return "", err
	}
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusPartialContent:
	case http.StatusOK:
		// server ignored the range: start over
		offset = 0
		h.Reset()
		if err := f.Truncate(0); err != nil {
			return "", err
		}
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			return "", err
		}
	case http.StatusRequestedRangeNotSatisfiable:
		// already complete
	default:
		return "", fmt.Errorf("download failed: %s", resp.Status)
	}

	total := src.size
	if total == 0 && resp.ContentLength > 0 {
		total = offset + resp.ContentLength
	}

	status := "pulling " + shortDigest(src.digest)
	w := &progressWriter{status: status, done: offset, total: total, report: progress}
	progress(status, offset, total)
	if resp.StatusCode != http.StatusRequestedRangeNotSatisfiable {
		if _, err := io.Copy(io.MultiWriter(f, h, w), resp.Body); err != nil {
			return "", err
		}
	}
	progress(status, w.done, max(total, w.done))

	progress("verifying sha256 digest", 0, 0)
	digest := "sha256:" + hex.EncodeToString(h.Sum(nil))
	if src.digest != "" && digest != src.digest {
		f.Close()
		os.Remove(partial)
		return "", fmt.Errorf("digest mismatch: expected %s, got %s", src.digest, digest)
	}

	if err := f.Close(); err != nil {
		return "", err
	}
	return digest, os.Rename(partial, s.BlobPath(digest))
}

type progressWriter struct {
	status      string
	done, total int64
	last        time.Time
	report      Progress
}

func (w *progressWriter) Write(p []byte) (int, error) {
	w.done += int64(len(p))
	if time.Since(w.last) > 200*time.Millisecond {
		w.last = time.Now()
		w.report(w.status, w.done, w.total)
	}
	return len(p), nil
}

func getJSON(ctx context.Context, url string, v any, headers ...string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	for i := 0; i+1 < len(headers); i += 2 {
		req.Header.Set(headers[i], headers[i+1])
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return errors.New(resp.Status)
	}
	return json.NewDecoder(resp.Body).Decode(v)
}

func shortDigest(d string) string {
	d = strings.TrimPrefix(d, "sha256:")
	if len(d) > 12 {
		d = d[:12]
	}
	if d == "" {
		return "model"
	}
	return d
}
