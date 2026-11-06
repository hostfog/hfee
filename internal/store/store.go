// Package store keeps downloaded models on disk.
//
// Layout:
//
//	<dir>/blobs/sha256-<hex>        model weights (GGUF)
//	<dir>/manifests/<name>.json     name -> blob, plus model details
package store

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/hostfog/hfee/internal/gguf"
)

var ErrNotFound = errors.New("model not found")

type Details struct {
	Family        string `json:"family"`
	ParameterSize string `json:"parameter_size"`
	Quantization  string `json:"quantization_level"`
	ContextLength int    `json:"context_length"`
}

type Model struct {
	Name     string    `json:"name"`
	Digest   string    `json:"digest"`
	Size     int64     `json:"size"`
	Source   string    `json:"source"`
	Modified time.Time `json:"modified_at"`
	Details  Details   `json:"details"`
}

type Store struct {
	dir string
}

func DefaultDir() string {
	if d := os.Getenv("HFEE_MODELS"); d != "" {
		return d
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ".hfee/models"
	}
	return filepath.Join(home, ".hfee", "models")
}

func Open(dir string) (*Store, error) {
	for _, sub := range []string{"blobs", "manifests"} {
		if err := os.MkdirAll(filepath.Join(dir, sub), 0o755); err != nil {
			return nil, err
		}
	}
	return &Store{dir: dir}, nil
}

// Normalize lower-cases a model name and adds the default tag.
func Normalize(name string) string {
	name = strings.ToLower(strings.TrimSpace(name))
	if i := strings.LastIndex(name, ":"); i < 0 || i < strings.LastIndex(name, "/") {
		name += ":latest"
	}
	return name
}

func (s *Store) manifestPath(name string) string {
	return filepath.Join(s.dir, "manifests", url.QueryEscape(Normalize(name))+".json")
}

func (s *Store) BlobPath(digest string) string {
	return filepath.Join(s.dir, "blobs", strings.Replace(digest, ":", "-", 1))
}

func (s *Store) Get(name string) (Model, error) {
	var m Model
	data, err := os.ReadFile(s.manifestPath(name))
	if errors.Is(err, os.ErrNotExist) {
		return m, fmt.Errorf("%w: %s", ErrNotFound, Normalize(name))
	}
	if err != nil {
		return m, err
	}
	return m, json.Unmarshal(data, &m)
}

// Path returns the weights file for a model.
func (s *Store) Path(name string) (string, error) {
	m, err := s.Get(name)
	if err != nil {
		return "", err
	}
	return s.BlobPath(m.Digest), nil
}

func (s *Store) List() ([]Model, error) {
	entries, err := os.ReadDir(filepath.Join(s.dir, "manifests"))
	if err != nil {
		return nil, err
	}

	var models []Model
	for _, e := range entries {
		data, err := os.ReadFile(filepath.Join(s.dir, "manifests", e.Name()))
		if err != nil {
			continue
		}
		var m Model
		if json.Unmarshal(data, &m) == nil {
			models = append(models, m)
		}
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Modified.After(models[j].Modified) })
	return models, nil
}

func (s *Store) save(m Model) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.manifestPath(m.Name) + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.manifestPath(m.Name))
}

// Delete removes a model, and its weights if no other name uses them.
func (s *Store) Delete(name string) error {
	m, err := s.Get(name)
	if err != nil {
		return err
	}
	if err := os.Remove(s.manifestPath(name)); err != nil {
		return err
	}

	models, err := s.List()
	if err != nil {
		return err
	}
	for _, other := range models {
		if other.Digest == m.Digest {
			return nil
		}
	}
	return os.Remove(s.BlobPath(m.Digest))
}

// register records a downloaded or imported blob under name.
func (s *Store) register(name, digest, source string) (Model, error) {
	path := s.BlobPath(digest)
	st, err := os.Stat(path)
	if err != nil {
		return Model{}, err
	}

	details, err := readDetails(path)
	if err != nil {
		return Model{}, err
	}

	m := Model{
		Name:     Normalize(name),
		Digest:   digest,
		Size:     st.Size(),
		Source:   source,
		Modified: time.Now().UTC(),
		Details:  details,
	}
	return m, s.save(m)
}

var fileTypes = map[uint64]string{
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
	10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
	16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 32: "BF16",
}

func readDetails(path string) (Details, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return Details{}, err
	}
	defer f.Close()

	arch := f.String("general.architecture")
	d := Details{Family: arch, ContextLength: f.Int(arch+".context_length", 0)}
	if ft, ok := f.Uint("general.file_type"); ok {
		d.Quantization = fileTypes[ft]
	}

	var params int
	for _, t := range f.Tensors {
		params += t.Elements()
	}
	switch {
	case params >= 1e9:
		d.ParameterSize = fmt.Sprintf("%.1fB", float64(params)/1e9)
	default:
		d.ParameterSize = fmt.Sprintf("%.0fM", float64(params)/1e6)
	}
	return d, nil
}
