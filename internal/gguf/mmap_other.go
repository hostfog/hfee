//go:build !unix && !windows

package gguf

import "os"

// mapFile falls back to reading the whole file on platforms without mmap support here.
func mapFile(path string) ([]byte, func() error, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	return data, func() error { return nil }, nil
}
