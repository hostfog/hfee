package llm

import (
	"strings"
	"unicode/utf8"
)

func commonPrefix(a, b []int) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}

// utf8Buffer holds back bytes until they form complete UTF-8 characters,
// since a single character can be split across tokens.
type utf8Buffer struct {
	pending []byte
}

func (b *utf8Buffer) add(p []byte) string {
	b.pending = append(b.pending, p...)
	for cut := 0; cut <= utf8.UTFMax && cut <= len(b.pending); cut++ {
		end := len(b.pending) - cut
		if utf8.Valid(b.pending[:end]) {
			out := string(b.pending[:end])
			b.pending = append(b.pending[:0], b.pending[end:]...)
			return out
		}
	}
	return b.flush()
}

func (b *utf8Buffer) flush() string {
	out := string(b.pending)
	b.pending = b.pending[:0]
	return out
}

// stopper withholds text that might be the start of a stop sequence.
type stopper struct {
	stops []string
	buf   string
}

func (s *stopper) push(text string) (string, bool) {
	s.buf += text
	for _, stop := range s.stops {
		if stop == "" {
			continue
		}
		if i := strings.Index(s.buf, stop); i >= 0 {
			out := s.buf[:i]
			s.buf = ""
			return out, true
		}
	}

	hold := 0
	for _, stop := range s.stops {
		for k := min(len(stop)-1, len(s.buf)); k > hold; k-- {
			if strings.HasSuffix(s.buf, stop[:k]) {
				hold = k
				break
			}
		}
	}
	out := s.buf[:len(s.buf)-hold]
	s.buf = s.buf[len(s.buf)-hold:]
	return out, false
}

func (s *stopper) flush() string {
	out := s.buf
	s.buf = ""
	return out
}
