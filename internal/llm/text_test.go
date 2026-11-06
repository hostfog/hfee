package llm

import "testing"

func TestStopper(t *testing.T) {
	s := stopper{stops: []string{"</end>"}}
	var out string
	for _, piece := range []string{"Hello", " wor", "ld</", "en", "d> ignored"} {
		text, hit := s.push(piece)
		out += text
		if hit {
			break
		}
	}
	if out != "Hello world" {
		t.Errorf("got %q", out)
	}
}

func TestStopperFlushesPartialMatch(t *testing.T) {
	s := stopper{stops: []string{"STOP"}}
	text, _ := s.push("go ST")
	if text != "go " {
		t.Errorf("got %q, want held-back prefix", text)
	}
	if rest := s.flush(); rest != "ST" {
		t.Errorf("flush = %q", rest)
	}
}

func TestUTF8Buffer(t *testing.T) {
	var b utf8Buffer
	euro := []byte("€") // 3 bytes
	if got := b.add(euro[:2]); got != "" {
		t.Errorf("partial rune emitted: %q", got)
	}
	if got := b.add(euro[2:]); got != "€" {
		t.Errorf("got %q", got)
	}
}
