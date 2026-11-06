package tokenizer

import (
	"reflect"
	"testing"
)

func TestLlama3Split(t *testing.T) {
	tests := []struct {
		in   string
		want []string
	}{
		{"Hello world", []string{"Hello", " world"}},
		{"It's 12345!", []string{"It", "'s", " ", "123", "45", "!"}},
		{"a  b\n\n  c", []string{"a", " ", " b", "\n\n", " ", " c"}},
		{"x  ", []string{"x", "  "}},
	}
	for _, tt := range tests {
		if got := splitWith(tt.in, llama3Rule(3)); !reflect.DeepEqual(got, tt.want) {
			t.Errorf("split(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestGPT2Split(t *testing.T) {
	got := splitWith("I'm here  now", gpt2Rule)
	want := []string{"I", "'m", " here", " ", " now"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestMerge(t *testing.T) {
	ranks := map[string]int{"a b": 0, "ab c": 1, "b c": 2}
	got := merge([]string{"a", "b", "c", "b", "c"}, func(l, r string) (float64, bool) {
		n, ok := ranks[l+" "+r]
		return float64(n), ok
	})
	want := []string{"abc", "bc"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}
