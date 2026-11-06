// Package tokenizer implements the SentencePiece and byte-level BPE
// tokenizers described by GGUF metadata.
package tokenizer

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/hostfog/hfee/internal/gguf"
)

const (
	typeNormal      = 1
	typeUnknown     = 2
	typeControl     = 3
	typeUserDefined = 4
	typeByte        = 6
)

type kind int

const (
	spm kind = iota // SentencePiece (llama, mistral)
	bpe             // byte-level BPE (llama 3, qwen)
)

type Tokenizer struct {
	kind   kind
	tokens []string
	scores []float32
	types  []int32
	index  map[string]int
	ranks  map[string]int // "left right" -> merge rank
	split  func(string) []string

	special map[byte][]string // special token strings by first byte, longest first

	BOS            int
	EOS            int
	EOT            int
	addBOS         bool
	addSpacePrefix bool
}

func FromGGUF(f *gguf.File) (*Tokenizer, error) {
	t := &Tokenizer{
		tokens:  f.Strings("tokenizer.ggml.tokens"),
		scores:  f.Floats("tokenizer.ggml.scores"),
		types:   f.Ints("tokenizer.ggml.token_type"),
		index:   map[string]int{},
		special: map[byte][]string{},
		BOS:     f.Int("tokenizer.ggml.bos_token_id", -1),
		EOS:     f.Int("tokenizer.ggml.eos_token_id", -1),
		EOT:     f.Int("tokenizer.ggml.eot_token_id", -1),
	}
	if len(t.tokens) == 0 {
		return nil, fmt.Errorf("model has no vocabulary")
	}

	switch m := f.String("tokenizer.ggml.model"); m {
	case "llama":
		t.kind = spm
		t.scores = mergeScores(t.scores, t.tokens)
		t.addBOS = f.Bool("tokenizer.ggml.add_bos_token", true)
		t.addSpacePrefix = f.Bool("tokenizer.ggml.add_space_prefix", true)
	case "gpt2":
		t.kind = bpe
		t.addBOS = f.Bool("tokenizer.ggml.add_bos_token", false)
		t.ranks = map[string]int{}
		for i, m := range f.Strings("tokenizer.ggml.merges") {
			t.ranks[m] = i
		}
		t.split = preTokenizer(f.String("tokenizer.ggml.pre"))
	default:
		return nil, fmt.Errorf("unsupported tokenizer %q", m)
	}

	for i, tok := range t.tokens {
		t.index[tok] = i
		if tt := t.tokenType(i); (tt == typeControl || tt == typeUserDefined) && tok != "" {
			t.special[tok[0]] = append(t.special[tok[0]], tok)
		}
	}
	for _, list := range t.special {
		sort.Slice(list, func(i, j int) bool { return len(list[i]) > len(list[j]) })
	}
	return t, nil
}

// mergeScores returns usable SentencePiece scores. Some converters write all
// zeros. Vocabularies are ordered by merge priority, so fall back to -id,
// except for runs of spaces, which SentencePiece adds with the lowest score.
func mergeScores(scores []float32, tokens []string) []float32 {
	if len(scores) == len(tokens) {
		for _, s := range scores {
			if s != scores[0] {
				return scores
			}
		}
	}
	out := make([]float32, len(tokens))
	for i, tok := range tokens {
		out[i] = -float32(i)
		if len(tok) > len("▁") && strings.Trim(tok, "▁") == "" {
			out[i] = -1e9
		}
	}
	return out
}

func (t *Tokenizer) tokenType(id int) int32 {
	if id < len(t.types) {
		return t.types[id]
	}
	return typeNormal
}

func (t *Tokenizer) VocabSize() int { return len(t.tokens) }

// Token returns the raw vocabulary entry for id.
func (t *Tokenizer) Token(id int) string {
	if id < 0 || id >= len(t.tokens) {
		return ""
	}
	return t.tokens[id]
}

// Lookup returns the id of an exact vocabulary entry.
func (t *Tokenizer) Lookup(s string) (int, bool) {
	id, ok := t.index[s]
	return id, ok
}

// IsEnd reports whether generating id should end the response.
func (t *Tokenizer) IsEnd(id int) bool {
	return id == t.EOS || id == t.EOT || t.tokenType(id) == typeControl
}

// Encode converts text to token ids. Special tokens written in the text
// (such as <|im_start|>) are recognised and kept as single tokens.
func (t *Tokenizer) Encode(text string, addSpecial bool) []int {
	var ids []int
	if addSpecial && t.addBOS && t.BOS >= 0 {
		ids = append(ids, t.BOS)
	}

	prevSpecial := true
	for len(text) > 0 {
		i, tok := t.nextSpecial(text)
		if i > 0 {
			ids = t.encodeText(ids, text[:i], prevSpecial)
			prevSpecial = false
		}
		if tok == "" {
			break
		}
		ids = append(ids, t.index[tok])
		prevSpecial = true
		text = text[i+len(tok):]
	}
	return ids
}

// nextSpecial finds the first special token in s. It returns len(s), "" if there is none.
func (t *Tokenizer) nextSpecial(s string) (int, string) {
	for i := 0; i < len(s); i++ {
		for _, tok := range t.special[s[i]] {
			if strings.HasPrefix(s[i:], tok) {
				return i, tok
			}
		}
	}
	return len(s), ""
}

func (t *Tokenizer) encodeText(ids []int, text string, prevSpecial bool) []int {
	if t.kind == spm {
		return t.encodeSPM(ids, text, prevSpecial)
	}
	return t.encodeBPE(ids, text)
}

func (t *Tokenizer) encodeSPM(ids []int, text string, prevSpecial bool) []int {
	if t.addSpacePrefix && prevSpecial {
		text = " " + text
	}
	text = strings.ReplaceAll(text, " ", "▁")

	syms := strings.Split(text, "")
	syms = merge(syms, func(left, right string) (float64, bool) {
		id, ok := t.index[left+right]
		if !ok {
			return 0, false
		}
		return -float64(t.scores[id]), true
	})

	for _, s := range syms {
		if id, ok := t.index[s]; ok {
			ids = append(ids, id)
			continue
		}
		for _, b := range []byte(s) {
			if id, ok := t.index[fmt.Sprintf("<0x%02X>", b)]; ok {
				ids = append(ids, id)
			}
		}
	}
	return ids
}

func (t *Tokenizer) encodeBPE(ids []int, text string) []int {
	for _, word := range t.split(text) {
		var sb strings.Builder
		for _, b := range []byte(word) {
			sb.WriteRune(byteToRune[b])
		}

		syms := strings.Split(sb.String(), "")
		syms = merge(syms, func(left, right string) (float64, bool) {
			r, ok := t.ranks[left+" "+right]
			return float64(r), ok
		})

		for _, s := range syms {
			if id, ok := t.index[s]; ok {
				ids = append(ids, id)
			}
		}
	}
	return ids
}

// Piece returns the bytes a token decodes to. Control tokens decode to nothing.
func (t *Tokenizer) Piece(id int) []byte {
	if id < 0 || id >= len(t.tokens) {
		return nil
	}
	tok := t.tokens[id]

	switch t.tokenType(id) {
	case typeControl:
		return nil
	case typeUserDefined:
		return []byte(tok)
	case typeByte:
		if v, err := strconv.ParseUint(strings.TrimSuffix(strings.TrimPrefix(tok, "<0x"), ">"), 16, 8); err == nil {
			return []byte{byte(v)}
		}
	}

	if t.kind == spm {
		return []byte(strings.ReplaceAll(tok, "▁", " "))
	}

	out := make([]byte, 0, len(tok))
	for _, r := range tok {
		if b, ok := runeToByte[r]; ok {
			out = append(out, b)
		}
	}
	return out
}

// byteToRune is GPT-2's reversible mapping from bytes to printable runes.
var byteToRune, runeToByte = func() ([256]rune, map[rune]byte) {
	var fwd [256]rune
	rev := map[rune]byte{}
	n := 0
	for b := 0; b < 256; b++ {
		printable := (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
		r := rune(b)
		if !printable {
			r = rune(256 + n)
			n++
		}
		fwd[b] = r
		rev[r] = byte(b)
	}
	return fwd, rev
}()
