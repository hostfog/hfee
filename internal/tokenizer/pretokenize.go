package tokenizer

import "unicode"

// preTokenizer returns the word splitter for a GGUF "tokenizer.ggml.pre" value.
// The splitters are hand-written equivalents of the regular expressions the
// models were trained with, since Go's regexp has no lookahead.
func preTokenizer(name string) func(string) []string {
	switch name {
	case "llama-bpe", "llama3", "smaug-bpe", "dbrx":
		return func(s string) []string { return splitWith(s, llama3Rule(3)) }
	case "qwen2", "deepseek-r1-qwen":
		return func(s string) []string { return splitWith(s, llama3Rule(1)) }
	default:
		return func(s string) []string { return splitWith(s, gpt2Rule) }
	}
}

func splitWith(s string, rule func(rs []rune, i int) int) []string {
	rs := []rune(s)
	var out []string
	for i := 0; i < len(rs); {
		j := rule(rs, i)
		out = append(out, string(rs[i:j]))
		i = j
	}
	return out
}

func isLetter(r rune) bool  { return unicode.IsLetter(r) }
func isNumber(r rune) bool  { return unicode.IsNumber(r) }
func isSpace(r rune) bool   { return unicode.IsSpace(r) }
func isNewline(r rune) bool { return r == '\r' || r == '\n' }
func isSymbol(r rune) bool  { return !isSpace(r) && !isLetter(r) && !isNumber(r) }

func runWhile(rs []rune, i int, ok func(rune) bool) int {
	for i < len(rs) && ok(rs[i]) {
		i++
	}
	return i
}

// contraction matches 's 't 're 've 'm 'll 'd at i and returns the end, or -1.
func contraction(rs []rune, i int, fold bool) int {
	if rs[i] != '\'' || i+1 >= len(rs) {
		return -1
	}
	lower := func(r rune) rune {
		if fold {
			return unicode.ToLower(r)
		}
		return r
	}
	switch lower(rs[i+1]) {
	case 's', 't', 'm', 'd':
		return i + 2
	}
	if i+2 < len(rs) {
		switch string([]rune{lower(rs[i+1]), lower(rs[i+2])}) {
		case "re", "ve", "ll":
			return i + 3
		}
	}
	return -1
}

// trailingSpace matches \s+(?!\S) falling back to \s+.
func trailingSpace(rs []rune, i int) int {
	e := runWhile(rs, i, isSpace)
	if e < len(rs) && e-i > 1 {
		return e - 1
	}
	return e
}

// llama3Rule implements:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,n}|
//	 ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
func llama3Rule(maxDigits int) func([]rune, int) int {
	return func(rs []rune, i int) int {
		if e := contraction(rs, i, true); e > 0 {
			return e
		}

		r := rs[i]
		j := i
		if !isLetter(r) && !isNumber(r) && !isNewline(r) && i+1 < len(rs) && isLetter(rs[i+1]) {
			j++
		}
		if isLetter(rs[j]) {
			return runWhile(rs, j, isLetter)
		}

		if isNumber(r) {
			e := i
			for e < len(rs) && e-i < maxDigits && isNumber(rs[e]) {
				e++
			}
			return e
		}

		j = i
		if r == ' ' {
			j++
		}
		if j < len(rs) && isSymbol(rs[j]) {
			return runWhile(rs, runWhile(rs, j, isSymbol), isNewline)
		}

		if isSpace(r) {
			e := runWhile(rs, i, isSpace)
			for k := e - 1; k >= i; k-- {
				if isNewline(rs[k]) {
					return k + 1
				}
			}
			return trailingSpace(rs, i)
		}
		return i + 1
	}
}

// gpt2Rule implements:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
func gpt2Rule(rs []rune, i int) int {
	if e := contraction(rs, i, false); e > 0 {
		return e
	}

	j := i
	if rs[i] == ' ' {
		j++
	}
	if j < len(rs) {
		for _, class := range []func(rune) bool{isLetter, isNumber, isSymbol} {
			if class(rs[j]) {
				return runWhile(rs, j, class)
			}
		}
	}

	if isSpace(rs[i]) {
		return trailingSpace(rs, i)
	}
	return i + 1
}
