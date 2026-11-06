package llm

import (
	"strings"

	"github.com/hostfog/hfee/internal/tokenizer"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Template turns a conversation into the prompt format a model was trained on.
// GGUF files ship Jinja templates; rather than run Jinja we recognise the
// common formats from their markers.
type Template struct {
	Name string
	eos  string
}

func detectTemplate(jinja string, tok *tokenizer.Tokenizer) Template {
	t := Template{Name: "chatml", eos: tok.Token(tok.EOS)}
	has := func(s string) bool {
		_, inVocab := tok.Lookup(s)
		return strings.Contains(jinja, s) || (jinja == "" && inVocab)
	}

	switch {
	case has("<|start_header_id|>"):
		t.Name = "llama3"
	case has("<|im_start|>"):
		t.Name = "chatml"
	case strings.Contains(jinja, "[INST]"):
		t.Name = "mistral"
	case strings.Contains(jinja, "<|user|>") || has("<|user|>"):
		t.Name = "zephyr"
	}
	return t
}

// Format renders messages and opens the assistant's turn.
func (t Template) Format(msgs []Message) string {
	var b strings.Builder

	switch t.Name {
	case "llama3":
		for _, m := range msgs {
			b.WriteString("<|start_header_id|>" + m.Role + "<|end_header_id|>\n\n" + strings.TrimSpace(m.Content) + "<|eot_id|>")
		}
		b.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")

	case "mistral":
		system := ""
		for _, m := range msgs {
			switch m.Role {
			case "system":
				system = m.Content + "\n\n"
			case "user":
				b.WriteString("[INST] " + system + m.Content + " [/INST]")
				system = ""
			case "assistant":
				b.WriteString(m.Content + t.eos)
			}
		}

	case "zephyr":
		for _, m := range msgs {
			b.WriteString("<|" + m.Role + "|>\n" + m.Content + t.eos + "\n")
		}
		b.WriteString("<|assistant|>\n")

	default: // chatml
		for _, m := range msgs {
			b.WriteString("<|im_start|>" + m.Role + "\n" + m.Content + "<|im_end|>\n")
		}
		b.WriteString("<|im_start|>assistant\n")
	}
	return b.String()
}
