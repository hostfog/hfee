# HFee

By [HostFog Ltd](https://github.com/hostfog).

Run large language models on your own machine. HFee downloads GGUF models,
runs them on the CPU and serves them over an Ollama-compatible API and an
OpenAI-compatible API.

It is written in pure Go with no third-party dependencies, so it builds to a
single binary for Linux, macOS and Windows.

```
$ hfee run qwen2.5:0.5b
>>> What is the capital of France?
The capital of France is Paris.
```

## Install

Requires Go 1.22 or newer.

```bash
go install github.com/hostfog/hfee/cmd/hfee@latest
```

Or build from source:

```bash
git clone https://github.com/hostfog/hfee
cd hfee
go build -o hfee ./cmd/hfee
```

## Usage

```
hfee run <model> [prompt]    chat with a model (downloads it if needed)
hfee pull <model>            download a model
hfee create <name> <file>    add a local .gguf file
hfee list                    list downloaded models
hfee show <model>            show model details
hfee ps                      list loaded models
hfee stop <model>            unload a model
hfee rm <model>              delete a model
hfee serve                   start the API server
```

`hfee run` starts the server in the background if it isn't running.
In a chat, use `"""` for multi-line input, `/clear` to reset, `/bye` to exit,
and Ctrl+C to stop a reply. Add `--verbose` to see token rates.

### Models

| Name                                   | Source                          |
|----------------------------------------|---------------------------------|
| `qwen2.5:0.5b`, `llama3.2:1b`, ...     | the Ollama model library        |
| `hf.co/<user>/<repo>[:<quant>]`        | a GGUF file on Hugging Face     |
| `hfee create mymodel ./model.gguf`     | a file you already have         |

Supported architectures: Llama (1/2/3.x), Mistral, Qwen 2/2.5, Qwen 3, and
models built on them (TinyLlama, DeepSeek-R1 distills of Qwen and Llama).

Supported weight types: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.

Small models run best on a CPU: start with `qwen2.5:0.5b`, `qwen3:0.6b`
or `llama3.2:1b`.

## API

The server listens on `127.0.0.1:11500`.

Ollama-compatible:

```bash
curl http://localhost:11500/api/chat -d '{
  "model": "qwen2.5:0.5b",
  "messages": [{ "role": "user", "content": "Hello" }]
}'
```

Endpoints: `POST /api/generate`, `POST /api/chat`, `POST /api/pull`,
`GET /api/tags`, `POST /api/show`, `DELETE /api/delete`, `GET /api/ps`,
`GET /api/version`.

OpenAI-compatible, so existing SDKs and apps work by changing the base URL:

```bash
curl http://localhost:11500/v1/chat/completions -d '{
  "model": "qwen2.5:0.5b",
  "stream": true,
  "messages": [{ "role": "user", "content": "Hello" }]
}'
```

Endpoints: `POST /v1/chat/completions`, `GET /v1/models`.

Generation options (`options` in the Ollama API): `temperature`, `top_k`,
`top_p`, `min_p`, `repeat_penalty`, `repeat_last_n`, `seed`, `num_predict`,
`stop`. The context window is set for the server with `HFEE_NUM_CTX`.

## Configuration

| Variable          | Default            |                                   |
|-------------------|--------------------|-----------------------------------|
| `HFEE_HOST`       | `127.0.0.1:11500`  | address to listen on / connect to |
| `HFEE_MODELS`     | `~/.hfee/models`   | where models are stored           |
| `HFEE_KEEP_ALIVE` | `5m`               | how long an idle model stays loaded (`-1` = forever) |
| `HFEE_NUM_CTX`    | `4096`             | context window per conversation, in tokens |
| `HFEE_NUM_PARALLEL` | `4`              | conversations answered at the same time |

## How it works

```
cmd/hfee            CLI
internal/server     HTTP API and model loading
internal/store      model downloads and storage
internal/llm        transformer forward pass, sampling, chat templates
internal/tokenizer  SentencePiece and byte-level BPE tokenizers
internal/gguf       GGUF file reader (memory-mapped)
internal/ggml       quantised weight formats and matrix kernels
```

- Weights are memory-mapped and used in their quantised form. Activations
  are quantised to int8 and multiplied with integer AVX2 kernels (Go
  assembly) on x86-64, with a portable Go fallback elsewhere; work is
  spread across all CPU cores.
- Several conversations are answered at once: each step runs one token from
  every active conversation through the model together, so the weights are
  read once for all of them.
- The key/value cache is kept between requests, so in a chat only the new
  message is processed.
- Chat templates are recognised from the model's metadata (Llama 3, ChatML,
  Mistral, Zephyr).

HFee runs on the CPU only, so speed depends mostly on memory bandwidth and
core count. On a 2-core VPS, `qwen2.5:0.5b` generates about 15 tokens per
second and `llama3.2:1b` about 7.

## Licence

MIT © 2026 HostFog Ltd. HFee AI, a chat website built on HFee, is at
[hostfog/hfee-ai](https://github.com/hostfog/hfee-ai).
