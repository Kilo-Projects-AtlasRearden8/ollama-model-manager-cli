# ollama-model-manager

A beginner-friendly Python CLI to list, pull, delete, and compare Ollama models locally, with a model-card viewer and RAM estimator. No API keys required — talks directly to your local Ollama server.

Built with care by **Asdale Dynamics, LLC**.

---

## Features

- **List models** – See every model on your Ollama server with size, parameters, quantization, and digest
- **Pull models** – Download new models with live progress output
- **Delete models** – Remove models you no longer need
- **Model cards** – View detailed model information (family, parameters, template, modelfile)
- **RAM estimator** – Get a quick estimate of how much RAM a model needs for a given context length
- **Compare models** – Side-by-side comparison of two models
- **Zero runtime dependencies** – Uses only the Python standard library
- **Local-first** – No telemetry, no cloud calls, no secrets

---

## Installation

Requires Python 3.8+.

```bash
# From this repository
pip install -e .

# Or run directly without installing
python -m ollama_model_manager --help
```

After installing, the `ollama-mm` command is available.

---

## Quick Start

```bash
# Start Ollama if it isn't running
ollama serve

# List your models
ollama-mm list

# Pull a new model
ollama-mm pull llama3.2

# See a model card
ollama-mm show llama3.2

# Estimate RAM for a 4K context window
ollama-mm ram llama3.2

# Estimate RAM for an 8K context window
ollama-mm ram llama3.2 --context-len 8192

# Compare two models
ollama-mm compare llama3.2 phi3:mini

# Delete a model
ollama-mm delete old-model
```

---

## Commands

| Command | Description |
|---------|-------------|
| `list` | List all models on the Ollama server |
| `pull <model>` | Pull a model (streams progress) |
| `delete <model>` | Delete a model |
| `show <model>` | View the model card |
| `ram <model> [--context-len N]` | Estimate RAM needed (default context: 4096) |
| `compare <model1> <model2>` | Side-by-side comparison |
| `version` | Print version |

All commands accept `--url URL` to point at a different Ollama server (default: `http://localhost:11434`).

---

## RAM Estimator – How It Works

The RAM estimate is a **heuristic**, not a guarantee. It breaks down into three parts:

1. **Weights (quantized on disk)** – The model file size as reported by Ollama. Quantized weights in memory are roughly the same size.
2. **KV cache** – Estimated using: `parameters (billions) × context length × 0.07 MB per token per billion params`. This approximates fp16 K+V cache per token.
3. **Runtime overhead** – A fixed ~1 GB for the Python/Ollama process and OS headroom.

Example output:

```
RAM estimate for llama3.2
  Parameters:                  ~3 billion
  Weights (quantized on disk): 2.00 GB
  KV cache (context 4,096):    0.86 GB
  Runtime overhead:            1.00 GB
  Estimated total RAM:         3.86 GB (3953 MB)
```

*Actual usage varies by quantization, backend, and workload. Measure with your own setup for precision.*

---

## Testing

Tests use a mock Ollama server (no real Ollama needed, no API keys).

```bash
# From the repository root
python -m pytest tests/ -v
```

---

## Credits

Built by **Asdale Dynamics, LLC** — beginner-friendly AI education, books, guides, and practical local-AI tools.

- Founder: Dylan Asdale (RN turned AI author)
- Book: *The Massive AI Manual*
- Hub: <https://dylanasdale.github.io>

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Copyright (c) 2026 Asdale Dynamics, LLC