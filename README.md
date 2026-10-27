# ollama-model-manager

A friendly Python command-line tool for managing Ollama models on your own computer.
List, pull, delete, compare, view model cards, and estimate RAM usage — all local, no
API keys, no cloud, no fuss.

Made with care by **Asdale Dynamics, LLC**.

---

## Who this is for

If you're new to running local AI models and want a simple way to manage them without
wrestling with complicated commands, this tool is for you. It's built to be clear,
beginner-friendly, and safe to use on your own machine.

---

## What it does

| Command | What it does |
|---------|--------------|
| `list` | Shows every model on your Ollama server, with size, parameters, and format |
| `pull <model>` | Downloads a new model from Ollama (watches progress for you) |
| `delete <model>` | Removes a model you no longer need |
| `show <model>` | Opens the model card: family, size, quantization, prompt template, modelfile |
| `ram <model>` | Estimates how much RAM a model needs to run |
| `compare <m1> <m2>` | Puts two models side by side so you can see the differences |
| `version` | Prints the version number |

---

## Install

You need Python 3.8 or newer.

```bash
# Clone or download this folder, then:
pip install -e .
```

That installs the `ollama-mm` command. You can also run it without installing:

```bash
python -m ollama_model_manager --help
```

---

## Getting started (step by step)

### 1. Start Ollama

Make sure Ollama is running on your computer:

```bash
ollama serve
```

### 2. See what models you already have

```bash
ollama-mm list
```

You'll see a table with each model's name, size, parameter count, quantization, and digest.

### 3. Pull a new model

```bash
ollama-mm pull llama3.2
```

You'll see a progress bar as it downloads. When it's done, you're ready to use it.

### 4. Look at a model card

```bash
ollama-mm show llama3.2
```

This shows the model's family (like "llama"), parameter size (like "3B"), quantization
(like "Q4_K_M"), on-disk size, digest, last modified date, prompt template, and modelfile
if one exists.

### 5. Estimate RAM

```bash
ollama-mm ram llama3.2
```

This estimates how much RAM your computer needs to run the model, including the model
weights, a KV-cache estimate for your context window, and a small overhead allowance.
The default context is 4096 tokens. Change it with `--context-len`:

```bash
ollama-mm ram llama3.2 --context-len 8192
```

### 6. Compare two models

```bash
ollama-mm compare llama3.2 phi3:mini
```

You'll see family, parameters, quantization, size, digest, and modified date side by side,
plus a size difference note.

### 7. Delete a model

```bash
ollama-mm delete old-model
```

---

## How the RAM estimate works

The RAM estimate is a rough guide, not a guarantee. It adds up three pieces:

1. **Model weights** – The on-disk size Ollama reports. Quantized weights in memory are
   roughly the same size.
2. **KV cache** – Estimated using: `parameters (billions) × context length × 0.07 MB per
   token per billion parameters`. This approximates the fp16 key+value cache per token.
3. **Runtime overhead** – A fixed ~1 GB for the Ollama process and operating system.

Example output:

```
RAM estimate for llama3.2
  Parameters:                  ~3 billion
  Weights (quantized on disk): 2.00 GB
  KV cache (context 4,096):    0.86 GB
  Runtime overhead:            1.00 GB
  Estimated total RAM:         3.86 GB (3953 MB)
```

Your real usage may vary based on quantization, backend, and what else is running. Measure
with your own setup for precision.

---

## Running the tests

Tests use a mock Ollama server — no real Ollama installation and no API keys needed:

```bash
python -m pytest tests/ -v
```

You should see all 26 tests pass.

---

## Credits

Built by **Asdale Dynamics, LLC** — beginner-friendly AI education, books, guides, and
practical local-AI tools.

- Founder: Dylan Asdale (registered nurse turned AI author)
- Book: *The Massive AI Manual*
- Hub: https://dylanasdale.github.io

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Copyright (c) 2026 Asdale Dynamics, LLC