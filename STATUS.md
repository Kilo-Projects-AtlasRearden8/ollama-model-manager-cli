# STATUS

ollama-model-manager — beginner-friendly Ollama model CLI

## What was built

A Python command-line tool for managing Ollama models locally, with no API keys,
no cloud calls, and no telemetry. Everything talks directly to a local Ollama
server (http://localhost:11434 by default).

### Commands
- `list` — List models on the Ollama server (with optional JSON output)
- `pull <model>` — Download a model with live progress
- `delete <model>` — Remove a model (graceful if not found)
- `show <model>` — View a model card (family, params, quant, template, modelfile)
- `ram <model>` — Estimate RAM needed (weights + KV cache + overhead)
- `compare <m1> <m2>` — Side-by-side comparison of two models
- `version` — Print the version

### Modules
- `ollama_model_manager/client.py` — Thin stdlib-only Ollama HTTP API wrapper
- `ollama_model_manager/models.py` — ModelDetail and RamEstimate data classes
- `ollama_model_manager/ram.py` — RAM usage heuristic estimator
- `ollama_model_manager/card.py` — Model-card plain-text renderer
- `ollama_model_manager/compare.py` — Side-by-side model comparison
- `ollama_model_manager/cli.py` — argparse CLI with subcommands
- `tests/mock_server.py` — Mock Ollama HTTP server for tests
- `tests/test_ollama_manager.py` — 26 tests covering all functionality

### Files
- `LICENSE` — MIT License, Copyright (c) 2026 Asdale Dynamics, LLC
- `README.md` — Beginner-friendly guide with step-by-step instructions
- `pyproject.toml` — Package config, defines `ollama-mm` console script
- `.gitignore` — Excludes Python build artifacts and venv

## How to run it

### Install
```bash
pip install -e .
```

### Use
```bash
# Start Ollama first (separate terminal)
ollama serve

# List models
ollama-mm list

# Pull a model
ollama-mm pull llama3.2

# View a model card
ollama-mm show llama3.2

# Estimate RAM
ollama-mm ram llama3.2 --context-len 4096

# Compare two models
ollama-mm compare llama3.2 phi3:mini

# Delete a model
ollama-mm delete old-model

# Help
ollama-mm --help
ollama-mm list --help
```

### Without installing
```bash
python -m ollama_model_manager --help
```

### Run tests
```bash
python -m pytest tests/ -v
```

All tests use a mock Ollama server — no real Ollama installation and no API keys required.

## Verification results

- **Tests**: 26 passed, 0 failed
- **CLI manual test**: All commands (list, pull, delete, show, ram, compare, version)
  verified end-to-end against the mock server
- **Installed command**: `ollama-mm` entry point works after `pip install -e .`
- **No secrets**: No API keys, tokens, or personal data in any source file
- **No .github/workflows**: None present
- **License**: MIT, exact copyright line "Copyright (c) 2026 Asdale Dynamics, LLC"
- **README credit**: Asdale Dynamics, LLC credited throughout

## Known gaps

1. **RAM estimator is heuristic** — The KV-cache calculation is an approximation.
   Real RAM usage varies by quantization, backend, and workload. Measure with your
   own setup for precision.
2. **Progress parsing** — Pull progress is parsed from Ollama's ndjson stream. If
   Ollama changes its progress format, the progress display may need updating.
3. **No configuration file** — All options are CLI flags. A config file could be
   added later for default URL, context length, etc.
4. **No interactive prompts** — All actions require explicit arguments; no TUI.
5. **Python 3.8+** — Uses `from __future__ import annotations` and `X | None` type
   syntax guarded by the future import; tested on Python 3.13.
6. **Single server only** — Points at one Ollama URL at a time via `--url`.

## Commit

Single commit on `master` branch. Plain commit message, no co-author trailers.
Never pushed, no remote repository created.