"""Tests for ollama-model-manager against a mock Ollama server."""

from __future__ import annotations

import io
import json
import sys
from unittest.mock import patch

import pytest

from ollama_model_manager import __version__
from ollama_model_manager.card import render_card
from ollama_model_manager.client import (
    DEFAULT_BASE_URL,
    ModelNotFoundError,
    OllamaClient,
    OllamaError,
)
from ollama_model_manager.compare import compare_models
from ollama_model_manager.models import ModelDetail
from ollama_model_manager.ram import estimate_ram

from mock_server import MockOllamaServer


@pytest.fixture
def server():
    with MockOllamaServer() as srv:
        yield srv


@pytest.fixture
def client(server):
    return OllamaClient(base_url=server.base_url)


def test_default_url():
    assert DEFAULT_BASE_URL == "http://localhost:11434"


def test_list_models_empty(client):
    assert client.list_models() == []


def test_list_models(client):
    client.pull_model("llama3.2:1b")
    client.pull_model("llama3.2:3b")
    models = client.list_models()
    names = {m.name for m in models}
    assert names == {"llama3.2:1b", "llama3.2:3b"}
    for m in models:
        assert m.size_gb > 0
        assert m.short_digest


def test_pull_streams_progress(client):
    events = []

    def on_progress(line):
        events.append(line)

    result = client.pull_model("phi3:mini", on_progress=on_progress)
    assert result.get("status") == "success"
    assert any(e.get("status") == "pulling manifest" for e in events)
    assert "phi3:mini" in {m.name for m in client.list_models()}


def test_pull_model_not_found_propagates(client):
    # Mock server always succeeds on pull, so test the error path through a
    # client pointed at a dead port instead.
    dead = OllamaClient(base_url="http://127.0.0.1:1", timeout=2)
    with pytest.raises(OllamaError):
        dead.pull_model("anything")


def test_delete_model(client):
    client.pull_model("tiny-group")
    assert client.delete_model("tiny-group") is True
    assert client.delete_model("tiny-group") is False
    assert all(m.name != "tiny-group" for m in client.list_models())


def test_delete_missing_returns_false(client):
    assert client.delete_model("does-not-exist") is False


def test_show_model(client):
    client.pull_model("codellama:7b")
    detail = client.show_model("codellama:7b")
    assert detail.name == "codellama:7b"
    assert detail.family == "llama"
    assert detail.parameter_size == "7B"
    assert detail.quantization_level == "Q4_K_M"


def test_show_model_not_found(client):
    with pytest.raises(ModelNotFoundError):
        client.show_model("ghost-model")


def test_health(client):
    assert client.health() is True
    dead = OllamaClient(base_url="http://127.0.0.1:1", timeout=2)
    assert dead.health() is False


def test_render_card():
    # Use 2 GiB = 2 * 1024**3 bytes for exact size
    detail = ModelDetail(
        name="llama3.2",
        size=2 * 1024**3,
        digest="sha256:" + "b" * 64,
        parameter_size="3B",
        quantization_level="Q4_K_M",
        family="llama",
        template="{{ .Prompt }}",
    )
    card = render_card(detail)
    assert "Model card: llama3.2" in card
    assert "Family:      llama" in card
    assert "Parameters:  3B" in card
    assert "2.00 GB" in card
    assert "Prompt template:" in card
    assert "{{ .Prompt }}" in card


def test_render_card_modelfile():
    detail = ModelDetail(
        name="custom",
        raw={"modelfile": "FROM llama3.2\nSYSTEM you are helpful"},
    )
    card = render_card(detail)
    assert "Modelfile:" in card
    assert "FROM llama3.2" in card


def test_compare_models():
    a = ModelDetail(name="small", size=1_000_000_000, parameter_size="1B", family="llama")
    b = ModelDetail(name="big", size=8_000_000_000, parameter_size="8B", family="llama")
    out = compare_models(a, b)
    assert "Model comparison" in out
    assert "small" in out and "big" in out
    assert "8B" in out and "1B" in out
    assert "Size difference" in out


def test_estimate_ram_known_values():
    detail = ModelDetail(name="test-7b", size=4 * 1024**3, parameter_size="7B")
    est = estimate_ram(detail, context_length=4096)
    assert est.model_name == "test-7b"
    assert abs(est.weights_gb - 4.0) < 1e-9
    assert est.parameter_size_billions == 7.0
    assert est.kv_cache_gb > 0
    assert est.total_gb == est.weights_gb + est.kv_cache_gb + est.overhead_gb
    assert est.overhead_gb == 1.0
    formatted = est.formatted()
    assert "Estimated total RAM" in formatted
    assert "GB" in formatted


def test_estimate_ram_no_parameter_size_falls_back():
    detail = ModelDetail(name="mystery", size=2_000_000_000, parameter_size="")
    est = estimate_ram(detail)
    assert est.total_gb > est.weights_gb


def test_version():
    assert __version__ == "0.1.0"


# --- CLI tests -------------------------------------------------------------


def _run_cli(argv):
    old_argv, old_stdout, old_stderr = sys.argv, sys.stdout, sys.stderr
    sys.argv = ["ollama-mm"] + argv
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        from ollama_model_manager.cli import main

        try:
            code = main()
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
        out = sys.stdout.getvalue()
        err = sys.stderr.getvalue()
    finally:
        sys.argv, sys.stdout, sys.stderr = old_argv, old_stdout, old_stderr
    return code, out, err


def test_cli_list(server):
    client = OllamaClient(base_url=server.base_url)
    client.pull_model("llama3.2:1b")
    code, out, err = _run_cli(["list", "--url", server.base_url])
    assert code == 0
    assert "llama3.2:1b" in out
    assert "model(s) found" in out

    code, out, err = _run_cli(["list", "--url", server.base_url, "--json"])
    assert code == 0
    data = json.loads(out)
    assert data[0]["name"] == "llama3.2:1b"


def test_cli_pull(server):
    code, out, err = _run_cli(["pull", "phi3:mini", "--url", server.base_url])
    assert code == 0
    assert "Pull complete: phi3:mini" in out
    assert "pulling manifest" in out


def test_cli_delete(server):
    client = OllamaClient(base_url=server.base_url)
    client.pull_model("to-delete")
    code, out, err = _run_cli(["delete", "to-delete", "--url", server.base_url])
    assert code == 0
    assert "Deleted: to-delete" in out

    code, out, err = _run_cli(["delete", "missing", "--url", server.base_url])
    assert code == 0
    assert "Not found" in out


def test_cli_show(server):
    client = OllamaClient(base_url=server.base_url)
    client.pull_model("codellama:7b")
    code, out, err = _run_cli(["show", "codellama:7b", "--url", server.base_url])
    assert code == 0
    assert "Model card: codellama:7b" in out


def test_cli_show_not_found(server):
    code, out, err = _run_cli(["show", "ghost", "--url", server.base_url])
    assert code == 1
    assert "Error:" in out or "error" in (out + err).lower()


def test_cli_ram(server):
    client = OllamaClient(base_url=server.base_url)
    client.pull_model("llama3.2:1b")
    code, out, err = _run_cli(["ram", "llama3.2:1b", "--url", server.base_url])
    assert code == 0
    assert "Estimated total RAM" in out


def test_cli_ram_custom_context(server):
    client = OllamaClient(base_url=server.base_url)
    client.pull_model("llama3.2:1b")
    code, out, err = _run_cli(
        ["ram", "llama3.2:1b", "--url", server.base_url, "--context-len", "8192"]
    )
    assert code == 0
    assert "context 8,192" in out


def test_cli_compare(server):
    client = OllamaClient(base_url=server.base_url)
    client.pull_model("small-a")
    client.pull_model("big-b")
    code, out, err = _run_cli(
        ["compare", "small-a", "big-b", "--url", server.base_url]
    )
    assert code == 0
    assert "Model comparison" in out
    assert "Size difference" in out


def test_cli_version():
    code, out, err = _run_cli(["version"])
    assert code == 0
    assert f"ollama-mm {__version__}" in out


def test_cli_no_command():
    code, out, err = _run_cli([])
    assert code == 2
    assert "usage" in err.lower() or "usage" in (out + err).lower()
