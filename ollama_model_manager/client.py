"""Client for the local Ollama HTTP API.

Uses only the Python standard library so the tool runs with zero external
dependencies. Talks to http://localhost:11434 by default.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable, Optional

from .models import ModelDetail

DEFAULT_BASE_URL = "http://localhost:11434"
REQUEST_TIMEOUT = 60


class OllamaError(Exception):
    """Base error for Ollama client failures."""


class ModelNotFoundError(OllamaError):
    """Raised when a model is not found on the server."""


class OllamaClient:
    """Thin wrapper around Ollama's REST API (/api/tags, /api/pull, ...)."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = REQUEST_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
    ) -> dict:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404:
                raise ModelNotFoundError(f"model not found: {path}") from exc
            raise OllamaError(f"Ollama API error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(f"cannot reach Ollama at {self.base_url}: {exc}") from exc
        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"invalid JSON from Ollama: {body!r}") from exc

    def health(self) -> bool:
        """Return True if the Ollama server responds to /api/tags."""
        try:
            self.list_models()
            return True
        except OllamaError:
            return False

    def list_models(self) -> list[ModelDetail]:
        """List every model available on the Ollama server."""
        data = self._request("GET", "/api/tags")
        models = data.get("models", []) or []
        return [ModelDetail.from_dict(m) for m in models]

    def pull_model(
        self,
        name: str,
        on_progress: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Pull a model, streaming progress to on_progress if provided."""
        last: dict = {}
        for line in self._stream("POST", "/api/pull", {"name": name, "stream": True}):
            if on_progress is not None:
                try:
                    on_progress(line)
                except Exception:
                    # Progress callbacks must never break a pull.
                    pass
            last = line
        if last.get("error"):
            raise OllamaError(f"pull failed for {name}: {last.get('error')}")
        return last

    def _stream(self, method: str, path: str, payload: dict):
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield {"raw": line}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404:
                raise ModelNotFoundError(f"model not found: {path}") from exc
            raise OllamaError(f"Ollama API error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(f"cannot reach Ollama at {self.base_url}: {exc}") from exc

    def delete_model(self, name: str) -> bool:
        """Delete a model. Returns True on success; False if not found."""
        try:
            self._request("DELETE", "/api/delete", {"name": name})
            return True
        except ModelNotFoundError:
            return False

    def show_model(self, name: str) -> ModelDetail:
        """Return the model card details for a model."""
        data = self._request("POST", "/api/show", {"name": name})
        if not data or "name" not in data and "modelfile" not in data:
            raise ModelNotFoundError(f"model not found: {name}")
        return ModelDetail.from_dict(data)
