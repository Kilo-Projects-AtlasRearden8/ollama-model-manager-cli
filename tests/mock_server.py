"""Mock Ollama HTTP server for tests.

A tiny threaded http.server that implements the subset of the Ollama API used
by ollama-model-manager: GET /api/tags, POST /api/pull, DELETE /api/delete,
POST /api/show. Models live in a shared, resettable registry so pull/delete/
show interact realistically.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

DEFAULT_PORT = 0  # let the OS pick a free port


class MockOllamaHandler(BaseHTTPRequestHandler):
    registry: Dict[str, dict] = {}

    def log_message(self, *args) -> None:
        # Keep test output quiet.
        pass

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            self._send_json(200, {"models": list(self.registry.values())})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/api/pull":
            data = self._read_json()
            name = data.get("name", "")
            stream = data.get("stream", True)
            existing = self.registry.get(name)
            if existing is None:
                # Vary size by name: "big" gets 8GB, "small" gets 1GB, default 1GB
                if "big" in name.lower():
                    size = 8 * 1024**3
                    param_size = "8B"
                elif "small" in name.lower():
                    size = 1 * 1024**3
                    param_size = "1B"
                else:
                    size = 1 * 1024**3
                    param_size = "7B"
                entry = {
                    "name": name,
                    "model": name,
                    "size": size,
                    "digest": "sha256:" + "a" * 64,
                    "modified_at": "2026-01-01T00:00:00Z",
                    "details": {
                        "parameter_size": param_size,
                        "quantization_level": "Q4_K_M",
                        "family": "llama",
                    },
                }
                self.registry[name] = entry
            else:
                entry = existing
            if not stream:
                self._send_json(200, {"status": "success"})
                return
            # Stream Ollama-style ndjson progress lines.
            lines = [
                {"status": "pulling manifest"},
                {"status": "pulling", "digest": entry["digest"], "total": 100, "completed": 0},
                {"status": "pulling", "digest": entry["digest"], "total": 100, "completed": 50},
                {"status": "pulling", "digest": entry["digest"], "total": 100, "completed": 100},
                {"status": "success"},
            ]
            body = "".join(json.dumps(line) + "\n" for line in lines).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/show":
            data = self._read_json()
            name = data.get("name", "")
            entry = self.registry.get(name)
            if entry is None:
                self._send_json(404, {"error": f"model '{name}' not found"})
                return
            self._send_json(200, entry)
        else:
            self._send_json(404, {"error": "not found"})

    def do_DELETE(self) -> None:
        if self.path == "/api/delete":
            data = self._read_json()
            name = data.get("name", "")
            if name in self.registry:
                del self.registry[name]
                self._send_json(200, {"status": "deleted", "deleted": True})
            else:
                self._send_json(404, {"error": f"model '{name}' not found"})
        else:
            self._send_json(404, {"error": "not found"})


class MockOllamaServer:
    """Context-manager-style wrapped ThreadingHTTPServer."""

    def __init__(self, port: int = DEFAULT_PORT) -> None:
        MockOllamaHandler.registry = {}
        self._server = ThreadingHTTPServer(("127.0.0.1", port), MockOllamaHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self.port = self._server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"

    def start(self) -> "MockOllamaServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> "MockOllamaServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
