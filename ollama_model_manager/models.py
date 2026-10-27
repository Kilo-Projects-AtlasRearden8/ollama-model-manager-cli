"""Data models for Ollama model information."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelDetail:
    """A single Ollama model as returned by the API."""

    name: str
    size: int = 0
    digest: str = ""
    modified_at: str = ""
    parameter_size: str = ""
    quantization_level: str = ""
    family: str = ""
    template: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def size_gb(self) -> float:
        return self.size / (1024 ** 3)

    @property
    def short_digest(self) -> str:
        return self.digest[:12] if self.digest else ""

    @classmethod
    def from_dict(cls, data: dict) -> "ModelDetail":
        details = data.get("details", {}) or {}
        return cls(
            name=data.get("name", ""),
            size=int(data.get("size", 0) or 0),
            digest=data.get("digest", ""),
            modified_at=data.get("modified_at", ""),
            parameter_size=details.get("parameter_size", ""),
            quantization_level=details.get("quantization_level", ""),
            family=details.get("family", ""),
            template=data.get("template", ""),
            raw=data,
        )


@dataclass
class RamEstimate:
    """Estimated RAM needed to run a model."""

    model_name: str
    weights_gb: float
    kv_cache_gb: float
    overhead_gb: float
    total_gb: float
    context_length: int
    parameter_size_billions: float | None = None

    @property
    def total_mb(self) -> float:
        return self.total_gb * 1024

    def formatted(self) -> str:
        lines = [
            f"RAM estimate for {self.model_name}",
            f"  Weights (quantized on disk): {self.weights_gb:.2f} GB",
            f"  KV cache (context {self.context_length:,}): {self.kv_cache_gb:.2f} GB",
            f"  Runtime overhead:            {self.overhead_gb:.2f} GB",
            f"  Estimated total RAM:         {self.total_gb:.2f} GB ({self.total_mb:.0f} MB)",
        ]
        if self.parameter_size_billions is not None:
            lines.insert(
                1,
                f"  Parameters:                  ~{self.parameter_size_billions:g} billion",
            )
        return "\n".join(lines)
