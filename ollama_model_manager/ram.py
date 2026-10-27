"""RAM usage estimator for Ollama models.

Heuristic (clearly labeled): quantized weights live in roughly the on-disk
size, plus a KV-cache estimate for the context window and a fixed runtime
overhead. Not a guarantee - measure real usage with your own workload.
"""

from __future__ import annotations

import re

from .models import ModelDetail, RamEstimate

# Bytes per token of KV cache per billion parameters (fp16 K+V, approximate).
_KV_BYTES_PER_TOKEN_PER_BILLION = 0.07 * 1_000_000
# Fixed runtime overhead (Python/Ollama process + OS headroom).
_OVERHEAD_GB = 1.0

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([btm]?)", re.IGNORECASE)


def _parse_parameter_size(value: str) -> float | None:
    """Extract billions of parameters from strings like '7B', '13B', '70B'."""
    if not value:
        return None
    match = _NUMBER_RE.search(value.strip())
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "t":
        return number * 1000
    if unit == "m":
        return number / 1000
    # Default Ollama convention is billions ('B'); a bare number is treated as B.
    return number


def estimate_ram(model: ModelDetail, context_length: int = 4096) -> RamEstimate:
    """Estimate RAM (GB) needed to load and run the model."""
    weights_gb = model.size_gb
    params_b = _parse_parameter_size(model.parameter_size)
    if params_b is None:
        # Fallback: rough params from size (4-bit quant ~ 0.5 GB per billion).
        params_b = weights_gb / 0.5 if weights_gb > 0 else 0.0
    kv_cache_gb = (
        params_b * context_length * _KV_BYTES_PER_TOKEN_PER_BILLION
    ) / (1024 ** 3)
    total_gb = weights_gb + kv_cache_gb + _OVERHEAD_GB
    return RamEstimate(
        model_name=model.name,
        weights_gb=round(weights_gb, 4),
        kv_cache_gb=round(kv_cache_gb, 4),
        overhead_gb=_OVERHEAD_GB,
        total_gb=round(total_gb, 4),
        context_length=context_length,
        parameter_size_billions=round(params_b, 3) if params_b else None,
    )
