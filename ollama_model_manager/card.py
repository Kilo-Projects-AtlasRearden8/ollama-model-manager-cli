"""Model-card viewer: render Ollama /api/show output as plain text."""

from __future__ import annotations

from .models import ModelDetail


def render_card(detail: ModelDetail) -> str:
    lines = [
        f"Model card: {detail.name}",
        f"  Family:      {detail.family or '(not reported)'}",
        f"  Parameters:  {detail.parameter_size or '(not reported)'}",
        f"  Quantization:{detail.quantization_level or '(not reported)'}",
        f"  Size:        {detail.size_gb:.2f} GB ({detail.size:,} bytes)",
        f"  Digest:      {detail.short_digest}",
        f"  Modified:    {detail.modified_at or '(not reported)'}",
    ]
    if detail.template:
        lines.append("")
        lines.append("Prompt template:")
        lines.append(detail.template)
    raw = detail.raw or {}
    if raw.get("modelfile"):
        lines.append("")
        lines.append("Modelfile:")
        lines.append(str(raw["modelfile"]))
    return "\n".join(lines)
