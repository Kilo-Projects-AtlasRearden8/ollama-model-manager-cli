"""Side-by-side comparison of two Ollama models."""

from __future__ import annotations

from .models import ModelDetail


def _field(label: str, a: str, b: str) -> str:
    label_w = len(label)
    a_w = len(a)
    b_w = len(b)
    col_w = max(label_w, a_w, b_w) + 2
    return f"  {label:<{col_w}} {a:<{col_w}} {b}"


def compare_models(a: ModelDetail, b: ModelDetail) -> str:
    lines = [
        "Model comparison",
        "  " + "=" * 56,
        f"  {'':<24}{a.name:<28}{b.name}",
        f"  {'':<24}{'=' * len(a.name):<28}{'=' * len(b.name)}",
        _field("Family:", a.family or "-", b.family or "-"),
        _field("Parameters:", a.parameter_size or "-", b.parameter_size or "-"),
        _field("Quantization:", a.quantization_level or "-", b.quantization_level or "-"),
        _field("Size:", f"{a.size_gb:.2f} GB", f"{b.size_gb:.2f} GB"),
        _field("Digest:", a.short_digest, b.short_digest),
        _field("Modified:", a.modified_at or "-", b.modified_at or "-"),
    ]
    size_diff = b.size - a.size
    if size_diff:
        bigger = b.name if size_diff > 0 else a.name
        lines.append("")
        lines.append(f"  Size difference: {bigger} is {abs(size_diff) / (1024**3):.2f} GB larger")
    return "\n".join(lines)