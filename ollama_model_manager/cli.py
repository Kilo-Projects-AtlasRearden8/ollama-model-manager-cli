"""Command-line interface for ollama-model-manager.

Commands:
  list        List models on the Ollama server
  pull        Pull a model
  delete      Delete a model
  show        View a model card
  ram         Estimate RAM needed to run a model
  compare     Compare two models side by side
  version     Print the version
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from . import __version__
from .card import render_card
from .client import DEFAULT_BASE_URL, ModelNotFoundError, OllamaClient, OllamaError
from .compare import compare_models
from .ram import estimate_ram


def _build_parser() -> argparse.ArgumentParser:
    # Global parser for shared options
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--url",
        default=DEFAULT_BASE_URL,
        help=f"Ollama server base URL (default: {DEFAULT_BASE_URL})",
    )

    parser = argparse.ArgumentParser(
        prog="ollama-mm",
        description="Manage Ollama models locally: list, pull, delete, compare, view cards, estimate RAM.",
        parents=[parent_parser],
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List models on the Ollama server", parents=[parent_parser])
    p_list.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON output"
    )

    p_pull = sub.add_parser("pull", help="Pull a model from Ollama", parents=[parent_parser])
    p_pull.add_argument("model", help="Model name, e.g. llama3.2")

    p_del = sub.add_parser("delete", help="Delete a model", parents=[parent_parser])
    p_del.add_argument("model", help="Model name to delete")

    p_show = sub.add_parser("show", help="View a model card", parents=[parent_parser])
    p_show.add_argument("model", help="Model name")

    p_ram = sub.add_parser("ram", help="Estimate RAM needed to run a model", parents=[parent_parser])
    p_ram.add_argument("model", help="Model name")
    p_ram.add_argument(
        "--context-len",
        type=int,
        default=4096,
        help="Context window length for the KV-cache estimate (default: 4096)",
    )

    p_cmp = sub.add_parser("compare", help="Compare two models side by side", parents=[parent_parser])
    p_cmp.add_argument("model1")
    p_cmp.add_argument("model2")

    sub.add_parser("version", help="Print the version and exit", parents=[parent_parser])

    return parser


def _format_model(m) -> str:
    return (
        f"  {m.name:<32} {m.size_gb:>6.2f} GB  {m.parameter_size or '-':<14} "
        f"{m.quantization_level or '-':<12} {m.short_digest}"
    )


def _print_list(models, as_json: bool) -> int:
    if as_json:
        import json

        print(
            json.dumps(
                [
                    {
                        "name": m.name,
                        "size": m.size,
                        "size_gb": round(m.size_gb, 3),
                        "digest": m.digest,
                        "parameter_size": m.parameter_size,
                        "quantization_level": m.quantization_level,
                        "modified_at": m.modified_at,
                    }
                    for m in models
                ],
                indent=2,
            )
        )
    else:
        if not models:
            print("No models found on the Ollama server.")
            return 0
        print(f"{'Model':<32} {'Size':>8} {'Parameters':<14} {'Quant':<12} {'Digest'}")
        print(f"{'-' * 32} {'-' * 8} {'-' * 14} {'-' * 12} {'-' * 12}")
        for m in models:
            print(_format_model(m))
        print(f"\n{len(models)} model(s) found.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(f"ollama-mm {__version__}")
        return 0

    client = OllamaClient(base_url=args.url)

    try:
        if args.command == "list":
            return _print_list(client.list_models(), args.json)

        if args.command == "pull":
            print(f"Pulling {args.model} ...")

            def on_progress(line: dict) -> None:
                status = line.get("status", "")
                if line.get("total") and line.get("completed"):
                    total = line["total"]
                    completed = line["completed"]
                    pct = completed / total * 100 if total else 0
                    print(f"  {status}: {completed}/{total} ({pct:.0f}%)", end="\r")
                elif status:
                    print(f"  {status}")

            client.pull_model(args.model, on_progress=on_progress)
            print(f"\nPull complete: {args.model}")
            return 0

        if args.command == "delete":
            ok = client.delete_model(args.model)
            if ok:
                print(f"Deleted: {args.model}")
                return 0
            print(f"Not found (nothing deleted): {args.model}")
            return 0

        if args.command == "show":
            detail = client.show_model(args.model)
            print(render_card(detail))
            return 0

        if args.command == "ram":
            detail = client.show_model(args.model)
            estimate = estimate_ram(detail, context_length=args.context_len)
            print(estimate.formatted())
            return 0

        if args.command == "compare":
            a = client.show_model(args.model1)
            b = client.show_model(args.model2)
            print(compare_models(a, b))
            return 0

    except ModelNotFoundError as exc:
        print(f"Error: {exc}")
        print("Tip: run 'ollama-mm list' to see available models.")
        return 1
    except OllamaError as exc:
        print(f"Error: {exc}")
        print("Is Ollama running? Start it with: ollama serve")
        return 1

    parser.print_help(sys.stderr)
    return 2