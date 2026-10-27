"""Allow running the package as: python -m ollama_model_manager"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
