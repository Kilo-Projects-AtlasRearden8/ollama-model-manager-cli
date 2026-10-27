"""ollama_model_manager - local Ollama model management CLI.

List, pull, delete, compare Ollama models, view model cards, and estimate
RAM usage. Local-first, no secrets, beginner friendly.

Copyright (c) 2026 Asdale Dynamics, LLC
"""

__version__ = "0.1.0"

from .client import OllamaClient, OllamaError, ModelNotFoundError
from .models import ModelDetail, RamEstimate

__all__ = [
    "OllamaClient",
    "OllamaError",
    "ModelNotFoundError",
    "ModelDetail",
    "RamEstimate",
    "__version__",
]
