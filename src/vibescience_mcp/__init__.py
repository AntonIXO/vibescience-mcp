"""vibescience-mcp — a scientific experiment log (MCP server).

Markdown vault is the source of truth; SQLite is a disposable index.
"""

from .core import Store, VibeScienceError
from .models import (
    Diagnostic,
    Experiment,
    Hypothesis,
    Intervention,
    Paper,
    Problem,
    Verdict,
)

__all__ = [
    "Store",
    "VibeScienceError",
    "Diagnostic",
    "Experiment",
    "Hypothesis",
    "Intervention",
    "Paper",
    "Problem",
    "Verdict",
]
__version__ = "0.1.0"
