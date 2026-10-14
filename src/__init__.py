"""Clawd Codex - Claude Code Python Implementation."""

import os
from pathlib import Path

__version__ = "0.2.0"
__author__ = "Clawd Codex Team"

# Set managed skills directory for built-in skills
_MANAGED_SKILLS_DIR = Path(__file__).parent.parent / "managed_skills"
if _MANAGED_SKILLS_DIR.exists():
    os.environ.setdefault("CLAWD_MANAGED_SKILLS_DIR", str(_MANAGED_SKILLS_DIR))

from .config import load_config, get_provider_config

try:  # pragma: no cover
    from .providers.base import BaseProvider
except Exception:  # pragma: no cover
    BaseProvider = None  # type: ignore[assignment]

__all__ = [
    "__version__",
    "__author__",
    "load_config",
    "get_provider_config",
    "BaseProvider",
]
