"""Module-level tests for brief 036 — environment capability card.

Covers the detection → card contract via an injectable detector hook
(no live subprocesses, no loop wiring). Loop wiring (turn-0 prompt
injection) is a later wave and is intentionally not tested here.
"""

from __future__ import annotations

import platform
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tool_system.env_card import build_env_card


def _ctx(root: str = "/tmp/wk") -> SimpleNamespace:
    return SimpleNamespace(workspace_root=Path(root))


def _win_no_bash() -> dict:
    return {"os_name": "Windows", "bash": None, "powershell": r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"}


def _linux_with_bash() -> dict:
    return {"os_name": "Linux", "bash": "/usr/bin/bash", "powershell": None}


def test_windows_no_bash_card_warns_and_names_powershell() -> None:
    card = build_env_card(_ctx(), detector=_win_no_bash)
    assert card, "expected a non-empty card"
    assert "Bash will fail" in card
    assert "PowerShell" in card
    assert "Select-String" in card
    assert "Grep" in card


def test_linux_with_bash_card_has_no_warning() -> None:
    card = build_env_card(_ctx(), detector=_linux_with_bash)
    assert card, "expected a non-empty card"
    assert "bash" in card.lower()
    assert "will fail" not in card.lower()
    assert "not installed" not in card.lower()


def test_detection_exception_returns_empty_string_no_raise() -> None:
    def boom() -> dict:
        raise RuntimeError("platform blew up")

    assert build_env_card(_ctx(), detector=boom) == ""


def test_broken_context_returns_empty_string_no_raise() -> None:
    class Exploding:
        @property
        def workspace_root(self):
            raise RuntimeError("no root")

    assert build_env_card(Exploding(), detector=_linux_with_bash) == ""


def test_card_content_quality() -> None:
    card = build_env_card(_ctx("/work"), detector=_win_no_bash)
    lines = card.splitlines()
    # Compact: roughly ~8 lines.
    assert 6 <= len(lines) <= 12, f"card has {len(lines)} lines:\n{card}"
    # Explicit DO/DON'T lines present.
    assert any(l.strip().startswith("- DO:") for l in lines), "missing DO line"
    assert any(l.strip().startswith("- DON'T:") for l in lines), "missing DON'T line"
    # Workspace root and cd-persistence note present.
    assert "Workspace root:" in card
    assert "/work" in card
    assert "cd" in card


def test_default_detector_is_monkeypatchable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live detection path must be interceptable via shutil.which/platform."""
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    def fake_which(name: str):
        return r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" if name == "powershell" else None

    monkeypatch.setattr(shutil, "which", fake_which)
    card = build_env_card(_ctx())
    assert "Bash will fail" in card
    assert card == "" or "OS: Windows" in card


def test_default_detector_live_never_raises() -> None:
    card = build_env_card(_ctx())
    assert isinstance(card, str)
