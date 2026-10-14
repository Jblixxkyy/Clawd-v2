"""Brief 019: verification loops for the agent harness.

Two mechanisms, both fail-open (a verification/corrector problem must never
break the agent turn):

1. Post-edit reflection (aider ``run_one`` / ``reflected_message`` pattern):
   after a turn whose tool batch included successful file edits, run the
   configured ``lint_cmd`` / ``test_cmd``. On failure the raw failure output
   becomes the next user message, capped at :data:`VERIFY_MAX_REFLECTIONS`
   failed reflections per run — then the failure is surfaced to the user
   instead of looping.

2. Edit-corrector (gemini-cli ``editCorrector.ts`` pattern): when an Edit
   tool call fails on mechanical grounds (bad whitespace/anchor/escaping),
   try a cheap secondary model with a strict "fix only the mechanics, do not
   invent a new edit" prompt (LRU-cached), fall back to deterministic regex
   repair, and only then surface the failure to the main model. The
   corrector may only change the *anchor* (``old_string``); ``new_string``
   and every other field are byte-identical, and a repaired anchor must
   occur verbatim in the file — so it can never change edit semantics.

3. Approval-aware policy (codex "Validating your work" pattern):
   non-interactive/background runs verify proactively; interactive runs
   propose-first ("tests failed — fix and re-run?"); explicit test tasks
   (the user's prompt asks for tests) bypass the hold.

Configuration: the ``"verify"`` section of ``~/.clawd/config.json``::

    "verify": {
      "enabled": true,
      "lint_cmd": "ruff check .",
      "test_cmd": "pytest -q",
      "timeout": 300
    }

``lint_cmd``/``test_cmd`` default to repo-detected commands (eslint/ruff /
pytest / npm test markers); an explicit ``null`` disables that stage.
``tool_context.verify_config`` overrides the file section outright (used by
tests and programmatic runs).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

#: Hard cap on failed reflections per run (aider's ``max_reflections = 3``).
VERIFY_MAX_REFLECTIONS = 3

_VERIFY_TIMEOUT_DEFAULT = 300
_VERIFY_OUTPUT_LIMIT = 8_000
_CORRECTOR_CACHE_SIZE = 64


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _detect_repo_commands(root: Path) -> dict[str, Optional[str]]:
    """Repo-detected lint/test defaults. Never raises."""
    lint_cmd: Optional[str] = None
    test_cmd: Optional[str] = None
    try:
        pkg = root / "package.json"
        if pkg.is_file():
            try:
                scripts = json.loads(pkg.read_text(encoding="utf-8")).get("scripts", {})
            except Exception:
                scripts = {}
            if isinstance(scripts, dict) and scripts.get("test"):
                test_cmd = "npm test"
            eslint_configs = (
                "eslint.config.js",
                "eslint.config.mjs",
                "eslint.config.cjs",
                ".eslintrc",
                ".eslintrc.js",
                ".eslintrc.cjs",
                ".eslintrc.json",
                ".eslintrc.yaml",
                ".eslintrc.yml",
            )
            if any((root / name).exists() for name in eslint_configs):
                lint_cmd = "npx eslint ."
        py_markers = ("pyproject.toml", "setup.py", "setup.cfg", "pytest.ini", "tox.ini")
        if any((root / name).is_file() for name in py_markers):
            test_cmd = test_cmd or "python -m pytest -q"
            ruff_configured = (root / "ruff.toml").is_file() or (root / ".ruff.toml").is_file()
            if not ruff_configured:
                pyproject = root / "pyproject.toml"
                if pyproject.is_file():
                    try:
                        ruff_configured = "[tool.ruff]" in pyproject.read_text(encoding="utf-8")
                    except Exception:
                        ruff_configured = False
            if ruff_configured:
                try:
                    import importlib.util

                    has_ruff = importlib.util.find_spec("ruff") is not None
                except Exception:
                    has_ruff = False
                if has_ruff or shutil.which("ruff"):
                    lint_cmd = lint_cmd or "ruff check ."
    except Exception:
        pass
    return {"lint_cmd": lint_cmd, "test_cmd": test_cmd}


def _read_verify_section_no_create() -> dict[str, Any]:
    """Read the user's ``"verify"`` config section without side effects.

    ``load_config()`` *creates* ``~/.clawd/config.json`` (mkdir + write)
    when it is missing — a visible filesystem mutation the agent loop must
    never perform just to check whether verification is configured. This
    reads the file directly and returns {} when absent/unreadable. Never
    raises.
    """
    try:
        path = Path.home() / ".clawd" / "config.json"
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        section = (data or {}).get("verify", {}) or {}
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def resolve_verify_config(
    workspace_root: Any,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Merge user config over repo-detected lint/test defaults.

    Args:
        workspace_root: repo root used for marker detection.
        config: full config dict (``"verify"`` section is read); when None
            the live ``~/.clawd/config.json`` is read directly (never
            created — see :func:`_read_verify_section_no_create`). Never
            raises.

    Returns ``{"enabled", "lint_cmd", "test_cmd", "timeout"}``.
    """
    root = Path(workspace_root) if workspace_root else Path(".")
    detected = _detect_repo_commands(root)
    section: dict[str, Any] = {}
    try:
        if config is None:
            section = _read_verify_section_no_create()
        else:
            section = (config or {}).get("verify", {}) or {}
            if not isinstance(section, dict):
                section = {}
    except Exception:
        section = {}
    if "lint_cmd" in section:
        lint_cmd = section["lint_cmd"] or None
    else:
        lint_cmd = detected["lint_cmd"]
    if "test_cmd" in section:
        test_cmd = section["test_cmd"] or None
    else:
        test_cmd = detected["test_cmd"]
    try:
        timeout = int(section.get("timeout", _VERIFY_TIMEOUT_DEFAULT))
    except Exception:
        timeout = _VERIFY_TIMEOUT_DEFAULT
    return {
        "enabled": bool(section.get("enabled", True)),
        "lint_cmd": lint_cmd,
        "test_cmd": test_cmd,
        "timeout": timeout,
    }


# ---------------------------------------------------------------------------
# Command runner
# ---------------------------------------------------------------------------


def run_verify_command(
    cmd: str,
    cwd: Any,
    timeout: int = _VERIFY_TIMEOUT_DEFAULT,
) -> tuple[str, str]:
    """Run a verify command.

    Returns ``(status, output)`` with status in ``{"pass", "fail", "skip"}``.
    ``"skip"`` means the runner itself broke (not a code failure) — the
    caller should log it and move on, never reflect it. Never raises.
    """
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as te:
        out = str(getattr(te, "stdout", "") or "") + str(getattr(te, "stderr", "") or "")
        return "fail", f"[timed out after {timeout}s]\n{out[-4000:]}".strip()
    except Exception as exc:  # fail-open: a broken runner is not a code failure
        return "skip", f"[verify runner error: {exc}]"
    output = (proc.stdout or "") + (proc.stderr or "")
    output = output[-_VERIFY_OUTPUT_LIMIT:].strip() or "(no output)"
    return ("pass" if proc.returncode == 0 else "fail"), output


# ---------------------------------------------------------------------------
# Explicit test-task detection (codex "test tasks bypass the hold")
# ---------------------------------------------------------------------------


_TEST_TASK_RE = re.compile(
    r"(pytest|npm test|make test|run the tests?|tests? (must|should|need to|have to) pass"
    r"|make sure .*tests? pass|test suite|\blint\b)",
    re.IGNORECASE,
)


def is_explicit_test_task(conversation: Any) -> bool:
    """True when the user's task prompt explicitly asks for tests/lint."""
    try:
        messages = getattr(conversation, "messages", None) or []
        for msg in messages:
            if getattr(msg, "role", None) == "user":
                content = getattr(msg, "content", "")
                # Only the first user message is the task prompt; later ones
                # are harness-injected notices.
                return isinstance(content, str) and bool(_TEST_TASK_RE.search(content))
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Edit corrector
# ---------------------------------------------------------------------------


_MECHANICAL_RE = re.compile(r"old_string not found", re.IGNORECASE)
_NON_MECHANICAL_RE = re.compile(
    r"file does not exist|must be read first|permission|expected_occurrences"
    r"|ambiguous|multiple occurrences|must be strings|must be a non-empty",
    re.IGNORECASE,
)


def is_mechanical_edit_error(text: Any) -> bool:
    """True when an Edit failure looks like a whitespace/anchor miss.

    Only the 0-occurrence "old_string not found" case qualifies: ambiguity,
    occurrence-count mismatches, missing files, and permission problems are
    semantic or environmental, never mechanical.
    """
    if not isinstance(text, str) or not text:
        return False
    if _NON_MECHANICAL_RE.search(text):
        return False
    return bool(_MECHANICAL_RE.search(text))


def _strip_all_ws(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _align_trailing_ws(old: str, anchor: str, content: str) -> str:
    """Extend a repaired anchor over the file's immediately following
    whitespace when the intended anchor ended with whitespace.

    Without this, repairing ``"…1\\n"`` -> ``"…1"`` while ``new_string``
    ends with ``"\\n"`` would duplicate the line break. Only extends over
    the exact trailing run the model intended. Never raises.
    """
    try:
        trailing = old[len(old.rstrip()):]
        if not trailing or anchor.endswith(trailing):
            return anchor
        idx = content.find(anchor)
        if idx < 0:
            return anchor
        if content.startswith(trailing, idx + len(anchor)):
            return anchor + trailing
        return anchor
    except Exception:
        return anchor


def regex_repair_anchor(old: str, content: str) -> Optional[str]:
    """Deterministic whitespace-flexible anchor repair.

    Returns the actual file text to use as the repaired ``old_string``, or
    None when no safe repair exists. A repair is only safe when the
    whitespace-flexible match is unambiguous (exactly one occurrence in the
    file) — otherwise the failure is surfaced to the model. Never raises.
    """
    try:
        if not isinstance(old, str) or not old or old in content:
            return None
        stripped = old.strip()
        if stripped and stripped in content and content.count(stripped) == 1:
            return _align_trailing_ws(old, stripped, content)
        tokens = [t for t in re.split(r"\s+", old.strip()) if t]
        if not tokens:
            return None
        pattern = r"\s+".join(re.escape(t) for t in tokens)
        matches = list(re.finditer(pattern, content))
        if len(matches) == 1:
            return _align_trailing_ws(old, matches[0].group(0), content)
        return None
    except Exception:
        return None


_CORRECTOR_SYSTEM = (
    "You repair failed exact-string file edits. The edit failed because "
    "old_string did not match the file — almost always a whitespace, "
    "indentation, or escaping mismatch.\n"
    "Fix ONLY whitespace/indentation/escaping mechanics so the anchor "
    "matches the file. Do NOT invent a new edit. Do NOT change what the "
    "edit does. The corrected anchor MUST appear verbatim in the file "
    "content below. If you cannot fix it mechanically, reply "
    '{"old_string": null}.'
)


def _parse_corrector_json(text: Any) -> Any:
    try:
        if not isinstance(text, str):
            return None
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        if isinstance(data, dict):
            return data.get("old_string")
    except Exception:
        pass
    try:
        m = re.search(r'"old_string"\s*:\s*(null|"(?:[^"\\]|\\.)*")', text or "")
        if m:
            return json.loads(m.group(1))
    except Exception:
        pass
    return None


def _anchor_is_acceptable(original_old: str, candidate: Any, content: str) -> bool:
    """Semantics guard for a cheap-model repaired anchor. Never raises.

    The candidate must appear verbatim in the file AND differ from the
    intended anchor solely in whitespace. A retokenized identifier or a
    changed literal (``return 2`` -> ``return 1``) is a semantic change, not
    a mechanical fix, and is refused — the failure surfaces to the main
    model instead.
    """
    try:
        if not isinstance(candidate, str) or not candidate or candidate == original_old:
            return False
        if candidate not in content:
            return False
        return _strip_all_ws(original_old) == _strip_all_ws(candidate)
    except Exception:
        return False


def _cheap_model_name(provider: Any) -> tuple[Optional[str], bool]:
    """(model, is_dedicated_small_tier). Never raises."""
    try:
        from ..model_tiers import resolve_models, utility_model

        current = getattr(provider, "model", None)
        cheap = utility_model(current if isinstance(current, str) else None)
        models = resolve_models(None)
        small = models.get("small") or ""
        is_small = bool(cheap) and bool(small) and cheap == small and cheap != current
        return (cheap if isinstance(cheap, str) and cheap else None), is_small
    except Exception:
        current = getattr(provider, "model", None)
        return (current if isinstance(current, str) else None), False


class EditCorrector:
    """Cheap-model + regex repair for mechanical Edit failures.

    ``correct()`` returns ``(corrected_input, method)`` or None.
    ``method`` is ``"cache"`` / ``"cheap_model"`` / ``"regex"``. The returned
    input is the original input with only ``old_string`` values replaced —
    ``new_string`` and every other field are byte-identical, so semantics
    cannot change. Never raises.
    """

    def __init__(self, cache_size: int = _CORRECTOR_CACHE_SIZE) -> None:
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_size = max(1, int(cache_size))

    @staticmethod
    def _cache_key(file_path: str, old_string: str, error_sig: str) -> str:
        blob = json.dumps(
            {"f": file_path, "o": old_string, "e": error_sig},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str, content: str) -> Optional[str]:
        try:
            anchor = self._cache.get(key)
            if anchor is None:
                return None
            # LRU touch.
            self._cache.move_to_end(key)
            # The file may have changed since the correction was cached.
            if anchor in content:
                return anchor
            self._cache.pop(key, None)
            return None
        except Exception:
            return None

    def _cache_put(self, key: str, anchor: str) -> None:
        try:
            self._cache[key] = anchor
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        except Exception:
            pass

    def _read_file(self, tool_input: dict, context: Any) -> Optional[tuple[str, str]]:
        """(file_path_str, content) or None. Never raises."""
        try:
            file_path = tool_input.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                return None
            path = Path(file_path)
            if not path.is_absolute():
                root = getattr(context, "workspace_root", None) or Path(".")
                path = Path(root) / path
            try:
                allowed = context.ensure_allowed_path(str(path))
            except Exception:
                allowed = path
            content = allowed.read_text(encoding="utf-8", errors="replace")
            return str(allowed), content
        except Exception:
            return None

    def _correct_anchor(
        self,
        old_string: str,
        file_path: str,
        content: str,
        error_text: str,
        provider: Any,
    ) -> tuple[Optional[str], str]:
        """(repaired_anchor, method) — method "" when unrepaired."""
        key = self._cache_key(file_path, old_string, error_text[:200])
        cached = self._cache_get(key, content)
        if cached is not None:
            return cached, "cache"
        # Stage 1: cheap secondary model with the strict mechanics-only
        # prompt (brief 028's small-model tier when configured, otherwise
        # the current model with the same strict prompt).
        if provider is not None:
            try:
                cheap_model, _is_small = _cheap_model_name(provider)
                excerpt = content if len(content) <= 12_000 else content[:12_000] + "\n[…truncated…]"
                response = provider.chat(
                    [
                        {"role": "system", "content": _CORRECTOR_SYSTEM},
                        {
                            "role": "user",
                            "content": (
                                f"File content:\n{excerpt}\n\n"
                                f"Failed old_string:\n{old_string}\n\n"
                                f"Error:\n{error_text[:1000]}\n\n"
                                'Reply with ONLY the JSON object {"old_string": "..."} '
                                "or {\"old_string\": null}."
                            ),
                        },
                    ],
                    model=cheap_model,
                )
                candidate = _parse_corrector_json(getattr(response, "content", ""))
                if _anchor_is_acceptable(old_string, candidate, content):
                    assert isinstance(candidate, str)
                    candidate = _align_trailing_ws(old_string, candidate, content)
                    self._cache_put(key, candidate)
                    return candidate, "cheap_model"
            except Exception:
                pass
        # Stage 2: deterministic regex fallback.
        repaired = regex_repair_anchor(old_string, content)
        if repaired is not None:
            self._cache_put(key, repaired)
            return repaired, "regex"
        return None, ""

    def correct(
        self,
        tool_input: Any,
        error_text: Any,
        context: Any,
        provider: Any,
    ) -> Optional[tuple[dict[str, Any], str]]:
        """Attempt to repair a failed Edit input.

        Returns ``(corrected_input, method)`` or None when the failure is
        not mechanical, no safe repair exists, or anything goes wrong.
        Never raises.
        """
        try:
            if not is_mechanical_edit_error(error_text):
                return None
            if not isinstance(tool_input, dict):
                return None
            read = self._read_file(tool_input, context)
            if read is None:
                return None
            file_path, content = read

            # Collect the anchors to repair: single-edit or multi-hunk.
            hunks = tool_input.get("edits")
            if hunks is not None:
                if not isinstance(hunks, list) or not hunks:
                    return None
                indexed = [
                    (i, h.get("old_string"))
                    for i, h in enumerate(hunks)
                    if isinstance(h, dict) and isinstance(h.get("old_string"), str)
                ]
            else:
                old = tool_input.get("old_string")
                if not isinstance(old, str):
                    return None
                indexed = [(-1, old)]

            methods: list[str] = []
            repaired_any = False
            new_hunks = None
            if hunks is not None:
                new_hunks = [dict(h) if isinstance(h, dict) else h for h in hunks]
            single_repair = None
            for i, old_string in indexed:
                anchor, method = self._correct_anchor(
                    old_string, file_path, content, str(error_text), provider
                )
                if anchor is None:
                    # One unrepairable anchor vetoes the whole repair: a
                    # partial fix would change which edit the model asked for.
                    return None
                repaired_any = True
                methods.append(method)
                if i == -1:
                    single_repair = anchor
                else:
                    assert new_hunks is not None
                    new_hunks[i]["old_string"] = anchor

            if not repaired_any:
                return None
            corrected = dict(tool_input)
            if new_hunks is not None:
                corrected["edits"] = new_hunks
            else:
                corrected["old_string"] = single_repair
            # Overall method: the strongest stage actually used.
            method = "cheap_model" if "cheap_model" in methods else methods[0]
            return corrected, method
        except Exception:
            return None
