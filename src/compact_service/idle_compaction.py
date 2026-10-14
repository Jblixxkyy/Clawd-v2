"""Idle auto-compaction (brief 026 -- module side only).

Claude Code 2.1.299 parity ("idleCompaction"): a conversation that exceeds
a token threshold and then sits idle for a configured duration is compacted
automatically in the background -- no user action, no blocking freeze. The
insight is behavioral, not technical: users don't compact proactively, they
hit the context wall mid-thought. Compacting during idle time means the
next interaction starts fresh instead of starting with a multi-minute
freeze (or a wall).

The mechanism is scheduling around the existing background compaction
(brief 022), not a model feature -- deliberately loop-agnostic, like
async_compaction: the loop-wiring owns the checkpoints and supplies the
LLM-backed ``work_fn``; this module only implements the trigger
discipline:

- :class:`IdleCompactor`: per-session last-activity tracking,
  ``should_trigger()`` (idle window + token threshold + background-work
  guard + per-idle-window once-only), ``mark_triggered()`` /
  ``record_compaction()`` bookkeeping, and the quiet-notice queue,
- :func:`any_background_work_running`: read-only probe of the Bash
  background-job and subagent registries (fail-open),
- :func:`read_idle_compaction_config`: read-only config probe -- never
  ``load_config()`` from per-run code paths (mkdir/write side effect).

Intended call order (for the loop-wiring worker)::

    compactor = get_idle_compactor(session_id)   # per-session, cached
    compactor.configure(read_idle_compaction_config(override))

    # at run start (before any new work):
    last = compactor.last_activity      # gap since the PREVIOUS run
    compactor.note_activity()           # this run's arrival
    ...
    if compactor.should_trigger(now, last, token_count,
                                any_background_work_running()):
        task = manager.start_background_compaction(snapshot,
                                                   work_fn=idle_work_fn)
        compactor.mark_triggered(before_tokens=token_count)
        # ... the 022 checkpoints adopt; on adoption of the idle result:
        notice = compactor.record_compaction(before, after)
        # ... the notice is delivered via take_pending_notice() at the
        # start of the next run.

    # at run end:
    compactor.note_activity()           # model/tool completion

This module is fully unit-testable standalone: inject ``config`` and a
fake ``clock``; no LLM, no loop.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "DEFAULT_SESSION_KEY",
    "IDLE_COMPACTION_DEFAULTS_FALLBACK",
    "IdleCompactor",
    "any_background_work_running",
    "read_idle_compaction_config",
    "get_idle_compactor",
    "reset_idle_compactors",
]

#: Registry key used when the run has no session id.
DEFAULT_SESSION_KEY = "default"

#: Fallback defaults when src/config.py can't be imported (never expected;
#: kept so this module stays importable standalone).
IDLE_COMPACTION_DEFAULTS_FALLBACK: dict[str, Any] = {
    "enabled": False,
    "min_tokens": 200_000,
    "idle_minutes": 54,
}


def _idle_defaults() -> dict[str, Any]:
    """The ``idle_compaction`` defaults, without config-file side effects."""
    try:
        from ..config import _IDLE_COMPACTION_DEFAULTS

        return dict(_IDLE_COMPACTION_DEFAULTS)
    except Exception:
        return dict(IDLE_COMPACTION_DEFAULTS_FALLBACK)


class IdleCompactor:
    """Per-session idle-compaction trigger discipline (brief 026).

    Tracks last user/model activity, decides whether the idle window has
    elapsed with a big-enough context, enforces "at most once per idle
    window", and queues the quiet adoption notice for the next
    interaction. All public methods are fail-open: they never raise.
    """

    def __init__(
        self,
        *,
        config: Optional[dict] = None,
        session_id: Any = None,
        clock: Any = None,
    ) -> None:
        self._config: dict[str, Any] = dict(config or {})
        self.session_id = session_id
        self._clock = clock or time.time
        self._lock = threading.Lock()
        # Last user/model activity (epoch seconds). Updated on user-message
        # arrival and on run completion; read (before updating) at run
        # start to measure the idle gap since the previous run.
        self.last_activity: Optional[float] = None
        # Trigger bookkeeping for the per-idle-window once-only rule.
        self._last_trigger_at: Optional[float] = None
        self._last_trigger_before_tokens: Optional[int] = None
        # Quiet notice queued for the next interaction.
        self._pending_notice: Optional[str] = None
        # Last completed idle compaction's before/after counts.
        self.last_compaction: Optional[dict[str, Any]] = None

    # -- config ---------------------------------------------------------

    def configure(self, config: Optional[dict]) -> None:
        """Refresh config (the loop re-probes each run). Never raises."""
        try:
            self._config = dict(config or {})
        except Exception:
            pass

    def is_enabled(self) -> bool:
        """Master switch (default OFF)."""
        try:
            return bool(self._config.get("enabled", False))
        except Exception:
            return False

    def min_tokens(self) -> int:
        """Context-size threshold for an idle compaction."""
        try:
            return int(self._config.get("min_tokens", 200_000) or 0)
        except Exception:
            return 200_000

    def idle_minutes(self) -> float:
        """Idle duration before triggering."""
        try:
            return float(self._config.get("idle_minutes", 54) or 0)
        except Exception:
            return 54.0

    def idle_seconds(self) -> float:
        try:
            return max(0.0, self.idle_minutes() * 60.0)
        except Exception:
            return 0.0

    # -- activity -------------------------------------------------------

    def note_activity(self, ts: Optional[float] = None) -> None:
        """Record user/model activity (arrival or completion). Never raises."""
        try:
            stamp = float(ts) if ts is not None else float(self._clock())
        except Exception:
            return
        try:
            with self._lock:
                self.last_activity = stamp
        except Exception:
            pass

    @property
    def last_trigger_before_tokens(self) -> Optional[int]:
        return self._last_trigger_before_tokens

    # -- trigger discipline ----------------------------------------------

    def should_trigger(
        self,
        now: float,
        last_activity: Optional[float],
        token_count: int,
        background_busy: bool,
    ) -> bool:
        """True when an idle background compaction should launch.

        All four gates must pass: the feature is enabled, no background
        work (Bash jobs / subagents) is running, the session has been idle
        for at least ``idle_minutes`` (measured from ``last_activity``,
        the gap since the previous run), the context holds at least
        ``min_tokens``, and no compaction was already triggered for this
        idle window. Fail-open: any internal problem returns False -- the
        check must never block a turn.
        """
        try:
            if not self.is_enabled():
                return False
            if background_busy:
                return False
            if last_activity is None:
                return False
            idle_seconds = self.idle_seconds()
            if idle_seconds <= 0:
                return False
            try:
                idle_for = float(now) - float(last_activity)
            except Exception:
                return False
            if idle_for < idle_seconds:
                return False
            try:
                tokens = int(token_count or 0)
            except Exception:
                return False
            if tokens < self.min_tokens():
                return False
            # Per-idle-window once-only: a trigger is consumed until NEW
            # activity arrives after it (the idle window ends on activity).
            with self._lock:
                last_trigger = self._last_trigger_at
            if last_trigger is not None:
                try:
                    if float(last_activity) <= float(last_trigger):
                        return False
                except Exception:
                    return False
            return True
        except Exception:
            return False

    def mark_triggered(self, before_tokens: Optional[int] = None) -> None:
        """Record that an idle compaction was launched. Never raises."""
        try:
            try:
                before = (
                    int(before_tokens) if before_tokens is not None else None
                )
            except Exception:
                before = None
            with self._lock:
                self._last_trigger_at = float(self._clock())
                self._last_trigger_before_tokens = before
        except Exception:
            pass

    def record_compaction(self, before_tokens: Any, after_tokens: Any) -> str:
        """Record a completed idle compaction; queue the quiet notice.

        Returns the notice text (also stashed for ``take_pending_notice``).
        Never raises.
        """
        try:
            before = int(before_tokens or 0)
        except Exception:
            before = 0
        try:
            after = int(after_tokens or 0)
        except Exception:
            after = 0
        notice = f"[Harness] compacted while idle ({before}\u2192{after} tokens)"
        try:
            with self._lock:
                try:
                    stamp = float(self._clock())
                except Exception:
                    stamp = 0.0
                self.last_compaction = {
                    "before_tokens": before,
                    "after_tokens": after,
                    "at": stamp,
                }
                self._pending_notice = notice
        except Exception:
            pass
        return notice

    def take_pending_notice(self) -> Optional[str]:
        """Take the queued quiet notice (once). Never raises."""
        try:
            with self._lock:
                notice = self._pending_notice
                self._pending_notice = None
                return notice
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Background-work probe (guardrail: never idle-compact while background
# work is running).
# ---------------------------------------------------------------------------


def any_background_work_running() -> bool:
    """True when a Bash background job or background subagent is running.

    Read-only probe of the tool registries in
    ``src/tool_system/tools/bash.py`` (``_BG_JOBS``) and
    ``src/tool_system/tools/subagent.py`` (``_BG_SUBAGENTS``). Fail-open:
    returns False on any error (a broken probe must not wedge the loop).
    """
    # Bash background jobs: job["process"] is a Popen; poll() is None while
    # the process is still alive.
    try:
        from ..tool_system.tools.bash import _BG_JOBS, _BG_LOCK

        with _BG_LOCK:
            jobs = list(_BG_JOBS.values())
        for job in jobs:
            try:
                proc = (job or {}).get("process")
                if proc is not None and proc.poll() is None:
                    return True
            except Exception:
                continue
    except Exception:
        pass
    # Background subagents: records start with status "running" and flip to
    # done/failed/stalled when the child loop finishes (done_event set).
    try:
        from ..tool_system.tools.subagent import _BG_SUBAGENTS
        from ..tool_system.tools import subagent as _subagent_mod

        lock = getattr(_subagent_mod, "_BG_LOCK", None)
        if lock is not None:
            with lock:
                records = list(_BG_SUBAGENTS.values())
        else:
            records = list(_BG_SUBAGENTS.values())
        for record in records:
            try:
                rec = record or {}
                if rec.get("status") != "running":
                    continue
                done_event = rec.get("done_event")
                if done_event is not None:
                    try:
                        if done_event.is_set():
                            continue
                    except Exception:
                        pass
                return True
            except Exception:
                continue
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Config probe (read-only: never load_config() from per-run code paths).
# ---------------------------------------------------------------------------


def read_idle_compaction_config(
    context_override: Optional[dict] = None,
) -> dict[str, Any]:
    """Read the ``idle_compaction`` config without side effects.

    Precedence: explicit override (tool_context seam / tests) > the
    ``idle_compaction`` section of ~/.clawd/config.json (read ONLY when the
    file already exists) > defaults. ``load_config()``/``get_config_path()``
    both mkdir ~/.clawd as a side effect, which must never happen as a side
    effect of running the loop -- so the file is probed with ``is_file()``
    first, exactly like the 022 wiring. Never raises.
    """
    try:
        if isinstance(context_override, dict) and context_override:
            merged = _idle_defaults()
            merged.update(context_override)
            return merged
        cfg_path = Path.home() / ".clawd" / "config.json"
        if cfg_path.is_file():
            from ..config import idle_compaction_config

            loaded = idle_compaction_config() or {}
            if isinstance(loaded, dict) and loaded:
                return dict(loaded)
    except Exception:
        pass
    return _idle_defaults()


# ---------------------------------------------------------------------------
# Per-session registry: the compactor outlives one loop run so last-activity
# and trigger-window state persist across runs in the same process.
# ---------------------------------------------------------------------------

_compactors: dict[str, IdleCompactor] = {}
_compactors_lock = threading.Lock()


def get_idle_compactor(
    session_id: Any = None, *, config: Optional[dict] = None
) -> IdleCompactor:
    """Return the cached :class:`IdleCompactor` for a session.

    ``session_id`` None (runs without one, e.g. the repl) share the
    ``"default"`` compactor. Never raises.
    """
    try:
        key = str(session_id) if session_id else DEFAULT_SESSION_KEY
    except Exception:
        key = DEFAULT_SESSION_KEY
    try:
        with _compactors_lock:
            compactor = _compactors.get(key)
            if compactor is None:
                compactor = IdleCompactor(config=config, session_id=session_id)
                _compactors[key] = compactor
            return compactor
    except Exception:
        return IdleCompactor(config=config, session_id=session_id)


def reset_idle_compactors() -> None:
    """Drop all cached compactors (test helper). Never raises."""
    try:
        with _compactors_lock:
            _compactors.clear()
    except Exception:
        pass
