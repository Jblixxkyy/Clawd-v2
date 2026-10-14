"""JSONL trace recording for eval runs.

One JSON object per line, written incrementally so a crashed run still has a
partial trace. Line types:

- run:         run header (task, prompt, provider, model, timeout, policy)
- turn:        one provider call = one agent turn (tokens, duration)
- tool_use:    model requested a tool call (name, args summary, turn)
- tool_result: tool finished (summary, is_error, turn, duration)
- tool_error:  dispatch raised instead of returning a result
- result:      final outcome (response text, totals, wall time, timed_out)

Large payloads (full file contents, long outputs) are truncated with a
[truncated ...] marker; the recorder keeps summaries, not megabytes.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..tool_system.agent_loop import ToolEvent, summarize_tool_result, summarize_tool_use

_MAX_ARGS_CHARS = 2_000
_MAX_OUTPUT_CHARS = 4_000
_MAX_TEXT_CHARS = 4_000


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _truncate(value: Any, limit: int) -> Any:
    """Render value as compact JSON (or str) and cut it to `limit` chars."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    if len(text) > limit:
        return text[:limit] + f"...[truncated, {len(text) - limit} more chars]"
    return text


def _safe_json(obj: Any) -> Any:
    """JSON-round-trip so the trace file never contains non-serializable junk."""
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False, default=str))
    except Exception:
        return str(obj)


class TraceRecorder:
    """Incrementally appends trace events to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._turn = 0
        self._turn_t0: float | None = None
        self._started = False

    # -- lifecycle ---------------------------------------------------------

    def start_run(
        self,
        *,
        run_id: str,
        task_id: str,
        prompt: str,
        provider_name: str,
        model: str | None,
        timeout_s: float,
        max_turns: int,
        permission_policy: str = "auto-deny-on-ask (print-mode semantics)",
    ) -> None:
        self._write({
            "type": "run",
            "run_id": run_id,
            "task_id": task_id,
            "prompt": _truncate(prompt, _MAX_TEXT_CHARS),
            "provider": provider_name,
            "model": model,
            "timeout_s": timeout_s,
            "max_turns": max_turns,
            "permission_policy": permission_policy,
            "no_proxy_sanitized": True,
            "started_at": _utcnow(),
        })
        self._started = True

    def begin_turn(self) -> int:
        """Called immediately before each provider call; returns the turn number."""
        self._turn += 1
        self._turn_t0 = time.monotonic()
        return self._turn

    def end_turn(self, usage: Any) -> None:
        """Called immediately after each provider call with its usage dict."""
        duration = (time.monotonic() - self._turn_t0) if self._turn_t0 else 0.0
        self._write({
            "type": "turn",
            "turn": self._turn,
            "usage": _safe_json(usage or {}),
            "duration_s": round(duration, 3),
            "at": _utcnow(),
        })

    # -- tool events (agent_loop on_event callback) ------------------------

    def on_event(self, event: ToolEvent) -> None:
        base = {
            "at": _utcnow(),
            "turn": self._turn,
            "tool_use_id": event.tool_use_id,
            "tool_name": event.tool_name,
        }
        if event.kind == "tool_use":
            self._write({
                **base,
                "type": "tool_use",
                "input_summary": summarize_tool_use(event.tool_name, event.tool_input or {}),
                "input": _truncate(event.tool_input, _MAX_ARGS_CHARS),
            })
        elif event.kind == "tool_result":
            self._write({
                **base,
                "type": "tool_result",
                "is_error": event.is_error,
                "result_summary": summarize_tool_result(event.tool_name, event.tool_output),
                "output": _truncate(event.tool_output, _MAX_OUTPUT_CHARS),
            })
        elif event.kind == "tool_error":
            self._write({
                **base,
                "type": "tool_error",
                "is_error": True,
                "error": _truncate(event.error, _MAX_OUTPUT_CHARS),
            })

    # -- finish ------------------------------------------------------------

    def finish(
        self,
        *,
        response_text: str | None,
        usage: Any,
        num_turns: int,
        wall_time_s: float,
        timed_out: bool,
        cancelled: bool = False,
        error: str | None = None,
    ) -> None:
        self._write({
            "type": "result",
            "response_text": _truncate(response_text or "", _MAX_TEXT_CHARS),
            "usage": _safe_json(usage or {}),
            "num_turns": num_turns,
            "wall_time_s": round(wall_time_s, 3),
            "timed_out": timed_out,
            "cancelled": cancelled,
            "error": error,
            "finished_at": _utcnow(),
        })
        self.close()

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def _write(self, obj: dict[str, Any]) -> None:
        try:
            self._fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        except Exception:
            # Trace recording must never break a run.
            pass


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Read a trace.jsonl back into a list of event dicts (for analysis/tests)."""
    events: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events
