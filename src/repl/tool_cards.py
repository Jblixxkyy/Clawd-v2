"""Quiet tool-call rendering for the Clawd-Code terminal REPL.

ChatGPT-style quiet (brief 003): each tool call is ONE dim line
(``⠋ Read src/foo.py`` while running, ``✓ Read src/foo.py`` when done),
red-green diff lines only for Write/Edit where they matter, and red only on
errors. No bordered panels per call. Write/Edit completion lines carry a
compact ``(+N, -M)`` diff stat (gemini-cli DenseToolMessage style) so the
change size is visible at a glance without reading the diff.

``render_use`` / ``render_result`` / ``render_error`` return a *list* of
renderables (usually one line); the REPL splats them into
``console.print``. Background-job and subagent-Task lifecycle cards keep
their distinct bordered panels -- those are rare, stateful events, not
per-call noise.

Plugs into the existing ``on_event(ev: ToolEvent)`` hook in
``ClawdREPL.chat``. Uses ``summarize_tool_use`` / ``summarize_tool_result``
from ``src.tool_system.agent_loop``; nothing in the agent loop or the
existing summarize helpers is duplicated or modified here.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Callable

from rich.panel import Panel
from rich.text import Text

from ..tool_system.agent_loop import summarize_tool_result, summarize_tool_use

_MAX_DIFF_LINES = 40


def _unified_diff_lines(
    before: str,
    after: str,
    fromfile: str = "before",
    tofile: str = "after",
) -> list[str]:
    """Full unified diff of two strings (untruncated, terminators kept)."""
    return list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
            n=3,
            lineterm="",
        )
    )


def _diff_stat(diff: list[str]) -> tuple[int, int]:
    """``(added, removed)`` line counts over a full unified diff.

    Counts only real ``+``/``-`` content lines: the ``---``/``+++`` file
    headers and ``@@`` hunk headers are excluded.
    """
    added = removed = 0
    for line in diff:
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


_TASK_TOOL_NAMES = frozenset({"agent", "task"})


def is_task_tool(tool_name: str) -> bool:
    """True for the delegation tool (``Agent``, invocable via its ``Task`` alias).

    Used by the REPL to render dedicated task start/finish cards instead of
    the generic tool-call card. Matches the exact tool name only — task-list
    tools like ``TaskCreate``/``TaskGet`` are not included.
    """
    return isinstance(tool_name, str) and tool_name.lower() in _TASK_TOOL_NAMES


def _is_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _diff_renderables(
    diff: list[str],
    max_lines: int = _MAX_DIFF_LINES,
) -> list[Text]:
    """Unified diff lines as red/green rich Text lines, truncated to max_lines."""
    rendered: list[Text] = []
    overflow = max(0, len(diff) - max_lines)
    for line in diff[:max_lines]:
        # difflib yields lines with their original terminators (keepends=True
        # input); strip them so Text("\n").join() below doesn't double-space.
        line = line.rstrip("\r\n")
        if line.startswith("+++") or line.startswith("---"):
            rendered.append(Text(line, style="dim"))
        elif line.startswith("@@"):
            rendered.append(Text(line, style="dim cyan"))
        elif line.startswith("+"):
            rendered.append(Text(line, style="green"))
        elif line.startswith("-"):
            rendered.append(Text(line, style="red"))
        else:
            rendered.append(Text(line))
    if overflow:
        rendered.append(Text(f"… {overflow} more line{'s' if overflow != 1 else ''}", style="dim"))
    return rendered


def _read_file_safely(path_str: str) -> str | None:
    """Read a text file from disk; return None on any failure."""
    try:
        p = Path(path_str).expanduser()
        if not p.is_file():
            return None
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


class ToolCardRenderer:
    """Stateful quiet renderer for tool calls.

    Instantiate once per chat turn and call ``render_use`` on the ``tool_use``
    event, ``render_result`` on ``tool_result``, and ``render_error`` on
    ``tool_error``. Each returns a list of renderables (usually a single
    line) for the REPL to print. ``shorten`` is an optional callable (the
    REPL's ``_shorten_path_text``) applied to summaries for the existing
    cwd/workspace-relative display behavior.
    """

    def __init__(
        self,
        console,
        shorten: Callable[[str], str] | None = None,
    ) -> None:
        self.console = console
        self._shorten = shorten or (lambda s: s)
        # file path -> (before-content, after-content) captured at tool_use time
        self._write_stash: dict[str, tuple[str | None, str | None]] = {}
        # file path -> (before-content, old_string, new_string, replace_all)
        self._edit_stash: dict[str, tuple[str | None, str | None, str | None, bool]] = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _key_input(self, tool_name: str, tool_input: dict[str, Any] | None) -> str:
        """One-line key-input description, falling back to summarize_tool_use."""
        try:
            summary = summarize_tool_use(tool_name, tool_input or {})
        except Exception:
            summary = ""
        if _is_str(summary):
            return self._shorten(str(summary))
        return ""

    def _result_summary(self, tool_name: str, tool_output: Any) -> str:
        try:
            msg = summarize_tool_result(tool_name, tool_output)
        except Exception:
            msg = ""
        if not isinstance(msg, str):
            return ""
        prefix = f"{tool_name} · "
        if msg.startswith(prefix):
            msg = msg[len(prefix):]
        return self._shorten(msg)

    @staticmethod
    def _path_of(tool_name: str, payload: dict[str, Any] | None) -> str | None:
        if not isinstance(payload, dict):
            return None
        for key in ("file_path", "filePath", "path"):
            v = payload.get(key)
            if _is_str(v):
                return str(v)
        return None

    # ------------------------------------------------------------------
    # public API -- quiet lines (lists of renderables, usually one line)
    # ------------------------------------------------------------------
    def render_use(
        self, tool_name: str, tool_input: dict[str, Any] | None
    ) -> list:
        """One dim line shown when a tool call starts: ``⠋ Read src/foo.py``.

        Background-job starts keep their distinct panel. Write/Edit stash
        before-content here so ``render_result`` can diff without panels.
        """
        if self._is_background_start(tool_name, tool_input):
            return [self._render_background_start(tool_input or {})]
        lowered = tool_name.lower()

        if lowered == "write" and isinstance(tool_input, dict):
            path = self._path_of(tool_name, tool_input)
            content = tool_input.get("content")
            if path:
                self._write_stash[path] = (
                    _read_file_safely(path),
                    content if isinstance(content, str) else None,
                )

        if lowered == "edit" and isinstance(tool_input, dict):
            path = self._path_of(tool_name, tool_input)
            old = tool_input.get("old_string")
            new = tool_input.get("new_string")
            if path:
                self._edit_stash[path] = (
                    _read_file_safely(path),
                    old if isinstance(old, str) else None,
                    new if isinstance(new, str) else None,
                    bool(tool_input.get("replace_all", False)),
                )

        key = self._key_input(tool_name, tool_input)
        label = f"{tool_name} {key}".strip() if key else tool_name
        return [Text(f"⠋ {label}", style="dim")]

    # ------------------------------------------------------------------
    # background Bash jobs
    # ------------------------------------------------------------------
    @staticmethod
    def _is_background_start(tool_name: str, tool_input: dict[str, Any] | None) -> bool:
        return (
            tool_name.lower() == "bash"
            and isinstance(tool_input, dict)
            and bool(tool_input.get("run_in_background"))
            and not tool_input.get("action")
        )

    def _render_background_start(self, tool_input: dict[str, Any]) -> Panel:
        cmd = tool_input.get("command")
        body = [Text(self._shorten(str(cmd)) if isinstance(cmd, str) else "…", style="dim")]
        return Panel(
            Text("\n").join(body),
            title="[magenta]Bash[/]",
            subtitle="[magenta]background job · starting…[/]",
            border_style="magenta",
            expand=False,
        )

    def _render_background_result(
        self, tool_name: str, output_dict: dict[str, Any], is_error: bool = False
    ) -> Panel | None:
        """Distinct card for background-job lifecycle results.

        Returns None when the output is not a background-job payload, so
        normal Bash rendering is untouched.
        """
        job_id = output_dict.get("job_id")
        if not isinstance(job_id, str):
            return None

        def _job_id_line() -> Text:
            return Text(f"job {job_id}", style="dim")

        if output_dict.get("started"):
            cmd = output_dict.get("command")
            lines = [
                _job_id_line(),
                Text(
                    f"poll with Bash(action='poll', job_id='{job_id}')",
                    style="dim",
                ),
            ]
            if isinstance(cmd, str) and cmd.strip():
                lines.insert(0, Text(self._shorten(cmd), style="dim"))
            return Panel(
                Text("\n").join(lines),
                title="[magenta]Bash[/]",
                subtitle="[magenta]background · running[/]",
                border_style="magenta",
                expand=False,
            )

        if output_dict.get("forgotten"):
            return Panel(
                Text(f"job {job_id} forgotten", style="dim"),
                title="[magenta]Bash[/]",
                subtitle="[magenta]background · done[/]",
                border_style="magenta",
                expand=False,
            )

        if output_dict.get("killed"):
            return Panel(
                Text(f"job {job_id} killed", style="dim"),
                title="[magenta]Bash[/]",
                subtitle="[magenta]background · done[/]",
                border_style="magenta",
                expand=False,
            )

        if "running" in output_dict:
            running = bool(output_dict["running"])
            stdout = output_dict.get("stdout")
            tail = ""
            if isinstance(stdout, str) and stdout.strip():
                tail_lines = stdout.strip().splitlines()[-8:]
                tail = "\n".join(tail_lines)
            # A finished job with a nonzero exit code failed: tint it red.
            failed = is_error and not running
            border = "red" if failed else "magenta"
            title_style = "red" if failed else "magenta"
            if running:
                subtitle = "[magenta]background · running[/]"
                body = Text(tail or "…", style="dim")
            else:
                rc = output_dict.get("exit_code")
                subtitle = f"[{title_style}]background · done (exit {rc})[/]"
                body = Text(tail or "(no output)", style="dim")
            header = Text(f"job {job_id}", style="dim")
            return Panel(
                Text("\n").join([header, body]) if tail else header,
                title=f"[{title_style}]Bash[/]",
                subtitle=subtitle,
                border_style=border,
                expand=False,
            )

        return None

    def render_result(
        self,
        tool_name: str,
        tool_output: Any,
        is_error: bool,
    ) -> list:
        """Quiet result rendering.

        - errors: one red line (red is reserved for errors);
        - Write/Edit with a real diff: one dim ``✓`` line + red-green diff
          lines (no panel);
        - everything else: one dim ``✓`` line.
        Background-job payloads keep their distinct panel.
        """
        lowered = tool_name.lower()
        output_dict = tool_output if isinstance(tool_output, dict) else {}

        # Background-job payloads get their own card even when the poll
        # reports a nonzero exit code (is_error=True).
        if lowered == "bash":
            bg_panel = self._render_background_result(
                tool_name, output_dict, is_error=is_error
            )
            if bg_panel is not None:
                return [bg_panel]

        if is_error:
            msg = ""
            if isinstance(output_dict.get("error"), str):
                msg = output_dict["error"]
            if not msg:
                msg = self._result_summary(tool_name, tool_output) or "Error"
            return [Text(f"✗ {tool_name} — {msg}", style="red")]

        diff_lines: list[Text] = []
        diff_added = diff_removed = 0
        if lowered == "write":
            diff_lines, (diff_added, diff_removed) = self._write_diff(
                tool_name, output_dict
            )
        elif lowered == "edit":
            diff_lines, (diff_added, diff_removed) = self._edit_diff(
                tool_name, output_dict
            )

        summary = self._result_summary(tool_name, tool_output)
        label = f"{tool_name} {summary}".strip() if summary else tool_name
        first = Text(f"✓ {label}", style="dim")
        if diff_lines:
            # Compact (+N, -M) stat on the completion line, gemini-cli style:
            # at-a-glance change size without reading the full diff below.
            first.append_text(Text(" (", style="dim"))
            first.append_text(Text(f"+{diff_added}", style="green"))
            first.append_text(Text(", ", style="dim"))
            first.append_text(Text(f"-{diff_removed}", style="red"))
            first.append_text(Text(")", style="dim"))
        lines: list = [first]
        lines.extend(diff_lines)
        return lines

    def render_error(self, tool_name: str, error: str | None) -> list:
        """One red line for a ``tool_error`` event (exception during dispatch)."""
        return [Text(f"✗ {tool_name} — {error or 'Error'}", style="red")]

    # ------------------------------------------------------------------
    # task (subagent delegation) cards
    # ------------------------------------------------------------------
    def render_task_start(
        self,
        tool_name: str,
        tool_input: dict[str, Any] | None,
        task_id: str | None = None,
    ) -> Panel:
        """Distinct card shown when a subagent Task starts.

        ``task_id`` is the tool-call id from the ``tool_use`` event — the
        genuine identifier of this delegation in the conversation.
        """
        payload = tool_input if isinstance(tool_input, dict) else {}
        description = payload.get("description")
        desc_text = (
            self._shorten(description)
            if isinstance(description, str) and description.strip()
            else "…"
        )
        call_names: list[str] = []
        calls = payload.get("calls")
        if isinstance(calls, list):
            for c in calls:
                if isinstance(c, dict) and isinstance(c.get("name"), str):
                    call_names.append(str(c["name"]))

        lines: list[Text] = [Text(f"task id: {task_id or '—'}", style="dim")]
        lines.append(Text(desc_text))
        if call_names:
            preview = " → ".join(call_names[:5])
            if len(call_names) > 5:
                preview += f" → … (+{len(call_names) - 5} more)"
            noun = "call" if len(call_names) == 1 else "calls"
            lines.append(Text(f"{len(call_names)} {noun}: {preview}", style="dim"))

        return Panel(
            Text("\n").join(lines),
            title="[magenta]Task[/]",
            subtitle="[dim]running…[/]",
            border_style="magenta",
            expand=False,
        )

    def render_task_finish(
        self,
        tool_name: str,
        tool_output: Any,
        task_id: str | None = None,
        is_error: bool = False,
    ) -> Panel:
        """Completion card for a subagent Task: success/failure + where output went."""
        output = tool_output if isinstance(tool_output, dict) else {}
        lines: list[Text] = [Text(f"task id: {task_id or '—'}", style="dim")]

        num_succeeded = output.get("num_succeeded")
        num_failed = output.get("num_failed")
        denials = output.get("permission_denials")
        num_denials = len(denials) if isinstance(denials, list) else 0

        if is_error:
            err = output.get("error") if isinstance(output.get("error"), str) else ""
            note = output.get("note") if isinstance(output.get("note"), str) else ""
            if num_succeeded is not None and num_failed is not None:
                lines.append(
                    Text(
                        f"✗ failed after {num_succeeded}/{num_succeeded + num_failed} calls succeeded",
                        style="red",
                    )
                )
            else:
                lines.append(Text(f"✗ {err or 'task failed'}", style="red"))
            if note:
                lines.append(Text(note, style="dim"))
        else:
            if num_succeeded is not None and num_failed is not None:
                lines.append(
                    Text(
                        f"✓ {num_succeeded}/{num_succeeded + num_failed} calls succeeded",
                        style="green",
                    )
                )
            else:
                lines.append(Text("✓ done", style="green"))

        if num_denials:
            noun = "call" if num_denials == 1 else "calls"
            lines.append(
                Text(
                    f"⚠ {num_denials} {noun} auto-denied: needed user approval, "
                    "unavailable inside a subagent task",
                    style="yellow",
                )
            )

        lines.append(
            Text("full per-call output returned in the tool result", style="dim")
        )

        return Panel(
            Text("\n").join(lines),
            title="[magenta]Task[/]",
            subtitle="[red]failed[/]" if is_error else "[dim]done[/]",
            border_style="red" if is_error else "green",
            expand=False,
        )

    # ------------------------------------------------------------------
    # diff plumbing for Write / Edit
    # ------------------------------------------------------------------
    def _write_diff(
        self, tool_name: str, output: dict[str, Any]
    ) -> tuple[list[Text], tuple[int, int]]:
        """Diff renderables + ``(added, removed)`` stat for a Write call."""
        path = self._path_of(tool_name, output)
        before: str | None = None
        after: str | None = None
        if path and path in self._write_stash:
            before, after = self._write_stash.pop(path)
        if before is None and isinstance(output.get("originalFile"), str):
            before = output["originalFile"]
        if after is None and isinstance(output.get("content"), str):
            after = output["content"]
        if after is None:
            return [], (0, 0)
        diff = _unified_diff_lines(
            before or "", after, fromfile=path or "before", tofile=path or "after"
        )
        return _diff_renderables(diff), _diff_stat(diff)

    def _edit_diff(
        self, tool_name: str, output: dict[str, Any]
    ) -> tuple[list[Text], tuple[int, int]]:
        """Diff renderables + ``(added, removed)`` stat for an Edit call."""
        path = self._path_of(tool_name, output)
        before: str | None = None
        old: str | None = None
        new: str | None = None
        replace_all = False
        if isinstance(output.get("originalFile"), str):
            before = output["originalFile"]
        if isinstance(output.get("oldString"), str):
            old = output["oldString"]
        if isinstance(output.get("newString"), str):
            new = output["newString"]
        replace_all = bool(output.get("replaceAll", False))
        if before is None and path and path in self._edit_stash:
            before, old, new, replace_all = self._edit_stash.pop(path)
        if before is None or not old or not isinstance(new, str):
            return [], (0, 0)
        if old not in before:
            return [], (0, 0)
        after = before.replace(old, new) if replace_all else before.replace(old, new, 1)
        diff = _unified_diff_lines(
            before, after, fromfile=path or "before", tofile=path or "after"
        )
        return _diff_renderables(diff), _diff_stat(diff)
