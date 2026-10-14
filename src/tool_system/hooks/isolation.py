"""Isolated execution context for hook/plugin code.

Crash isolation, not a security boundary.

Following cline/cline@789f79e4, plugin/hook code never runs on the host's
main thread: every hook command executes inside a dedicated (daemon) worker
thread, and the hook itself is always a child process (``shell=True``), so
a segfaulting or non-zero-exiting hook dies inside its own process while
the worker thread's exceptions (including ``SystemExit``/``KeyboardInterrupt``
raised by plugin-side code) are captured, never propagated to the agent
loop. A per-plugin timeout abandons the worker; the loop continues
fail-open and the plugin is marked failed in the audit log
(``~/.clawd/hooks-audit.log``).

This protects loop liveness, not confidentiality or integrity: hook code
still runs with the harness's privileges and environment. Do not treat it
as a sandbox.

Loud failures: a plugin that cannot load (missing bootstrap script or
interpreter, bad config) raises a *named* error -- ``PluginLoadError``
subclasses -- carrying the exact missing piece and an actionable fix.
Callers that must stay fail-open (the runner, session start) collect these
via :func:`plugin_load_warnings` and surface them in ``/doctor`` output
and session-start warnings instead of skipping quietly.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Isolated execution
# ---------------------------------------------------------------------------

#: Extra grace beyond the hook's own timeout before the worker thread itself
#: is abandoned. The worker is a daemon thread, so abandoning it never
#: blocks interpreter shutdown.
_WORKER_GRACE_SECONDS = 5.0


def run_in_worker(
    fn: Callable[..., Any],
    *args: Any,
    timeout: float,
    name: str = "clawd-hook-worker",
) -> tuple[bool, Any]:
    """Run ``fn(*args)`` on a dedicated daemon worker thread.

    Returns ``(True, result)`` when ``fn`` returned in time,
    ``(False, exception)`` when it raised (any ``BaseException``, including
    ``SystemExit``/``KeyboardInterrupt`` -- captured, never propagated), and
    ``(False, TimeoutError(...))`` when the worker did not finish within
    ``timeout`` seconds (the abandoned worker is a daemon thread and cannot
    block the caller or interpreter shutdown).

    This function never raises: the host's main thread cannot be poisoned
    by whatever the worker did.
    """
    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["result"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001 - capture everything
            box["error"] = exc

    worker = threading.Thread(target=_target, name=name, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return False, TimeoutError(
            f"worker thread did not finish within {timeout:.1f}s; abandoned"
        )
    if "error" in box:
        return False, box["error"]
    return True, box.get("result")


@dataclass
class IsolatedResult:
    """Outcome of one hook command run inside the isolated context."""

    #: "ok" | "timeout" | "crashed" | "error"
    status: str
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    elapsed: float = 0.0
    #: Human-readable cause for "timeout"/"crashed"/"error".
    error: Optional[str] = None
    #: True when the host thread's own wait expired (defense in depth).
    worker_abandoned: bool = False


def run_command_isolated(
    command: str,
    stdin_data: str,
    timeout: float,
    env: Optional[dict[str, str]] = None,
) -> IsolatedResult:
    """Run one hook *command* in the isolated execution context.

    The subprocess runs on a dedicated daemon worker thread -- never the
    caller's (loop's) thread. Two timeout layers apply: the subprocess is
    killed after ``timeout`` seconds, and the worker itself is abandoned
    after ``timeout`` + a grace period even if the kill hangs. Exceptions
    from the worker are captured, never raised.
    """
    started = time.monotonic()

    def _run() -> subprocess.CompletedProcess:
        return subprocess.run(
            command,
            input=stdin_data,
            capture_output=True,
            text=True,
            shell=True,
            timeout=timeout,
            env=env,
        )

    completed, payload = run_in_worker(
        _run,
        timeout=timeout + _WORKER_GRACE_SECONDS,
        name=f"clawd-hook:{command[:48]}",
    )
    elapsed = time.monotonic() - started

    if not completed:
        if isinstance(payload, subprocess.TimeoutExpired):
            return IsolatedResult(
                status="timeout",
                elapsed=elapsed,
                error=(
                    f"hook command timed out after {timeout:.1f}s and was killed: "
                    f"{command}"
                ),
            )
        if isinstance(payload, TimeoutError):
            return IsolatedResult(
                status="timeout",
                elapsed=elapsed,
                worker_abandoned=True,
                error=(
                    f"hook execution exceeded {timeout:.1f}s and the worker "
                    f"thread was abandoned: {command}"
                ),
            )
        return IsolatedResult(
            status="error",
            elapsed=elapsed,
            error=f"hook worker crashed before producing a result: {payload!r}",
        )
    if isinstance(payload, subprocess.TimeoutExpired):
        return IsolatedResult(
            status="timeout",
            elapsed=elapsed,
            error=(
                f"hook command timed out after {timeout:.1f}s and was killed: "
                f"{command}"
            ),
        )
    if isinstance(payload, BaseException):
        return IsolatedResult(
            status="error",
            elapsed=elapsed,
            error=f"hook command failed to launch: {payload}",
        )
    proc = payload
    if proc.returncode != 0:
        return IsolatedResult(
            status="crashed",
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            elapsed=elapsed,
            error=(
                f"hook command exited with code {proc.returncode}: {command}"
                + (f" — stderr: {(proc.stderr or '').strip()[:300]}" if (proc.stderr or "").strip() else "")
            ),
        )
    return IsolatedResult(
        status="ok",
        returncode=0,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        elapsed=elapsed,
    )


# ---------------------------------------------------------------------------
# Named plugin-load errors (loud failures, no silent skips)
# ---------------------------------------------------------------------------


class PluginLoadError(Exception):
    """A hook plugin could not load.

    ``plugin`` names the hook command/entry, ``missing`` names the exact
    missing piece (path, interpreter, config key, JSON validity), and
    ``fix`` is an actionable remediation. ``str(err)`` is the full named
    message suitable for /doctor output and session-start warnings.
    """

    error_name = "PluginLoadError"

    def __init__(self, plugin: str, missing: str, fix: str) -> None:
        self.plugin = plugin
        self.missing = missing
        self.fix = fix
        super().__init__(
            f"{self.error_name}: hook plugin {plugin!r} cannot load — "
            f"missing {missing}. Fix: {fix}"
        )


class PluginBootstrapMissingError(PluginLoadError):
    """The plugin's bootstrap is missing: script file or interpreter absent."""

    error_name = "PluginBootstrapMissingError"


class PluginConfigError(PluginLoadError):
    """The plugin config itself is broken: invalid JSON, wrong shape, no command."""

    error_name = "PluginConfigError"


_SCRIPT_SUFFIXES = (".py", ".sh", ".js", ".mjs", ".ts")


def _runnable(first_token: str) -> bool:
    if os.path.isabs(first_token):
        return os.path.isfile(first_token) and os.access(first_token, os.X_OK)
    return shutil.which(first_token) is not None


def validate_hook_entry(event: str, entry: Any, index: int = 0) -> str:
    """Validate one hooks.json entry; return its normalized command string.

    Raises a named :class:`PluginLoadError` subclass naming the exact
    missing piece when the plugin cannot load. Never silently accepts a
    broken entry.
    """
    label = f"{event}[{index}]"
    if isinstance(entry, dict):
        command = entry.get("command")
    elif isinstance(entry, str):
        command = entry
    else:
        raise PluginConfigError(
            plugin=label,
            missing=f"a command string or {{'command': ...}} object (got {type(entry).__name__})",
            fix="give the entry a 'command' key, e.g. "
            f'{{\"{event}\": [{{\"command\": \"python3 /path/to/hook.py\"}}]}}',
        )
    if not command or not str(command).strip():
        raise PluginConfigError(
            plugin=label,
            missing="a non-empty 'command' value",
            fix="set the entry's 'command' to the shell command that runs the hook, "
            "e.g. {\"command\": \"python3 /path/to/hook.py\"}",
        )
    command = str(command)
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if not tokens:
        raise PluginConfigError(
            plugin=label,
            missing="a parseable command (shlex could not split it)",
            fix="quote the command properly, e.g. {\"command\": \"python3 /path/to/hook.py\"}",
        )
    interp, rest = tokens[0], tokens[1:]
    script_tokens = [tok for tok in rest if not tok.startswith("-")]
    scripts = [tok for tok in script_tokens if tok.endswith(_SCRIPT_SUFFIXES)]
    if scripts:
        # interpreter + script form, e.g. "python3 /path/to/hook.py"
        if not _runnable(interp):
            raise PluginBootstrapMissingError(
                plugin=f"{label} {command!r}",
                missing=f"interpreter {interp!r}",
                fix=f"install {interp!r} or fix the command in ~/.clawd/hooks.json",
            )
        for script in scripts:
            spath = Path(script).expanduser()
            if not spath.is_file():
                raise PluginBootstrapMissingError(
                    plugin=f"{label} {command!r}",
                    missing=f"hook script file {script}",
                    fix=f"create the script at {script} or fix the path in ~/.clawd/hooks.json",
                )
    else:
        # bare executable form
        if not _runnable(interp):
            raise PluginBootstrapMissingError(
                plugin=f"{label} {command!r}",
                missing=f"runnable command {interp!r}",
                fix=f"install {interp!r} or fix the command in ~/.clawd/hooks.json",
            )
    return command


def validate_hook_config(config: Any, source: str = "~/.clawd/hooks.json") -> None:
    """Validate a parsed hooks.json config; raise a named error on any problem.

    Raises :class:`PluginConfigError` / :class:`PluginBootstrapMissingError`
    naming the exact missing piece. An empty config (``{}``) is valid — no
    hooks configured is not a load failure.
    """
    if not isinstance(config, dict):
        raise PluginConfigError(
            plugin=source,
            missing="a JSON object at the top level",
            fix=f"make {source} an object mapping event names to hook commands, "
            'e.g. {"PreToolUse": [{"command": "python3 /path/to/hook.py"}]}',
        )
    for event, raw in config.items():
        entries = [raw] if isinstance(raw, str) else (raw if isinstance(raw, list) else None)
        if entries is None:
            raise PluginConfigError(
                plugin=f"{source}:{event}",
                missing="a command string or list of {'command': ...} entries",
                fix=f'give "{event}" a command string or a list, e.g. '
                f'{{\"{event}\": [{{\"command\": \"python3 /path/to/hook.py\"}}]}}',
            )
        for index, entry in enumerate(entries):
            validate_hook_entry(str(event), entry, index)


def _default_config_path() -> Path:
    return Path.home() / ".clawd" / "hooks.json"


def plugin_load_warnings(config_path: Optional[Path] = None) -> list[PluginLoadError]:
    """Collect named plugin-load errors for hooks.json without raising.

    Returns [] when hooks.json is absent (no hooks configured is not a
    load failure) or fully valid. A present-but-broken config yields one
    named error per problem -- surfaced by /doctor and session-start
    warnings instead of skipped quietly.
    """
    path = config_path or _default_config_path()
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        return [
            PluginConfigError(
                plugin=str(path),
                missing=f"readable file ({exc})",
                fix=f"check ownership/permissions of {path}",
            )
        ]
    import json as _json

    try:
        config = _json.loads(text)
    except ValueError as exc:
        return [
            PluginConfigError(
                plugin=str(path),
                missing=f"valid JSON ({exc})",
                fix=f"fix the JSON syntax in {path}, or delete it to disable hooks",
            )
        ]
    errors: list[PluginLoadError] = []
    if not isinstance(config, dict):
        errors.append(
            PluginConfigError(
                plugin=str(path),
                missing="a JSON object at the top level",
                fix=f"make {path} an object mapping event names to hook commands",
            )
        )
        return errors
    for event, raw in config.items():
        entries = [raw] if isinstance(raw, str) else (raw if isinstance(raw, list) else None)
        if entries is None:
            errors.append(
                PluginConfigError(
                    plugin=f"{path}:{event}",
                    missing="a command string or list of {'command': ...} entries",
                    fix=f'give "{event}" a command string or a list of {{"command": ...}} entries',
                )
            )
            continue
        for index, entry in enumerate(entries):
            try:
                validate_hook_entry(str(event), entry, index)
            except PluginLoadError as err:
                errors.append(err)
    return errors
