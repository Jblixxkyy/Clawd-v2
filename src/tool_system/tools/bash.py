from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..protocol import ToolResult
from ..registry import ToolSpec


_DANGEROUS_PATTERNS = [
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd\b\s+if=", re.IGNORECASE),
    re.compile(r"\brm\b.*\s+-rf\s+/\s*$", re.IGNORECASE),
    re.compile(r"\brm\b.*\s+-rf\s+/\s+"),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE),
]


def _truncate(s: str, limit: int = 20000) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + "\n\n... [truncated] ..."


def _bash_missing_result(command: str, cwd: Path) -> ToolResult:
    """Clean tool error for systems without a bash executable (stock Windows).

    Actionable: tells the user/agent to install Git Bash and points at the
    setup guide, instead of a bare "not found".
    """
    return ToolResult(
        name="Bash",
        output={
            "command": command,
            "cwd": str(cwd),
            "error": (
                "bash executable not found on this system. The Bash tool "
                "requires bash and runs bash syntax only -- it is never "
                "translated to another shell. On Windows, install Git Bash "
                "(https://git-scm.com/downloads) and restart, or see "
                "docs/guide/SETUP_GUIDE.md. If the PowerShell tool is enabled "
                '("enable_powershell": true in ~/.clawd/config.json), use it '
                "with PowerShell syntax instead."
            ),
            "started": False,
        },
        is_error=True,
    )


def _try_extract_cd(command: str) -> Path | None:
    stripped = command.strip()
    if not stripped.startswith("cd "):
        return None
    try:
        parts = shlex.split(stripped, posix=True)
    except ValueError:
        return None
    if len(parts) >= 2 and parts[0] == "cd":
        return Path(parts[1])
    return None


# ----------------------------------------------------------------------
# Background job registry.
#
# A deliberately small registry: job_id -> {process handle, captured output
# files, status}. No fancy cleanup — jobs live until the agent calls
# action="forget" (which also kills a still-running process and removes its
# output files). A registry entry is per-process, so background jobs do not
# survive a process restart.
# ----------------------------------------------------------------------

_BG_JOBS: dict[str, dict[str, Any]] = {}
_BG_LOCK = threading.Lock()
_BG_ACTIONS = ("poll", "forget", "kill")


def _new_job_id() -> str:
    return "job_" + uuid.uuid4().hex[:8]


def _get_job(job_id: Any) -> dict[str, Any]:
    if not isinstance(job_id, str) or not job_id.strip():
        raise ToolInputError("job_id must be a non-empty string")
    with _BG_LOCK:
        job = _BG_JOBS.get(job_id)
    if job is None:
        raise ToolInputError(f"unknown job_id: {job_id} (forgotten or never started)")
    return job


def _read_job_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = f.read()
    except OSError:
        return ""
    return _truncate(data)


def _start_background_job(command: str, cwd: Path) -> ToolResult:
    out_f = tempfile.NamedTemporaryFile(
        prefix="clawd-bg-", suffix=".stdout", delete=False
    )
    err_f = tempfile.NamedTemporaryFile(
        prefix="clawd-bg-", suffix=".stderr", delete=False
    )
    out_path, err_path = out_f.name, err_f.name
    try:
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=out_f,
            stderr=err_f,
            start_new_session=True,
        )
    except FileNotFoundError:
        # bash is not installed (e.g. stock Windows): fail as a clean tool
        # error instead of an uncaught traceback. On POSIX bash exists, so
        # this branch never fires and behavior is unchanged.
        for path in (out_path, err_path):
            try:
                os.unlink(path)
            except OSError:
                pass
        return _bash_missing_result(command, cwd)
    finally:
        out_f.close()
        err_f.close()
    job_id = _new_job_id()
    with _BG_LOCK:
        _BG_JOBS[job_id] = {
            "id": job_id,
            "command": command,
            "cwd": str(cwd),
            "pid": proc.pid,
            "process": proc,
            "out_path": out_path,
            "err_path": err_path,
            "started_at": time.time(),
        }
    return ToolResult(
        name="Bash",
        output={
            "job_id": job_id,
            "command": command,
            "cwd": str(cwd),
            "running": True,
            "started": True,
        },
        is_error=False,
    )


def _poll_background_job(job_id: str) -> ToolResult:
    job = _get_job(job_id)
    proc = job["process"]
    returncode = proc.poll()
    running = returncode is None
    output: dict[str, Any] = {
        "job_id": job_id,
        "command": job["command"],
        "running": running,
        "stdout": _read_job_file(job["out_path"]),
        "stderr": _read_job_file(job["err_path"]),
    }
    if not running:
        output["exit_code"] = returncode
    return ToolResult(
        name="Bash",
        output=output,
        is_error=(returncode is not None and returncode != 0),
    )


def _stop_process(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        return
    try:
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def _kill_background_job(job_id: str) -> ToolResult:
    job = _get_job(job_id)
    was_running = job["process"].poll() is None
    _stop_process(job["process"])
    return ToolResult(
        name="Bash",
        output={"job_id": job_id, "killed": True, "was_running": was_running},
        is_error=False,
    )


def _forget_background_job(job_id: str) -> ToolResult:
    job = _get_job(job_id)
    _stop_process(job["process"])
    with _BG_LOCK:
        _BG_JOBS.pop(job_id, None)
    for path in (job["out_path"], job["err_path"]):
        try:
            os.unlink(path)
        except OSError:
            pass
    return ToolResult(
        name="Bash",
        output={"job_id": job_id, "forgotten": True},
        is_error=False,
    )


class BashTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Bash",
            description=(
                "Execute a shell command. Pass run_in_background=true to run "
                "the command detached and get back a job_id immediately; then "
                "use action='poll' to read its output, action='kill' to stop "
                "it, or action='forget' to remove it (forget also kills a "
                "still-running job). timeout_s is ignored for background jobs."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout_s": {"type": "integer"},
                    "run_in_background": {"type": "boolean"},
                    "action": {"type": "string", "enum": list(_BG_ACTIONS)},
                    "job_id": {"type": "string"},
                },
                "required": [],
            },
            is_destructive=True,

        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        action = tool_input.get("action")
        if action is not None:
            # Job-management path: no new command is executed here, so the
            # dangerous-pattern scan does not apply (there is nothing to scan).
            if action not in _BG_ACTIONS:
                raise ToolInputError(
                    f"action must be one of {', '.join(_BG_ACTIONS)}"
                )
            job_id = tool_input.get("job_id")
            if action == "poll":
                return _poll_background_job(job_id)
            if action == "kill":
                return _kill_background_job(job_id)
            return _forget_background_job(job_id)

        command = tool_input.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolInputError("command must be a non-empty string")
        if "\x00" in command:
            raise ToolInputError("command contains NUL byte")

        for pat in _DANGEROUS_PATTERNS:
            if pat.search(command):
                raise ToolPermissionError("refusing to run potentially dangerous command")

        explicit_cwd = tool_input.get("cwd")
        if explicit_cwd is not None:
            # os.path.isabs is platform-neutral: identical to startswith("/")
            # on POSIX, and additionally accepts e.g. "C:\\..." on Windows.
            if not isinstance(explicit_cwd, str) or not os.path.isabs(explicit_cwd):
                raise ToolInputError("cwd must be an absolute path when provided")
            cwd = context.ensure_allowed_path(explicit_cwd)
        else:
            cwd = context.cwd or context.workspace_root

        if tool_input.get("run_in_background"):
            if not isinstance(tool_input.get("run_in_background"), bool):
                raise ToolInputError("run_in_background must be a boolean")
            # A bare `cd` in the background is meaningless (it only mutates
            # this session's context.cwd), so skip the cd special-case below.
            return _start_background_job(command, cwd)

        cd_target = _try_extract_cd(command)
        if cd_target is not None and command.strip().startswith("cd ") and len(command.strip().splitlines()) == 1:
            next_dir = (cwd / cd_target).expanduser().resolve() if not cd_target.is_absolute() else cd_target.expanduser().resolve()
            next_dir = context.ensure_allowed_path(next_dir)
            if not next_dir.exists() or not next_dir.is_dir():
                return ToolResult(name="Bash", output={"error": f"directory does not exist: {next_dir}"}, is_error=True)
            context.cwd = next_dir
            return ToolResult(name="Bash", output={"cwd": str(context.cwd), "stdout": "", "stderr": ""})

        timeout_s = tool_input.get("timeout_s", 60)
        if not isinstance(timeout_s, int) or timeout_s < 1 or timeout_s > 600:
            raise ToolInputError("timeout_s must be an integer between 1 and 600")

        try:
            completed = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except FileNotFoundError:
            # bash is not installed (e.g. stock Windows): fail as a clean tool
            # error instead of an uncaught traceback. On POSIX bash exists, so
            # this branch never fires and behavior is unchanged.
            return _bash_missing_result(command, cwd)

        stdout = _truncate(completed.stdout or "")
        stderr = _truncate(completed.stderr or "")
        output: dict[str, Any] = {
            "cwd": str(cwd),
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
        return ToolResult(name="Bash", output=output, is_error=completed.returncode != 0)

