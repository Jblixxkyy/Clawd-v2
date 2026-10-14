"""Non-interactive script mode for Clawd Codex.

`clawd -p "do X"` runs a single agent turn and prints the final response
text to stdout, then exits. No TTY required.

Setup mirrors ClawdREPL.__init__ (see src/repl/core.py) minus prompt_toolkit:
provider config -> provider instance -> Session -> tool registry ->
ToolContext(workspace_root=Path.cwd()).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.agent import Session
from src.config import get_default_provider, get_provider_config
from src.providers import get_provider_class
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry

# Exit codes
EXIT_OK = 0
EXIT_NO_KEY = 1  # provider misconfigured / no API key
EXIT_ERROR = 2  # anything else failed cleanly

# NOTE: never import prompt_toolkit here (no TTY in script mode).


def _resolve_auto_compact(auto_compact: bool | None) -> bool:
    """Resolve the auto-compact flag: explicit value wins, else config, else on."""
    if auto_compact is not None:
        return auto_compact
    try:
        from src.config import load_config
        return bool(load_config().get("auto_compact", True))
    except Exception:
        return True


def _warn_permission(tool_name: str, summary: str) -> None:
    """Warn that a tool was denied (auto-deny in script mode)."""
    msg = (
        f"permission required for {tool_name} '{summary}' "
        "- run interactively to approve"
    )
    if sys.stdout.isatty():
        try:
            from rich.console import Console

            Console().print(f"[dim]{msg}[/dim]")
            return
        except Exception:
            pass
    print(msg)


def _deny_permission(
    tool_name: str,
    message: str,
    suggestion: str | None,
) -> tuple[bool, bool]:
    """Permission handler for script mode: SAFE default -- always deny.

    Prints a warning (dim when on a TTY, plain when piped) and returns
    (allowed=False, continue_without_caching=False).
    """
    summary = (message or "").strip().split("\n")[0]
    if summary:
        _warn_permission(tool_name, summary)
    return False, False


def run_print(
    prompt_text: str,
    provider_name: str | None = None,
    stream: bool = False,
    permission_mode: str | None = None,
    auto_compact: bool | None = None,
) -> int:
    """Run ONE agent turn for ``prompt_text`` and print the answer.

    Args:
        prompt_text: The single user prompt to run.
        provider_name: Provider to use; defaults to get_default_provider().
        stream: Pass-through for the agent loop's streaming mode.
        permission_mode: Session permission mode name
            (default/acceptEdits/dontAsk/bypassPermissions); None keeps the
            default.
        auto_compact: Enable/disable auto-compaction; None reads the
            "auto_compact" config option (default enabled).

    Returns:
        0 on success (final response text printed to stdout),
        1 when the provider has no API key configured,
        2 on any other failure (readable message on stderr, no traceback).
    """
    try:
        name = provider_name or get_default_provider()

        try:
            config: dict[str, Any] = get_provider_config(name)
        except Exception:
            print(f"Error: unknown provider '{name}'.", file=sys.stderr)
            return EXIT_NO_KEY

        if not config.get("api_key"):
            print(
                f"Error: API key not configured for provider '{name}'.",
                file=sys.stderr,
            )
            print("Run `clawd login` to configure it.", file=sys.stderr)
            return EXIT_NO_KEY

        # Mirror ClawdREPL.__init__: provider -> session -> registry -> context
        provider_class = get_provider_class(name)
        provider = provider_class(
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            model=config.get("default_model"),
        )

        session = Session.create(name, provider.model)
        session.conversation.add_user_message(prompt_text)

        tool_registry = build_default_registry()
        tool_context = ToolContext(workspace_root=Path.cwd())
        tool_context.permission_handler = _deny_permission
        if permission_mode is not None:
            from src.tool_system.permissions import PermissionMode
            tool_context.permission_mode = PermissionMode.from_string(permission_mode)
        from src.config import powershell_opt_in
        tool_context.enable_powershell = powershell_opt_in()

        result = run_agent_loop(
            session.conversation,
            provider,
            tool_registry,
            tool_context,
            max_turns=50,
            stream=stream,
            auto_compact=_resolve_auto_compact(auto_compact),
            script_mode=True,
        )

        # Pipe-friendly: plain print, no rich markup.
        text = result.response_text or ""
        print(text)
        return EXIT_OK
    except KeyboardInterrupt:
        print("Error: interrupted.", file=sys.stderr)
        return EXIT_ERROR
    except Exception as e:
        # Readable message, NO traceback (script mode must be pipe-clean).
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ERROR
