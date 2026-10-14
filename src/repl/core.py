"""Interactive REPL for Clawd Codex."""

from __future__ import annotations

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style
    from prompt_toolkit.completion import WordCompleter
    try:
        from prompt_toolkit.completion import FuzzyCompleter
    except Exception:  # pragma: no cover
        FuzzyCompleter = None  # type: ignore
    from prompt_toolkit.key_binding import KeyBindings
except ModuleNotFoundError:  # pragma: no cover
    class FileHistory:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class AutoSuggestFromHistory:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class Style:  # type: ignore
        @staticmethod
        def from_dict(*args, **kwargs):
            return None

    class WordCompleter:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
    FuzzyCompleter = None  # type: ignore

    class KeyBindings:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class PromptSession:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def prompt(self, *args, **kwargs):
            raise EOFError()

try:
    from rich.console import Console, Group
    from rich.align import Align
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.columns import Columns
except ModuleNotFoundError:  # pragma: no cover
    class Console:  # type: ignore
        def print(self, *args, **kwargs):
            return None

    Group = None  # type: ignore
    Align = None  # type: ignore
    Panel = None  # type: ignore
    Table = None  # type: ignore
    Text = None  # type: ignore
    Columns = None  # type: ignore

    class Markdown:  # type: ignore
        def __init__(self, text: str):
            self.text = text
from pathlib import Path
import asyncio
import os
import sys
import json
from contextlib import nullcontext
from typing import Any

from src.agent import Session
from src.config import get_provider_config, powershell_opt_in
from src.outputStyles import resolve_output_style
from src.providers import get_provider_class
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import ChatMessage
from src.providers.minimax_provider import MinimaxProvider
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.protocol import ToolCall
from src.tool_system.agent_loop import ToolEvent, run_agent_loop, summarize_tool_result, summarize_tool_use, _fire_lifecycle_hook
from src.repl.tool_cards import ToolCardRenderer, is_task_tool

# New command system imports
from src.command_system import (
    CommandRegistry,
    CommandResult,
    create_command_context,
    execute_command_async,
    execute_command_sync,
    register_builtin_commands,
)
from src.command_surface import (
    TERMINAL_ONLY_COMMANDS,
    build_command_list,
    flatten_prompt_content,
    is_terminal_only,
    parse_slash,
    run_command,
    run_skill_tool,
)
from src.cost_tracker import CostTracker
from src.history import HistoryLog


# Slash commands with rich local interactive handlers in ClawdREPL.
# These bypass the shared command surface (which would report them as
# unknown) and fall through to handle_command's elif chain below.
_LOCAL_REPL_COMMANDS = frozenset({
    "permissions", "plan", "init", "model", "mcp", "resume", "rewind",
})

# Subset of TERMINAL_ONLY_COMMANDS that _handle_terminal_only actually
# implements. The rest (e.g. /model, /mcp, /plan, /permissions, /resume,
# /rewind) are terminal-only for the *web app* -- in the REPL they must fall
# through to the elif chain below, which implements their interactive
# pickers. Routing them to _handle_terminal_only would silently no-op.
_TERMINAL_ONLY_HANDLED = frozenset({
    "exit", "quit", "q", "multiline", "stream", "render-last", "load",
})


class ClawdREPL:
    """Interactive REPL for Clawd Codex."""

    def __init__(self, provider_name: str = "glm", stream: bool = True, auto_compact: bool | None = None):
        self.console = Console()
        self.provider_name = provider_name
        self.stream = stream
        self.multiline_mode = False
        # Auto-compact escape hatch: explicit CLI flag wins, then the
        # "auto_compact" config option, defaulting to enabled.
        if auto_compact is None:
            try:
                from src.config import load_config
                auto_compact = bool(load_config().get("auto_compact", True))
            except Exception:
                auto_compact = True
        self.auto_compact = auto_compact

        # Load configuration
        config = get_provider_config(provider_name)
        if not config.get("api_key"):
            self.console.print("[red]Error: API key not configured.[/red]")
            self.console.print("Run [bold]python -m src.cli login[/bold] to configure.")
            sys.exit(1)

        # Initialize provider
        provider_class = get_provider_class(provider_name)
        self.provider = provider_class(
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            model=config.get("default_model")
        )

        # Create session
        self.session = Session.create(
            provider_name,
            self.provider.model
        )

        self.tool_registry = build_default_registry()
        self.tool_context = ToolContext(workspace_root=Path.cwd())
        self.tool_context.session_id = self.session.session_id
        self.tool_context.enable_powershell = powershell_opt_in()
        self.tool_context.ask_user = self._ask_user_questions
        # Permission handler with status control for proper input handling
        self._current_status = None
        self.tool_context.permission_handler = self._handle_permission_request

        # Original built-in commands - define this FIRST!
        self._original_built_ins = [
            "/",
            "/help",
            "/exit",
            "/quit",
            "/q",
            "/clear",
            "/save",
            "/load",
            "/multiline",
            "/stream",
            "/render-last",
            "/tools",
            "/tool",
            "/skills",
            "/permissions",
            "/rewind",
            "/plan",
            "/init",
            "/model",
            "/mcp",
            "/resume",
        ]
        self._built_in_commands = list(self._original_built_ins)

        # Initialize new command system
        self._init_command_system()

        # Prompt toolkit with tab completion
        history_file = Path.home() / ".clawd" / "history"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        self.completer = WordCompleter(self._get_slash_command_words(), ignore_case=True)

        # Key bindings for multiline
        self.bindings = KeyBindings()
        if hasattr(self.bindings, "add"):
            # Bare "/" on an empty prompt opens the command palette.
            # Use a filter (not an in-handler check): when the filter doesn't
            # match, the key falls through to normal insertion, so slashes in
            # the middle of input (file paths!) are never eaten. The old
            # in-handler version swallowed every "/" it matched, breaking
            # paths like /tmp/foo on piped input (and any second slash
            # typed interactively).
            def _slash_palette_active() -> bool:
                if not sys.stdin.isatty():
                    return False
                try:
                    from prompt_toolkit.application.current import get_app
                    return get_app().current_buffer.text == ""
                except Exception:
                    return False

            try:
                from prompt_toolkit.filters import Condition as _Condition
                _slash_filter = _Condition(_slash_palette_active)
            except Exception:
                _slash_filter = None

            if _slash_filter is not None:
                @self.bindings.add("/", filter=_slash_filter)  # type: ignore[attr-defined]
                def _show_slash_completions(event):  # type: ignore[no-untyped-def]
                    buf = event.current_buffer
                    buf.insert_text("/")
                    buf.start_completion(select_first=False)

        try:
            self.prompt_session = PromptSession(
                history=FileHistory(str(history_file)),
                auto_suggest=AutoSuggestFromHistory(),
                completer=self.completer,
                style=Style.from_dict({
                    'prompt': 'bold blue',
                }),
                key_bindings=self.bindings,
                complete_while_typing=True,
            )
        except Exception:
            # No console (e.g. SSH session without a TTY): fall back to a
            # dummy that raises EOFError on prompt, like the import fallback.
            class _DummySession:
                def prompt(self, *args, **kwargs):
                    raise EOFError()
            self.prompt_session = _DummySession()

    def _ask_user_questions(self, questions: list[dict]) -> dict[str, str]:
        # Stop the Rich status spinner if running, so we can get clean input
        if self._current_status is not None:
            try:
                self._current_status.stop()
            except Exception:
                pass

        answers: dict[str, str] = {}
        for q in questions:
            question_text = str(q.get("question", "")).strip()
            options = q.get("options") or []
            multi = bool(q.get("multiSelect", False))
            if not question_text or not isinstance(options, list) or len(options) < 2:
                continue

            self.console.print(f"\n[bold]{question_text}[/bold]")
            labels: list[str] = []
            for i, opt in enumerate(options, start=1):
                label = str((opt or {}).get("label", "")).strip()
                desc = str((opt or {}).get("description", "")).strip()
                labels.append(label)
                self.console.print(f"  {i}. {label}  [dim]{desc}[/dim]")
            other_idx = len(labels) + 1
            self.console.print(f"  {other_idx}. Other  [dim]Provide custom text[/dim]")

            prompt = "Select (comma-separated) > " if multi else "Select > "
            raw = input(prompt).strip()
            if not raw:
                choice_str = "1"
            else:
                choice_str = raw

            selected: list[str] = []
            parts = [p.strip() for p in choice_str.split(",") if p.strip()]
            if not parts:
                parts = ["1"]
            for part in parts:
                try:
                    idx = int(part)
                except ValueError:
                    idx = -1
                if idx == other_idx:
                    free = input("Other > ").strip()
                    if free:
                        selected.append(free)
                    continue
                if 1 <= idx <= len(labels):
                    selected.append(labels[idx - 1])
            if not selected:
                selected = [labels[0]]
            answers[question_text] = ", ".join(selected) if multi else selected[0]

        # Restart spinner after getting answers
        if self._current_status is not None:
            try:
                self._current_status.start()
            except Exception:
                pass

        return answers

    def _handle_permission_request(
        self,
        tool_name: str,
        message: str,
        suggestion: str | None,
    ) -> tuple[bool, bool]:
        """Handle interactive permission requests from tools.

        Args:
            tool_name: Name of the tool requesting permission.
            message: Message explaining what permission is needed.
            suggestion: Optional suggestion for enabling the setting.

        Returns:
            Tuple of (allowed: bool, continue_without_caching: bool).
            continue_without_caching is always False since we don't cache in REPL.
        """
        # Stop the Rich status spinner if running, so we can get clean input
        if self._current_status is not None:
            try:
                self._current_status.stop()
            except Exception:
                pass

        # Plan approval gets its own UI (never "always allow" a plan review)
        if tool_name == "ExitPlanMode" and suggestion == "exit-plan":
            return self._handle_plan_approval(message)

        # "Always allow" memory: skip the prompt entirely for remembered tools
        from src.tool_system.permission_store import is_always_allowed, allow_always

        if is_always_allowed(tool_name):
            return True, False

        # Determine if this is a setting that can be enabled
        can_enable_setting = False
        setting_to_enable: str | None = None

        msg_lower = message.lower()
        if "allow_docs" in msg_lower or "documentation files" in msg_lower:
            pc = self.tool_context.permission_context
            if hasattr(pc, 'allow_docs') and not pc.allow_docs:
                can_enable_setting = True
                setting_to_enable = "allow_docs"

        # Build the prompt as a panel: tool, what it wants, and the choices
        options: list[tuple[str, str]] = [
            ("1", "Yes, allow once  [dim](y)[/dim]"),
            ("2", f"Always allow {tool_name}  [dim](a)[/dim]"),
            ("3", "No, deny  [dim](n)[/dim]"),
        ]
        if can_enable_setting:
            options.append(("e", f"Enable {setting_to_enable} and allow  [dim](e)[/dim]"))

        body_lines = [
            f"[bold cyan]{tool_name}[/bold cyan] wants permission",
            "",
            f"  {message}",
            "",
        ]
        for num, desc in options:
            body_lines.append(f"  [bold]{num}.[/bold] {desc}")
        body = "\n".join(body_lines)

        self.console.print("")
        self.console.print(
            Panel(
                body,
                title="[bold yellow]\u26a0 Permission Required[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

        # Get input - use standard input() which works after stopping status
        choice = input("Select option> ").strip().lower()

        if choice in ("1", "y", "yes", ""):
            return True, False
        elif choice in ("2", "a", "always"):
            allow_always(tool_name)
            self.console.print(
                f"[dim]\u2713 {tool_name} will be allowed automatically "
                f"(manage with /permissions)[/dim]"
            )
            return True, False
        elif choice in ("3", "n", "no"):
            return False, False
        elif choice in ("e", "enable") and can_enable_setting:
            self._enable_permission_setting(setting_to_enable)
            return True, False

        # Default to deny for invalid input
        self.console.print("[dim]Invalid choice, defaulting to deny.[/dim]")
        return False, False

    def _handle_plan_approval(self, message: str) -> tuple[bool, bool]:
        """Review UI for a plan proposed via ExitPlanMode.

        Shows the plan as markdown in a panel. Approve -> the tool runs and
        plan mode ends. Deny -> the tool is blocked and planning continues.
        """
        plan = message[len("PLAN:\n"):] if message.startswith("PLAN:\n") else None

        if plan:
            body = Markdown(plan)
        else:
            body = Text("No plan written.", style="dim")

        self.console.print("")
        self.console.print(
            Panel(
                body,
                title="[bold cyan]\U0001f4cb Plan Proposed[/bold cyan]",
                subtitle="[dim]review before the agent acts[/dim]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        self.console.print("  [bold]1.[/bold] Approve plan & exit plan mode  [dim](y)[/dim]")
        self.console.print("  [bold]2.[/bold] Keep planning  [dim](n)[/dim]")
        self.console.print("")

        choice = input("Select option> ").strip().lower()
        if choice in ("1", "y", "yes", ""):
            self.console.print("[dim]\u2713 Plan approved.[/dim]")
            return True, False
        self.console.print("[dim]Staying in plan mode.[/dim]")
        return False, False

    def _enable_permission_setting(self, setting_name: str | None) -> None:
        """Enable a permission setting in the tool context."""
        if not setting_name:
            return

        self.console.print(f"\n[dim]Enabling {setting_name}...[/dim]")

        if setting_name == "allow_docs":
            pc = self.tool_context.permission_context
            if hasattr(pc, 'allow_docs'):
                pc.allow_docs = True
                self.console.print(f"[green]✓ {setting_name} enabled for this session[/green]")
                return

        self.console.print(f"[dim]Could not enable {setting_name}.[/dim]")

    def _init_command_system(self):
        """Initialize the new command system."""
        # Also register to global registry so execute_command_async can find commands
        register_builtin_commands(None)  # None = use global registry

        # Create command registry and register built-ins
        self.command_registry = CommandRegistry()
        register_builtin_commands(self.command_registry)

        # Create cost tracker and history
        self.cost_tracker = CostTracker()
        self.history_log = HistoryLog()

        # Create command context
        self.command_context = create_command_context(
            workspace_root=Path.cwd(),
            conversation=self.session.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history_log,
        )

        # Merge new commands with built-in list for completion
        self._update_built_in_commands_with_command_system()

    def _update_built_in_commands_with_command_system(self):
        """Update the built-in commands list with commands from the new system."""
        # Start with original built-ins
        self._built_in_commands = list(self._original_built_ins)

        # Add commands from the new command system
        try:
            for cmd in self.command_registry.list_commands():
                cmd_name = f"/{cmd.name}"
                if cmd_name not in self._built_in_commands:
                    self._built_in_commands.append(cmd_name)
                # Add aliases
                for alias in cmd.aliases:
                    alias_name = f"/{alias}"
                    if alias_name not in self._built_in_commands:
                        self._built_in_commands.append(alias_name)
        except Exception:
            pass

    def _try_execute_new_command(self, command: str, args: str) -> tuple[bool, str | None]:
        """Try to execute a command using the new command system (sync path for LocalCommand only).

        Returns:
            Tuple of (handled: bool, result_text: str | None)
        """
        try:
            success, result_text, error = execute_command_sync(
                command, args, self.command_context
            )
            if success:
                return True, result_text
            else:
                return False, error
        except Exception as e:
            return False, str(e)

    async def _try_execute_command_async(self, command: str, args: str) -> CommandResult:
        """Execute a command asynchronously, supporting both LocalCommand and PromptCommand.

        Returns:
            CommandResult with the execution result
        """
        try:
            return await execute_command_async(command, args, self.command_context)
        except Exception as e:
            return CommandResult.error(command, str(e))

    def _handle_command_result(self, result: CommandResult) -> bool:
        """Handle the result of a command execution.

        Returns True if the command was handled, False otherwise.
        """
        if not result.success:
            if result.error:
                self.console.print(f"[red]{result.error}[/red]")
            return True

        if result.result_type == "text":
            if result.text:
                self.console.print("\n" + result.text)
                self.console.print()
            return True

        elif result.result_type == "prompt":
            # For PromptCommand, extract the text content and send to LLM
            prompt_text = ""
            for item in result.prompt_content:
                if item.get("type") == "text":
                    prompt_text = item.get("text", "")
                    break

            if prompt_text:
                # Send the prompt to the LLM for interactive execution
                # Use higher max_turns for complex commands like /init
                self.console.print("[dim]Initializing workspace setup...[/dim]")
                self.chat(prompt_text, max_turns=100)
            return True

        elif result.result_type == "skip":
            # Command handled silently
            return True

        return False

    def _get_slash_command_words(self) -> list[str]:
        """Use the shared command surface so CLI and web always match."""
        cwd = self.tool_context.cwd or self.tool_context.workspace_root
        try:
            infos = build_command_list(project_root=cwd)
            words = [f"/{info.name}" for info in infos]
        except Exception:
            # Fallback to hard-coded list if the surface fails
            words = list(self._built_in_commands)
            try:
                from src.skills.loader import get_all_skills

                for s in get_all_skills(project_root=cwd):
                    words.append(f"/{s.name}")
            except Exception:
                pass
        # Local interactive commands bypass the surface but still need
        # tab-completion entries.
        for name in sorted(_LOCAL_REPL_COMMANDS):
            w = f"/{name}"
            if w.lower() not in (x.lower() for x in words):
                words.append(w)
        deduped: list[str] = []
        seen: set[str] = set()
        for w in words:
            lw = w.lower()
            if lw in seen:
                continue
            seen.add(lw)
            deduped.append(w)
        return deduped

    def _refresh_completer(self) -> None:
        try:
            words = self._get_slash_command_words()
            try:
                base = WordCompleter(words, ignore_case=True, match_middle=True)
            except TypeError:
                base = WordCompleter(words, ignore_case=True)
            self.completer = FuzzyCompleter(base) if FuzzyCompleter is not None else base
            if hasattr(self, "prompt_session") and getattr(self.prompt_session, "completer", None) is not None:
                self.prompt_session.completer = self.completer
        except Exception:
            return

    def _open_command_center(self, query: str = "") -> None:
        """Full-screen command picker; printed palette when stdin is not a TTY."""
        import sys

        if not sys.stdin.isatty():
            self._show_slash_palette(query or None)
            return
        try:
            from src.repl.command_center import collect_palette_items, run_command_center

            selected = run_command_center(collect_palette_items(self), initial_query=query)
        except Exception as e:
            self.console.print(f"[dim]Command center unavailable ({e}); showing list.[/dim]")
            self._show_slash_palette(query or None)
            return
        if selected:
            self.handle_command(selected)

    def _show_slash_palette(self, query: str | None = None) -> None:
        q = (query or "").strip().lower()
        self.console.print("\n[bold]Available commands and skills:[/bold]")

        # Collect all commands
        all_commands: list[tuple[str, str, str]] = []  # (name, description, type)
        seen: set[str] = set()

        def add_command(name: str, desc: str, cmd_type: str = "command") -> None:
            if name in seen:
                return
            seen.add(name)
            if q and q not in name.lower() and q not in desc.lower():
                return
            all_commands.append((name, desc, cmd_type))

        # Add built-in commands
        for cmd in self._original_built_ins:
            if cmd == "/":
                continue
            add_command(cmd, "", "command")

        # Add commands from new command system
        try:
            for cmd in self.command_registry.list_commands():
                cmd_name = f"/{cmd.name}"
                if cmd_name in self._original_built_ins:
                    continue
                alias_str = f" (aliases: {', '.join(cmd.aliases)})" if cmd.aliases else ""
                add_command(f"{cmd_name}{alias_str}", cmd.description, "command")
        except Exception:
            pass

        # Add skills
        try:
            from src.skills.loader import get_all_skills

            cwd = self.tool_context.cwd or self.tool_context.workspace_root
            skills = list(get_all_skills(project_root=cwd))
            skills.sort(key=lambda s: s.name.lower())
            for s in skills:
                desc = (s.description or "").strip()
                add_command(f"/{s.name}", desc, "skill")
        except Exception:
            pass

        # Sort and display
        all_commands.sort(key=lambda x: x[0].lower())
        for name, desc, cmd_type in all_commands:
            if cmd_type == "skill":
                self.console.print(f"  [magenta]{name}[/magenta]")
                if desc:
                    self.console.print(f"    [dim]{desc}[/dim]")
            else:
                if desc:
                    self.console.print(f"  {name}  [dim]- {desc}[/dim]")
                else:
                    self.console.print(f"  {name}")

        self.console.print()

    def _shorten_path_text(self, text: str) -> str:
        root = str(self.tool_context.workspace_root)
        cwd = str(self.tool_context.cwd or self.tool_context.workspace_root)
        # os.sep instead of a hardcoded "/": identical on POSIX (os.sep == "/")
        # and matches backslash paths on Windows.
        for base in (cwd, root):
            prefix = base.rstrip(os.sep) + os.sep
            if text.startswith(prefix):
                return "." + os.sep + text[len(prefix):]
            text = text.replace(prefix, "")
        return text

    def _thinking_status(self):  # type: ignore[no-untyped-def]
        """Live "Thinking..." spinner — but only on a real terminal.

        Rich's console.status redraws one line in place on a TTY. When stdout
        is piped (scripting, logs, captures) it can't reposition the cursor,
        so every animation frame prints as a new line. Return a no-op context
        manager in that case for clean output.
        """
        if self.console.is_terminal:
            return self.console.status("[dim]Thinking...[/dim]", spinner="dots")
        return nullcontext()

    def _display_cwd(self) -> str:
        cwd = str(Path.cwd())
        home = str(Path.home())
        if cwd.startswith(home):
            return cwd.replace(home, "~", 1)
        return cwd

    def _truncate_middle(self, text: str, limit: int) -> str:
        if limit <= 0 or len(text) <= limit:
            return text
        if limit <= 3:
            return text[:limit]
        head = max(1, (limit - 1) // 2)
        tail = max(1, limit - head - 1)
        return f"{text[:head]}…{text[-tail:]}"

    def _print_startup_header(self):
        from src import __version__

        display_path = self._display_cwd()
        provider_label = f"{self.provider_name.upper()} Provider"
        model_label = self.provider.model or "Unknown model"

        mascot_ascii = "\n".join([
            "  /\\__/\\",
            " / o  o \\",
            "(  __  )",
            " \\/__/  ",
        ])

        if Panel is None or Group is None or Align is None or Table is None or Text is None or Columns is None:
            print(mascot_ascii)
            print(f"Clawd Codex v{__version__}")
            print(f"{model_label} · {provider_label}")
            print(f"{display_path}\n")
            return

        width = getattr(self.console, "width", 80)
        content_width = max(28, min(width - 12, 72))
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bright_black", justify="right", no_wrap=True)
        table.add_column(style="white", ratio=1)
        table.add_row("Version", Text.assemble(("Clawd Codex", "bold white"), ("  ", ""), (f"v{__version__}", "bold cyan")))
        table.add_row("Model", Text(model_label, style="bold magenta"))
        table.add_row("Provider", Text(provider_label, style="bold green"))
        table.add_row("Workspace", Text(self._truncate_middle(display_path, content_width - 12), style="bold blue"))

        footer = Text("/help  •  /tools  •  /stream  •  /render-last  •  /exit", style="dim")
        mascot_block = Text(mascot_ascii, style="bold orange3", no_wrap=True)
        body = Group(
            Columns([mascot_block, table], align="center", expand=False),
            Text(""),
            Align.center(footer),
        )
        header = Panel(
            body,
            border_style="bright_black",
            title="[bold bright_cyan] CLAWD CODE [/bold bright_cyan]",
            subtitle="[dim]interactive terminal[/dim]",
            padding=(1, 2),
        )
        self.console.print(header)
        self.console.print()

    def run(self):
        """Run the REPL."""
        self._print_startup_header()

        while True:
            try:
                self._refresh_completer()
                # Dynamic prompt based on multiline mode and plan mode
                # Using '❯' for a modern feel
                if self.multiline_mode:
                    prompt_text = '... '
                elif getattr(self.tool_context, "plan_mode", False):
                    prompt_text = '[plan] ❯ '
                else:
                    prompt_text = '❯ '
                from src.repl.status_bar import render_status
                self.console.print(f"[dim]{render_status(self)}[/dim]")
                user_input = self.prompt_session.prompt(
                    prompt_text,
                    multiline=self.multiline_mode
                )

                if not user_input.strip():
                    self.multiline_mode = False
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    self.handle_command(user_input)
                    continue

                # Send to LLM
                expanded = self._expand_skill_mentions(user_input)
                if expanded is None:
                    # Unknown $skill mention: the error was already printed;
                    # skip this turn rather than sending it unexpanded.
                    self.multiline_mode = False
                    continue
                self.chat(expanded)
                self.multiline_mode = False

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Type /exit to quit.[/yellow]")
                self.multiline_mode = False
                continue
            except EOFError:
                self.console.print("\n[blue]Goodbye![/blue]")
                break

    def handle_command(self, command: str):
        """Handle slash commands using the shared command surface."""
        raw = command.strip()
        if raw == "/":
            self._open_command_center()
            return
        if raw == "/skills":
            self._open_command_center()
            return
        if raw.startswith("/") and " " not in raw and raw.lower() not in (c.lower() for c in self._built_in_commands):
            query = raw[1:]
            if query:
                self._open_command_center(query=query)
                return

        # First, try the shared command surface
        if raw.startswith("/"):
            parsed = parse_slash(raw)
            if not parsed:
                return
            cmd_name, args = parsed
            lowered = cmd_name.lower()

            # Terminal-only commands with a dedicated local handler:
            # handle locally. Others in TERMINAL_ONLY_COMMANDS (e.g.
            # /model, /mcp) are terminal-only for the web app; in the
            # REPL they fall through to the elif chain below.
            if is_terminal_only(cmd_name) and lowered in _TERMINAL_ONLY_HANDLED:
                self._handle_terminal_only(cmd_name, args)
                return

            # Local interactive commands bypass the shared surface and fall
            # through to the elif chain below.
            if lowered not in _LOCAL_REPL_COMMANDS:
                cwd = self.tool_context.cwd or self.tool_context.workspace_root
                try:
                    result = asyncio.run(
                        run_command(
                            cmd_name,
                            args,
                            self.command_context,
                            project_root=cwd,
                            tool_registry=self.tool_registry,
                            tool_context=self.tool_context,
                            session=self.session,
                        )
                    )
                except Exception as e:
                    self.console.print(f"[red]Error executing /{cmd_name}: {e}[/red]")
                    return

                if not result.success:
                    self.console.print(f"[red]{result.error}[/red]")
                    return

                if result.result_type == "skip":
                    return

                if result.result_type == "prompt":
                    # PromptCommand (or skill) — feed expanded text to the model
                    expanded = flatten_prompt_content(result.prompt_content)
                    if expanded:
                        self.chat(expanded, max_turns=20)
                    return

                # LocalCommand — result.text is the answer
                if result.text:
                    self.console.print("\n" + result.text)
                self.console.print()
                return

        # Fall back to original command handling for terminal-specific features
        cmd = raw.lower()

        if cmd in ['/exit', '/quit', '/q']:
            self.console.print("[blue]Goodbye![/blue]")
            sys.exit(0)

        elif cmd == '/help':
            self.show_help()

        elif cmd == '/multiline':
            self.multiline_mode = not self.multiline_mode
            status = "enabled" if self.multiline_mode else "disabled"
            self.console.print(f"[green]Multiline mode {status}.[/green]")
            if self.multiline_mode:
                self.console.print("[dim]Press Meta+Enter or Esc+Enter to submit.[/dim]")

        elif cmd == '/stream' or cmd.startswith('/stream '):
            parts = raw.split(maxsplit=1)
            if len(parts) == 1:
                status = "enabled" if self.stream else "disabled"
                self.console.print(f"[green]Stream mode {status}.[/green]")
                return

            action = parts[1].strip().lower()
            if action in {"on", "true", "1", "enable", "enabled"}:
                self.stream = True
            elif action in {"off", "false", "0", "disable", "disabled"}:
                self.stream = False
            elif action == "toggle":
                self.stream = not self.stream
            else:
                self.console.print("[red]Usage: /stream [on|off|toggle][/red]")
                return

            status = "enabled" if self.stream else "disabled"
            self.console.print(f"[green]Stream mode {status}.[/green]")

        elif cmd == '/render-last':
            rendered = self._render_last_assistant_message()
            if not rendered:
                self.console.print("[yellow]No assistant response available to render.[/yellow]")

        elif cmd.startswith('/load'):
            parts = command.strip().split(maxsplit=1)
            if len(parts) < 2:
                self.console.print("[red]Usage: /load <session-id>[/red]")
            else:
                session_id = parts[1]
                self.load_session(session_id)

        elif cmd == '/skill':
            self._handle_skill_command()

        elif cmd == '/permissions' or cmd.startswith('/permissions '):
            self._handle_permissions_command(command)

        elif cmd == '/plan' or cmd.startswith('/plan '):
            self._handle_plan_command(command)

        elif cmd == '/init':
            self._handle_init_command(command)

        elif cmd == '/model' or cmd.startswith('/model '):
            self._handle_model_command(command)

        elif cmd == '/mcp' or cmd.startswith('/mcp '):
            self._handle_mcp_command(command)

        elif cmd == '/resume' or cmd.startswith('/resume '):
            self._handle_resume_command(command)

        elif cmd == '/rewind' or cmd.startswith('/rewind '):
            self._handle_rewind_command(command)

        elif cmd == '/context':
            # Populate command context config for context analysis
            self.command_context.config["provider"] = self.provider
            self.command_context.config["model"] = self.provider.model
            self.command_context.config["tool_schemas"] = [
                spec.to_dict() if hasattr(spec, "to_dict") else {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": dict(spec.input_schema) if hasattr(spec.input_schema, "keys") else spec.input_schema,
                }
                for spec in self.tool_registry.list_specs()
            ]
            self.command_context.config["system_prompt"] = ""
            # Try new command system
            try:
                handled, result_text = self._try_execute_new_command('context', '')
                if handled and result_text:
                    self.console.print(Markdown(result_text))
                    return
            except Exception:
                pass
            self.console.print("[yellow]/context analysis unavailable in this context.[/yellow]")

        elif cmd == '/compact' or cmd.startswith('/compact '):
            compact_args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
            # --hard keeps the legacy clear-everything behavior
            if "--hard" in compact_args.split():
                self.session.conversation.clear()
                self.console.print("[green]Conversation cleared (--hard).[/green]")
                return
            # Populate command context config for compact
            self.command_context.config["provider"] = self.provider
            self.command_context.config["model"] = self.provider.model
            self.command_context.config["system_prompt"] = ""
            # Try new command system (real summarization via compact_conversation)
            try:
                handled, result_text = self._try_execute_new_command('compact', compact_args)
                if handled and result_text:
                    self.console.print("\n[green]" + result_text + "[/green]")
                    return
            except Exception:
                pass
            # Fallback: real summarization via the shared compact path (never nukes)
            try:
                from src.command_system.builtins import compact_command_call
                result = compact_command_call(compact_args, self.command_context)
                self.console.print("\n[green]" + (result.value or "Conversation compacted.") + "[/green]")
            except Exception as e:
                self.console.print(f"[red]Compact failed: {e}[/red]")

        else:
            if raw.startswith("/"):
                if self._try_run_skill_slash(raw):
                    return
            self.console.print(f"[red]Unknown command: {command}[/red]")

    def _handle_terminal_only(self, name: str, args: str) -> None:
        """Handle commands that only exist in the terminal (no web equivalent)."""
        if name in ("exit", "quit", "q"):
            self.console.print("[blue]Goodbye![/blue]")
            _fire_lifecycle_hook("SessionEnd", {"session_id": getattr(self.session, "session_id", None), "reason": "exit"})
            sys.exit(0)
        elif name in ("multiline",):
            self.multiline_mode = not self.multiline_mode
            status = "enabled" if self.multiline_mode else "disabled"
            self.console.print(f"[green]Multiline mode {status}.[/green]")
            if self.multiline_mode:
                self.console.print("[dim]Press Meta+Enter or Esc+Enter to submit.[/dim]")
        elif name in ("stream",):
            parts = args.split(maxsplit=1) if args else []
            if not parts:
                status = "enabled" if self.stream else "disabled"
                self.console.print(f"[green]Stream mode {status}.[/green]")
                return
            action = parts[0].strip().lower()
            if action in {"on", "true", "1", "enable", "enabled"}:
                self.stream = True
            elif action in {"off", "false", "0", "disable", "disabled"}:
                self.stream = False
            elif action == "toggle":
                self.stream = not self.stream
            else:
                self.console.print("[red]Usage: /stream [on|off|toggle][/red]")
                return
            status = "enabled" if self.stream else "disabled"
            self.console.print(f"[green]Stream mode {status}.[/green]")
        elif name in ("render-last",):
            rendered = self._render_last_assistant_message()
            if not rendered:
                self.console.print("[yellow]No assistant response available to render.[/yellow]")
        elif name in ("load",):
            # /load <session-id> - load a saved session
            parts = args.split()
            if not parts:
                self.console.print("[red]Usage: /load <session-id>[/red]")
                return
            self.load_session(parts[0])

    def _try_run_skill_slash(self, raw: str) -> bool:
        """Use the shared command surface to run a skill."""
        parsed = parse_slash(raw)
        if not parsed:
            return False
        name, args = parsed
        if name.lower() in {c.lstrip("/").lower() for c in self._built_in_commands if c != "/"}:
            return False

        try:
            result = run_skill_tool(name, args, self.tool_registry, self.tool_context)
        except Exception as e:
            self.console.print(f"[red]Skill error: {e}[/red]")
            return True

        payload = result.output if isinstance(result.output, dict) else {}
        if result.is_error or not payload.get("success"):
            err = payload.get("error") if isinstance(payload.get("error"), str) else "Unknown skill error"
            self.console.print(f"[red]{err}[/red]")
            return True

        self.console.print(f"[dim]Launching skill: {payload.get('commandName', name)}[/dim]")
        meta_parts: list[str] = []
        loaded = payload.get("loadedFrom")
        if isinstance(loaded, str) and loaded:
            meta_parts.append(f"source={loaded}")
        model = payload.get("model")
        if isinstance(model, str) and model:
            meta_parts.append(f"model={model}")
        tools = payload.get("allowedTools")
        if isinstance(tools, list) and tools:
            shown = ", ".join(str(t) for t in tools[:6])
            more = f" (+{len(tools) - 6})" if len(tools) > 6 else ""
            meta_parts.append(f"tools={shown}{more}")
        if meta_parts:
            self.console.print(f"[dim]{' · '.join(meta_parts)}[/dim]")

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self.console.print("[red]Skill produced empty prompt[/red]")
            return True

        self.chat(prompt)
        return True

    def show_help(self):
        """Show help message."""
        help_text = """
**Available Commands:**

- `/` - Show all commands and skills
- `/help` - Show this help message
- `/exit`, `/quit`, `/q` - Exit the REPL
- `/clear`, `/reset`, `/new` - Clear conversation history
- `/save` - Save current session
- `/load <session-id>` - Load a previous session
- `/multiline` - Toggle multiline input mode
- `/stream [on|off|toggle]` - Toggle live response rendering
- `/render-last` - Re-render the last assistant reply as Markdown
- `/tools` - List available built-in tools
- `/tool <name> <json>` - Run a tool directly
- `/skills` - List all available skills
- `/permissions` - List always-allow rules and set the session permission mode (`/permissions forget <Tool>`, `/permissions clear`, `/permissions mode <default|acceptEdits|dontAsk|bypassPermissions>`)
- `/rewind` - List file checkpoints; `/rewind <#>` restores one
- `/plan` - Enter plan mode (explore only); `/plan off` to leave
- `/init` - Create CLAUDE.md file for the project
- `/model` - Switch provider/model mid-session
- `/mcp` - List MCP servers (`/mcp add <name> <command> [args...]`)
- `/resume` - Pick a saved session to resume
- `/cost` - Show session cost and usage
- `/compact` - Compact conversation to save context space

**Usage:**
- Type your message and press Enter to chat
- Use Tab for command completion
- Press Ctrl+C to interrupt current operation
- Press Ctrl+D to exit
- Use `/multiline` for multi-paragraph inputs
"""
        self.console.print(Markdown(help_text))

    def _handle_skill_command(self) -> None:
        """Handle /skill command - list all available skills."""
        try:
            from src.skills.loader import get_all_skills

            cwd = self.tool_context.cwd or self.tool_context.workspace_root
            skills = list(get_all_skills(project_root=cwd))
            skills.sort(key=lambda s: s.name.lower())

            if not skills:
                self.console.print("\n[bold]Available Skills:[/bold]")
                self.console.print("[dim]No skills found.[/dim]")
                self.console.print("[dim]Create skills in ~/.clawd/skills/ or ~/.claude/skills/ or .clawd/skills/ in your project.[/dim]")
                return

            # Group skills by source
            from collections import defaultdict
            by_source: dict[str, list] = defaultdict(list)
            for s in skills:
                loaded = getattr(s, "loaded_from", "") or "unknown"
                by_source[loaded].append(s)

            self.console.print(f"\n[bold]Available Skills ({len(skills)}):[/bold]")
            for source in sorted(by_source.keys()):
                source_skills = by_source[source]
                self.console.print(f"\n[cyan]{source.title()} Skills:[/cyan]")
                for s in source_skills:
                    desc = (getattr(s, "description", None) or "").strip()
                    user_invocable = getattr(s, "user_invocable", True)
                    inv_str = "" if user_invocable else " [dim](not user-invocable)[/dim]"
                    self.console.print(f"  [green]/{s.name}[/green]{inv_str}")
                    if desc:
                        self.console.print(f"    [dim]{desc}[/dim]")
            self.console.print()
        except Exception as e:
            self.console.print(f"[red]Error loading skills: {e}[/red]")

    def _handle_permissions_command(self, raw: str) -> None:
        """Manage 'always allow' permission rules and the session permission mode.

        /permissions            list remembered rules
        /permissions clear      forget all rules
        /permissions forget <Tool>  forget one rule
        /permissions mode       show the session permission mode
        /permissions mode <default|acceptEdits|dontAsk|bypassPermissions>
                                set the session permission mode
        """
        from src.tool_system.permission_store import (
            list_always_allowed,
            clear_rules,
            forget_tool,
        )

        parts = raw.strip().split()
        action = parts[1].lower() if len(parts) > 1 else "list"

        if action == "clear":
            clear_rules()
            self.console.print("[green]Permission rules cleared.[/green]")
            return

        if action == "forget":
            if len(parts) < 3:
                self.console.print("[red]Usage: /permissions forget <ToolName>[/red]")
                return
            name = parts[2]
            if forget_tool(name):
                self.console.print(f"[green]Forgot always-allow rule for {name}.[/green]")
            else:
                self.console.print(f"[yellow]No always-allow rule for {name}.[/yellow]")
            return

        if action == "mode":
            from src.tool_system.permissions import PermissionMode

            if len(parts) == 2:
                mode = getattr(self.tool_context, "permission_mode", PermissionMode.DEFAULT)
                self.console.print(f"[bold]Permission mode:[/bold] [cyan]{mode.value}[/cyan]")
                return
            if len(parts) >= 3:
                try:
                    new_mode = PermissionMode.from_string(parts[2])
                except ValueError:
                    self.console.print(
                        "[red]Usage: /permissions mode "
                        "<default|acceptEdits|dontAsk|bypassPermissions>[/red]"
                    )
                    return
                self.tool_context.permission_mode = new_mode
                self.console.print(
                    f"[green]Permission mode set to [cyan]{new_mode.value}[/cyan].[/green]"
                )
                return
            self.console.print(
                "[red]Usage: /permissions mode <default|acceptEdits|dontAsk|bypassPermissions>[/red]"
            )
            return

        # default: list
        allowed = list_always_allowed()
        if not allowed:
            self.console.print("\n[bold]Always-allow rules:[/bold] [dim](none)[/dim]")
            self.console.print(
                "[dim]Tools you mark 'Always allow' at a permission prompt appear here.[/dim]\n"
            )
            return
        self.console.print("\n[bold]Always-allow rules:[/bold]")
        for name in allowed:
            self.console.print(f"  [cyan]{name}[/cyan]  [dim]- allowed without prompting[/dim]")
        self.console.print("[dim]Use /permissions forget <Tool> or /permissions clear.[/dim]\n")

    def _handle_init_command(self, raw: str) -> None:
        """Generate a CLAUDE.md draft from the workspace and write it on confirm.

        /init

        The draft is shown in a panel for review. The user must explicitly
        confirm; an empty answer declines (safe default).
        """
        from src.command_system.builtins import generate_claude_md_draft

        root = Path(self.tool_context.workspace_root or Path.cwd())
        draft = generate_claude_md_draft(root)
        target = root / "CLAUDE.md"

        self.console.print()
        self.console.print(
            Panel(
                Markdown(draft),
                title="[bold cyan]CLAUDE.md draft[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        if target.exists():
            self.console.print(f"[yellow]CLAUDE.md already exists at {target}.[/yellow]")
        try:
            answer = input(f"Write CLAUDE.md to {target}? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer not in ("y", "yes"):
            self.console.print("[dim]Not written.[/dim]\n")
            return
        try:
            target.write_text(draft, encoding="utf-8")
        except OSError as e:
            self.console.print(f"[red]Could not write {target}: {e}[/red]")
            return
        self.console.print(f"[green]Wrote {target}[/green]\n")

    def _handle_model_command(self, raw: str) -> None:
        """Switch provider/model mid-session.

        /model              pick a provider, then a model
        /model <provider>   pick a model for that provider
        """
        from src.command_system.builtins import list_configured_providers

        providers = list_configured_providers()
        if not providers:
            self.console.print("[red]No providers configured.[/red]")
            return

        parts = raw.strip().split()
        preselect = parts[1].lower() if len(parts) > 1 else ""

        provider = None
        if preselect:
            provider = next((p for p in providers if p["name"].lower() == preselect), None)
            if provider is None:
                self.console.print(f"[red]Unknown provider: {parts[1]}[/red]")
                return

        if provider is None:
            self.console.print("\n[bold]Providers:[/bold]")
            for i, p in enumerate(providers, start=1):
                current = " [bold green](current)[/bold green]" if p["is_default"] else ""
                key = "[green]key set[/green]" if p["api_key_configured"] else "[yellow]no key[/yellow]"
                self.console.print(
                    f"  [cyan]{i}.[/cyan] {p['label']} [dim]({p['name']})[/dim]  "
                    f"[dim]{p['default_model']}[/dim]  {key}{current}"
                )
            self.console.print()
            try:
                choice = input("Select provider > ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return
            try:
                idx = int(choice)
            except ValueError:
                self.console.print("[red]Invalid selection.[/red]")
                return
            if not 1 <= idx <= len(providers):
                self.console.print("[red]Invalid selection.[/red]")
                return
            provider = providers[idx - 1]

        if not provider["api_key_configured"]:
            self.console.print(
                f"[red]No API key configured for {provider['name']}.[/red]\n"
                "[dim]Run [bold]python -m src.cli login[/bold] first.[/dim]"
            )
            return

        models = list(provider["available_models"] or [])
        if provider["default_model"] and provider["default_model"] not in models:
            models.insert(0, provider["default_model"])

        if len(models) == 1:
            model = models[0]
        else:
            self.console.print(f"\n[bold]Models for {provider['label']}:[/bold]")
            for i, m in enumerate(models, start=1):
                mark = " [bold green](default)[/bold green]" if m == provider["default_model"] else ""
                self.console.print(f"  [cyan]{i}.[/cyan] {m}{mark}")
            self.console.print()
            try:
                choice = input("Select model > ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return
            if not choice:
                model = provider["default_model"]
            else:
                try:
                    idx = int(choice)
                except ValueError:
                    self.console.print("[red]Invalid selection.[/red]")
                    return
                if not 1 <= idx <= len(models):
                    self.console.print("[red]Invalid selection.[/red]")
                    return
                model = models[idx - 1]

        # Rebuild the provider, mirroring the login flow.
        try:
            from src.config import get_provider_config, set_api_key, set_default_provider
            from src.providers import get_provider_class

            config = get_provider_config(provider["name"])
            provider_class = get_provider_class(provider["name"])
            new_provider = provider_class(
                api_key=config["api_key"],
                base_url=config.get("base_url"),
                model=model,
            )
        except Exception as e:
            self.console.print(f"[red]Could not initialize provider: {e}[/red]")
            return

        self.provider = new_provider
        self.provider_name = provider["name"]
        # Keep session metadata in sync so /save persists the new model.
        try:
            self.session.provider = provider["name"]
            self.session.model = model
        except Exception:
            pass
        try:
            set_api_key(provider["name"], api_key=config["api_key"], default_model=model)
            set_default_provider(provider["name"])
        except Exception:
            pass
        try:
            self.command_context.config["model"] = model
        except Exception:
            pass

        self.console.print(f"\n[green]Switched to {provider['name']} · {model}[/green]")
        from src.repl.status_bar import render_status
        self.console.print(f"[dim]{render_status(self)}[/dim]\n")

    def _handle_mcp_command(self, raw: str) -> None:
        """List configured MCP servers, or add a stdio server.

        /mcp                            list servers
        /mcp add <name> <command> [args...]   add a server (prompts for missing parts)
        """
        from src.command_system.builtins import (
            add_mcp_server,
            get_mcp_config_path,
            list_mcp_servers,
        )

        parts = raw.strip().split()
        action = parts[1].lower() if len(parts) > 1 else "list"

        if action == "add":
            name = parts[2] if len(parts) > 2 else ""
            command = parts[3] if len(parts) > 3 else ""
            extra_args = parts[4:] if len(parts) > 4 else []
            try:
                if not name:
                    name = input("Server name: ").strip()
                if not command:
                    command = input("Command: ").strip()
                if not extra_args:
                    arg_line = input("Args (space-separated, optional): ").strip()
                    extra_args = arg_line.split() if arg_line else []
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return
            try:
                entry = add_mcp_server(name, command, extra_args)
            except ValueError as e:
                self.console.print(f"[red]{e}[/red]")
                return
            arg_str = (" " + " ".join(entry["args"])) if entry["args"] else ""
            self.console.print(
                f"[green]Added MCP server '{entry['name']}': {entry['command']}{arg_str}[/green]"
            )
            self.console.print(f"[dim]Saved to {get_mcp_config_path()}[/dim]\n")
            return

        servers = list_mcp_servers()
        if not servers:
            self.console.print("\n[bold]MCP servers:[/bold] [dim](none)[/dim]")
            self.console.print("[dim]Add one with /mcp add <name> <command> [args...][/dim]\n")
            return
        body = []
        for s in servers:
            arg_str = (" " + " ".join(s["args"])) if s["args"] else ""
            body.append(f"[bold cyan]{s['name']}[/bold cyan]\n  [dim]{s['command']}{arg_str}[/dim]")
        self.console.print()
        self.console.print(
            Panel(
                "\n\n".join(body),
                title=f"[bold cyan]MCP servers ({len(servers)})[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        self.console.print("[dim]Add one with /mcp add <name> <command> [args...][/dim]\n")

    def _handle_resume_command(self, raw: str) -> None:
        """Pick a saved session to resume.

        /resume
        """
        from src.command_system.builtins import list_saved_sessions

        sessions = list_saved_sessions()
        if not sessions:
            self.console.print("[dim]No saved sessions found.[/dim]")
            return

        body = []
        for i, s in enumerate(sessions, start=1):
            ts = (s["updated_at"] or s["created_at"] or "").replace("T", " ")[:19]
            body.append(
                f"[cyan]{i}.[/cyan] [bold]{s['session_id']}[/bold]  [dim]{ts}[/dim]\n"
                f"   [dim]{s['provider']} · {s['model']} · {s['message_count']} messages[/dim]"
            )
        self.console.print()
        self.console.print(
            Panel(
                "\n".join(body),
                title="[bold cyan]Saved sessions[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        try:
            choice = input("Select session > ").strip()
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return
        try:
            idx = int(choice)
        except ValueError:
            self.console.print("[red]Invalid selection.[/red]")
            return
        if not 1 <= idx <= len(sessions):
            self.console.print("[red]Invalid selection.[/red]")
            return
        self.load_session(sessions[idx - 1]["session_id"])

    def _handle_plan_command(self, raw: str) -> None:
        """Toggle plan mode.

        /plan         enter plan mode (explore only, edits blocked)
        /plan off     leave plan mode
        """
        parts = raw.strip().split()
        action = parts[1].lower() if len(parts) > 1 else "on"

        if action in ("off", "exit", "disable"):
            if not self.tool_context.plan_mode:
                self.console.print("[dim]Plan mode is already off.[/dim]")
                return
            self.tool_context.plan_mode = False
            self.console.print("[green]Plan mode off.[/green] [dim]Edits allowed again.[/dim]")
            return

        if self.tool_context.plan_mode:
            self.console.print("[dim]Already in plan mode.[/dim]")
            return
        self.tool_context.plan_mode = True
        self.console.print("")
        self.console.print(
            Panel(
                "The agent can explore but not edit. When it has a plan, "
                "you'll review and approve it before anything changes.",
                title="[bold cyan]\U0001f4cb Plan Mode On[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    def _handle_rewind_command(self, raw: str) -> None:
        """Restore a file to an earlier checkpoint.

        /rewind           list checkpoints for this session
        /rewind <n>       restore checkpoint <n>
        """
        from src.tool_system import checkpoints

        session_id = self.session.session_id
        root = self.tool_context.workspace_root
        parts = raw.strip().split()
        if len(parts) == 1:
            cps = checkpoints.list_checkpoints(session_id, root)
            if not cps:
                self.console.print("[dim]No checkpoints this session yet.[/dim]")
                return
            self.console.print(f"\n[bold]File checkpoints[/bold] [dim](session {session_id})[/dim]")
            for cp in cps:
                note = "created by Write" if cp.get("new_file") else "before Write/Edit"
                self.console.print(
                    f"  [yellow]#{cp['n']:04d}[/yellow]  {cp['relpath']}  "
                    f"[dim]{cp['created'].replace('T', ' ')} - {note}[/dim]"
                )
            self.console.print("[dim]Use /rewind <#> to restore a file to that checkpoint.[/dim]\n")
            return
        try:
            n = int(parts[1])
        except ValueError:
            self.console.print("[red]Usage: /rewind <#>" + "[/red]")
            return
        try:
            rel = checkpoints.restore(session_id, n, root)
            self.console.print(f"[green]Restored {rel} from checkpoint #{n:04d}.[/green]")
        except ValueError as e:
            self.console.print(f"[red]{e}[/red]")

    def _is_recoverable_tool_error(self, tool_name: str, tool_output) -> bool:
        if not isinstance(tool_name, str):
            return False
        if not isinstance(tool_output, dict):
            return False
        name = tool_name.strip().lower()
        err = tool_output.get("error")
        if not isinstance(err, str):
            return False
        e = err.lower()
        if name == "read" and e.startswith("file not found:"):
            p = err.split(":", 2)[-1].strip()
            if "/.clawd/skills/" in p or "\\.clawd\\skills\\" in p or "/.claude/skills/" in p or "\\.claude\\skills\\" in p:
                return True
        return False

    def _provider_uses_system_kwarg(self) -> bool:
        return isinstance(self.provider, (AnthropicProvider, MinimaxProvider))

    def _build_direct_stream_payload(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        style_name = getattr(self.tool_context, "output_style_name", None)
        style_dir = getattr(self.tool_context, "output_style_dir", None)
        style_prompt = resolve_output_style(style_name, style_dir).prompt

        if self._provider_uses_system_kwarg():
            return self.session.conversation.get_messages(), (
                {"system": style_prompt} if style_prompt.strip() else {}
            )

        messages: list[dict[str, Any]] = []
        for msg in self.session.conversation.messages:
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
        if style_prompt.strip():
            messages = [{"role": "system", "content": style_prompt}, *messages]
        return messages, {}

    def _should_try_direct_stream(self, user_input: str) -> bool:
        if not self.stream:
            return False
        text = user_input.strip().lower()
        if not text or text.startswith("/"):
            return False
        if len(text) > 240:
            return False

        code_task_markers = (
            "/", "\\", "src/", "tests/", ".py", ".ts", ".md",
            "file", "files", "read", "write", "edit", "modify", "change",
            "search", "grep", "glob", "bash", "shell", "command", "run",
            "test", "fix", "bug", "refactor", "repo", "repository",
            "project", "workspace", "folder", "directory", "function",
            "class", "module", "code", "implementation", "readme",
            "pyproject", "package.json", "git", "commit", "diff", "tool",
            "文件", "代码", "仓库", "项目", "目录", "读取", "写入", "修改",
            "搜索", "运行", "测试", "修复", "命令", "工具", "函数", "类",
        )
        return not any(marker in text for marker in code_task_markers)

    def _stream_direct_response(self, on_text_chunk=None) -> str | None:
        streamed_chunks: list[str] = []

        try:
            api_messages, call_kwargs = self._build_direct_stream_payload()
            stream_iter = self.provider.chat_stream(api_messages, tools=None, **call_kwargs)
            for chunk in stream_iter:
                if not chunk:
                    continue
                streamed_chunks.append(chunk)
                if on_text_chunk is not None:
                    on_text_chunk(chunk)
        except Exception:
            # Safe fallback: only fall back when nothing has been emitted yet.
            if not streamed_chunks:
                return None
            raise

        if not streamed_chunks:
            return None

        full_response = "".join(streamed_chunks)
        self.session.conversation.add_assistant_message(full_response)
        return full_response

    def _get_last_assistant_text(self) -> str | None:
        for message in reversed(self.session.conversation.messages):
            if message.role != "assistant":
                continue
            content = message.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    block_type = getattr(block, "type", None)
                    if block_type == "text":
                        text = getattr(block, "text", "")
                        if isinstance(text, str) and text:
                            parts.append(text)
                joined = "".join(parts).strip()
                if joined:
                    return joined
        return None

    def _render_last_assistant_message(self) -> bool:
        text = self._get_last_assistant_text()
        if not text:
            return False
        self.console.print("\n[bold]Last Assistant Response[/bold]")
        self.console.print(Markdown(text))
        self.console.print()
        return True

    def _expand_skill_mentions(self, user_input: str) -> str | None:
        """Expand ``$``-skill mentions in user input before the agent loop (009).

        Returns the text to send to the model, or ``None`` when a mention
        could not be resolved (the error is printed and the turn is
        skipped). Input without any mention passes through byte-identical.
        """
        from src.repl.skill_mentions import find_skill_mentions, resolve_skill_mentions

        if not find_skill_mentions(user_input):
            return user_input

        resolution = resolve_skill_mentions(
            user_input,
            tool_registry=self.tool_registry,
            tool_context=self.tool_context,
        )
        if resolution.error:
            self.console.print(f"[red]{resolution.error}[/red]")
            return None

        invoked = ", ".join(f"${name}" for name in resolution.skills)
        self.console.print(f"[dim]Invoking skill: {invoked}[/dim]")
        return resolution.input_text

    def chat(self, user_input: str, max_turns: int | None = None):
        """Send message to LLM and display response.

        Args:
            user_input: The user message to send.
            max_turns: Maximum number of tool call turns (default from config, 50).
        """
        if max_turns is None:
            try:
                from src.config import load_config
                max_turns = int(load_config().get("agent", {}).get("max_turns", 50))
            except Exception:
                max_turns = 50
        # Add user message
        self.session.conversation.add_user_message(user_input)

        try:
            self.console.print("\n[bold]Assistant[/bold]")

            stream_started = False

            def _stop_status_once() -> None:
                nonlocal stream_started
                if self._current_status is not None and not stream_started:
                    try:
                        self._current_status.stop()
                    except Exception:
                        pass
                stream_started = True

            cards = ToolCardRenderer(self.console, shorten=self._shorten_path_text)
            def on_event(ev: ToolEvent) -> None:
                if ev.kind == "tool_use":
                    if is_task_tool(ev.tool_name):
                        self.console.print(cards.render_task_start(
                            ev.tool_name, ev.tool_input or {}, task_id=ev.tool_use_id))
                        return
                    self.console.print(*cards.render_use(ev.tool_name, ev.tool_input or {}))
                    return
                if ev.kind == "tool_result":
                    if is_task_tool(ev.tool_name):
                        self.console.print(cards.render_task_finish(
                            ev.tool_name, ev.tool_output, task_id=ev.tool_use_id,
                            is_error=ev.is_error))
                        return
                    if ev.is_error:
                        if self._is_recoverable_tool_error(ev.tool_name, ev.tool_output):
                            return
                        msg = ""
                        if isinstance(ev.tool_output, dict) and isinstance(ev.tool_output.get("error"), str):
                            msg = ev.tool_output["error"]
                        self.console.print(*cards.render_result(
                            ev.tool_name, {"error": msg or "Error"}, is_error=True))
                        return
                    self.console.print(*cards.render_result(ev.tool_name, ev.tool_output, is_error=False))
                    return
                if ev.kind == "tool_error":
                    self.console.print(*cards.render_error(ev.tool_name, ev.error))

            def on_text_chunk(chunk: str) -> None:
                if not chunk:
                    return
                _stop_status_once()
                self.console.print(chunk, end="", markup=False, highlight=False, soft_wrap=True)

            if self._should_try_direct_stream(user_input):
                self._current_status = self._thinking_status()
                with self._current_status:
                    direct_response = self._stream_direct_response(on_text_chunk=on_text_chunk)
                self._current_status = None
                if direct_response is not None:
                    self.console.print("\n")
                    return

            # Use agent loop with tools for any provider that supports it
            self._current_status = self._thinking_status()
            with self._current_status:
                result = run_agent_loop(
                    conversation=self.session.conversation,
                    provider=self.provider,
                    tool_registry=self.tool_registry,
                    tool_context=self.tool_context,
                    max_turns=max_turns,
                    stream=self.stream,
                    verbose=False,
                    on_event=on_event,
                    on_text_chunk=on_text_chunk if self.stream else None,
                    cost_tracker=self.cost_tracker,
                    auto_compact=self.auto_compact,
                )
            self._current_status = None

            # Per-turn usage (tokens + cost, with provider+model for pricing)
            # is recorded inside run_agent_loop via the cost tracker passed
            # above; keep the command-context pointer fresh for new commands.
            if hasattr(self, 'command_context') and self.command_context:
                self.command_context.cost_tracker = self.cost_tracker

            if self.stream and stream_started:
                self.console.print()
                self.console.print()
            else:
                self.console.print(Markdown(result.response_text))
                self.console.print("\n")

        except Exception as e:
            error_str = str(e)

            # Check for authentication errors
            if "401" in error_str or "authentication" in error_str.lower() or "令牌" in error_str:
                self.console.print(f"\n[red]❌ Authentication Error: {e}[/red]")
                self.console.print("\n[yellow]Your API key appears to be invalid or expired.[/yellow]")

                # Ask if user wants to reconfigure
                from rich.prompt import Prompt
                choice = Prompt.ask(
                    "\nWould you like to reconfigure your API key now?",
                    choices=["y", "n"],
                    default="y"
                )

                if choice == "y":
                    self._handle_relogin()
                else:
                    self.console.print("\n[dim]You can run [bold]clawd login[/bold] later to update your API key.[/dim]")
            else:
                # Generic error handling
                self.console.print(f"\n[red]Error: {e}[/red]")
                import traceback
                traceback.print_exc()

    def _handle_relogin(self):
        """Handle re-authentication when API key fails."""
        from rich.prompt import Prompt
        from src.config import set_api_key, set_default_provider
        from src.providers import PROVIDER_INFO

        self.console.print("\n[bold blue]🔑 Reconfigure API Key[/bold blue]\n")

        # Show available providers and defaults
        provider_names = list(PROVIDER_INFO.keys())
        self.console.print("[bold]Available providers:[/bold]")
        for name, info in PROVIDER_INFO.items():
            self.console.print(f"  [cyan]{name}[/cyan] - {info['label']} (default model: {info['default_model']})")
        self.console.print()

        # Select provider
        provider = Prompt.ask(
            "Select LLM provider",
            choices=provider_names,
            default=self.provider_name if self.provider_name in provider_names else "anthropic"
        )

        info = PROVIDER_INFO[provider]

        # Input API Key
        api_key = Prompt.ask(
            f"Enter {provider.upper()} API Key",
            password=True
        )

        if not api_key:
            self.console.print("\n[red]Error: API Key cannot be empty[/red]")
            return

        # Optional: Base URL (show default)
        self.console.print(f"\n[dim]Default:[/dim] {info['default_base_url']}")
        base_url = Prompt.ask(
            f"{provider.upper()} Base URL",
            default=info["default_base_url"]
        )

        # Optional: Default Model (show options)
        self.console.print(f"\n[dim]Available models:[/dim] {', '.join(info['available_models'])}")
        self.console.print(f"[dim]Default:[/dim] [bold]{info['default_model']}[/bold]")
        default_model = Prompt.ask(
            f"{provider.upper()} Default Model",
            default=info["default_model"]
        )

        # Save configuration
        set_api_key(provider, api_key=api_key, base_url=base_url, default_model=default_model)
        set_default_provider(provider)

        self.console.print(f"\n[green]✓ {provider.upper()} API Key updated successfully![/green]\n")

        # Reinitialize provider
        from src.config import get_provider_config
        from src.providers import get_provider_class

        config = get_provider_config(provider)
        provider_class = get_provider_class(provider)

        self.provider = provider_class(
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            model=config.get("default_model")
        )
        self.provider_name = provider

        self.console.print("[green]✓ Provider reinitialized. You can continue chatting![/green]\n")

    def save_session(self):
        """Save current session."""
        self.session.save()
        self.console.print(f"[green]Session saved: {self.session.session_id}[/green]")

    def load_session(self, session_id: str):
        """Load a previous session.

        Args:
            session_id: Session ID to load
        """
        from src.agent import Session

        loaded_session = Session.load(session_id)
        if loaded_session is None:
            self.console.print(f"[red]Session not found: {session_id}[/red]")
            return

        # Replace current session
        self.session = loaded_session
        self.console.print(f"[green]Session loaded: {session_id}[/green]")
        self.console.print(f"[dim]Provider: {loaded_session.provider}, Model: {loaded_session.model}[/dim]")
        self.console.print(f"[dim]Messages: {len(loaded_session.conversation.messages)}[/dim]")

        # Show conversation history
        if loaded_session.conversation.messages:
            self.console.print("\n[bold]Conversation History:[/bold]")
            for msg in loaded_session.conversation.messages[-5:]:  # Show last 5 messages
                role_color = "blue" if msg.role == "user" else "green"
                self.console.print(f"[{role_color}]{msg.role}[/{role_color}]: {msg.content[:100]}...")
