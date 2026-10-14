"""Interactive permission handling for tools.

This module provides the permission checking and interactive user consent flow
when tools require user permission before execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from .permissions import PermissionMode, ToolPermissionContext


class PermissionBehavior(Enum):
    """Possible permission behaviors."""
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True)
class PermissionResult:
    """Result of a permission check.

    Attributes:
        behavior: The permission behavior - allow, deny, or ask.
        message: Optional message to display to the user.
        suggestion: Optional suggestion for what the user could allow.
        updated_input: Optional updated input if permission modifies the request.
    """
    behavior: PermissionBehavior
    message: Optional[str] = None
    suggestion: Optional[str] = None
    updated_input: Optional[dict[str, Any]] = None

    @staticmethod
    def allow(updated_input: Optional[dict[str, Any]] = None) -> "PermissionResult":
        """Return an allow result."""
        return PermissionResult(behavior=PermissionBehavior.ALLOW, updated_input=updated_input)

    @staticmethod
    def deny(message: str) -> "PermissionResult":
        """Return a deny result."""
        return PermissionResult(behavior=PermissionBehavior.DENY, message=message)

    @staticmethod
    def ask(message: str, suggestion: Optional[str] = None) -> "PermissionResult":
        """Return an ask result."""
        return PermissionResult(behavior=PermissionBehavior.ASK, message=message, suggestion=suggestion)


@dataclass
class InteractivePermissionHandler:
    """Handler for interactive permission requests.

    This class provides the UI and logic for asking users for permission
    when tools need consent before execution.
    """

    console: Any = field(default=None)  # Rich console for output
    prompt_func: Optional[Callable[[str], str]] = field(default=None)  # Custom prompt function

    def __init__(
        self,
        console: Any = None,
        prompt_func: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Initialize the handler.

        Args:
            console: Rich console for formatted output. If None, uses print().
            prompt_func: Function to prompt user for input. If None, uses input().
        """
        self.console = console
        self.prompt_func = prompt_func or input

    def _print(self, message: str, style: str = "") -> None:
        """Print a message with optional styling."""
        if self.console:
            if style:
                self.console.print(f"[{style}]{message}[/{style}]")
            else:
                self.console.print(message)
        else:
            print(message)

    def _prompt(self, message: str) -> str:
        """Prompt user for input."""
        return self.prompt_func(message)

    def handle_permission_request(
        self,
        tool_name: str,
        permission_result: PermissionResult,
        context: Any,  # ToolContext
    ) -> tuple[PermissionBehavior, Optional[dict[str, Any]]]:
        """Handle an interactive permission request.

        Args:
            tool_name: Name of the tool requesting permission.
            permission_result: The permission result with behavior='ask'.
            context: The tool context (used to update permission settings).

        Returns:
            A tuple of (PermissionBehavior, updated_input).
            The behavior will be ALLOW or DENY based on user choice.
            The updated_input may be modified if user chooses to proceed.
        """
        if permission_result.behavior != PermissionBehavior.ASK:
            # Not an ask result - return as-is
            return permission_result.behavior, permission_result.updated_input

        message = permission_result.message or f"Tool '{tool_name}' requires permission to proceed."
        suggestion = permission_result.suggestion

        self._print("")
        self._print(f"[bold yellow]⚠ Permission Required[/bold yellow]", "yellow")
        self._print(f"  {message}")
        self._print("")

        # Show the suggestion if available
        if suggestion:
            self._print(f"  [dim]Suggestion: {suggestion}[/dim]")

        # Show available options
        options = ["y", "n"]
        option_descriptions = [
            "Yes, allow this action",
            "No, deny this action",
        ]

        # Check if this is a setting that can be permanently enabled
        can_enable_setting = self._can_enable_setting(permission_result, context)
        if can_enable_setting:
            options.insert(0, "e")
            option_descriptions.insert(0, f"Enable setting and allow ({can_enable_setting})")

        self._print("")
        for i, (opt, desc) in enumerate(zip(options, option_descriptions)):
            self._print(f"  {i + 1}. [{opt}] {desc}")

        self._print("")

        # Get user choice
        choice = self._prompt("Select option> ").strip().lower()

        # Parse choice
        if choice in ("1", "y", "yes", ""):
            return PermissionBehavior.ALLOW, permission_result.updated_input
        elif choice in ("2", "n", "no"):
            return PermissionBehavior.DENY, None
        elif choice == "e" and "e" in options:
            # User chose to enable the setting
            self._enable_setting(permission_result, context)
            return PermissionBehavior.ALLOW, permission_result.updated_input
        else:
            # Default to deny for invalid input
            self._print("[dim]Invalid choice, defaulting to deny.[/dim]")
            return PermissionBehavior.DENY, None

    def _can_enable_setting(
        self,
        permission_result: PermissionResult,
        context: Any,
    ) -> Optional[str]:
        """Check if a permission setting can be permanently enabled.

        Returns a description of the setting if it can be enabled, None otherwise.
        """
        message = permission_result.message or ""

        # Check for allow_docs setting
        if "allow_docs" in message.lower() or "documentation files" in message.lower():
            if hasattr(context, 'permission_context'):
                pc = context.permission_context
                if hasattr(pc, 'allow_docs') and not pc.allow_docs:
                    return "allow_docs"
        return None

    def _enable_setting(
        self,
        permission_result: PermissionResult,
        context: Any,
    ) -> None:
        """Enable a permission setting permanently."""
        setting_name = self._can_enable_setting(permission_result, context)
        if not setting_name:
            return

        self._print(f"\n[dim]Enabling {setting_name}...[/dim]")

        if hasattr(context, 'permission_context'):
            pc = context.permission_context
            if setting_name == "allow_docs" and hasattr(pc, 'allow_docs'):
                # Create a new permission context with allow_docs=True
                from .permissions import ToolPermissionContext
                new_pc = ToolPermissionContext(
                    deny_names=pc.deny_names,
                    deny_prefixes=pc.deny_prefixes,
                    workspace_root=pc.workspace_root,
                    additional_working_directories=pc.additional_working_directories,
                    allow_docs=True,
                )
                context.permission_context = new_pc
                self._print(f"[green]✓ {setting_name} enabled for this session[/green]")


# ---------------------------------------------------------------------------
# Session permission modes
# ---------------------------------------------------------------------------

# Tool names (lowercased) that ACCEPT_EDITS mode auto-allows. Everything else
# still prompts unless pre-approved.
_EDIT_AUTO_ALLOW_NAMES = frozenset({"edit", "write"})


def apply_permission_mode(
    tool_name: str,
    permission_result: "PermissionResult",
    mode: "PermissionMode",
    *,
    permission_context: "ToolPermissionContext | None" = None,
    always_allowed: bool = False,
) -> "PermissionResult":
    """Layer a session permission mode over a tool's own permission result.

    Deny->ask->allow layering: deny rules ALWAYS win, regardless of mode.
    Only the ASK layer is transformed:

    - DEFAULT: result unchanged (current interactive behavior).
    - ACCEPT_EDITS: ASK becomes ALLOW for Edit/Write-type tools; every
      other tool keeps its own result (prompts as usual).
    - DONT_ASK: ASK becomes ALLOW only when the tool is pre-approved
      (``always_allowed``); otherwise it becomes DENY with NO interactive
      prompt. This formalizes what ``clawd -p`` script mode already does.
    - BYPASS_PERMISSIONS: ASK becomes ALLOW for every tool. Deny rules
      (``permission_context`` deny names/prefixes) still win.

    Non-ASK results (ALLOW/DENY from the tool itself) are never upgraded by
    a mode -- a tool-level deny stays denied in every mode.
    """
    from .permissions import PermissionMode as _PermissionMode

    # Deny layer always wins, in every mode.
    if permission_context is not None and permission_context.blocks_tool(tool_name):
        return PermissionResult.deny(
            f"tool is blocked by permission context: {tool_name}"
        )
    if permission_result.behavior is not PermissionBehavior.ASK:
        return permission_result
    if mode is _PermissionMode.ACCEPT_EDITS:
        if tool_name.lower() in _EDIT_AUTO_ALLOW_NAMES:
            return PermissionResult.allow(
                updated_input=permission_result.updated_input
            )
        return permission_result
    if mode is _PermissionMode.DONT_ASK:
        if always_allowed:
            return PermissionResult.allow(
                updated_input=permission_result.updated_input
            )
        return PermissionResult.deny(
            f"permission denied: '{tool_name}' is not pre-approved "
            "(permission mode: dontAsk -- no interactive prompts)"
        )
    if mode is _PermissionMode.BYPASS_PERMISSIONS:
        return PermissionResult.allow(updated_input=permission_result.updated_input)
    # DEFAULT (or unknown): current behavior.
    return permission_result
