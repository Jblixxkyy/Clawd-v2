"""CLI entry point for Clawd Codex."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table


def main():
    """CLI main entry point."""
    # Quick path for --version
    if len(sys.argv) == 2 and sys.argv[1] in ['--version', '-v', '-V']:
        from src import __version__
        print(f"clawd-codex version {__version__} (Python)")
        return 0

    parser = argparse.ArgumentParser(
        description="Clawd Codex - Claude Code Python Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  clawd --version          Show version
  clawd login              Configure API keys
  clawd config             Show current configuration
  clawd web                Start the browser app for this workspace
  clawd web --reload       Restart the browser app automatically after source edits
  clawd --no-stream      Start REPL without live response rendering
  clawd                    Start interactive REPL
"""
    )

    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version information'
    )
    parser.add_argument(
        '--config',
        action='store_true',
        help='Show current configuration'
    )
    parser.add_argument(
        '--stream', '--no-stream',
        dest='stream',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Enable/disable live response rendering in the REPL (default: enabled)'
    )
    parser.add_argument(
        '-p', '--print',
        dest='print_',
        metavar='PROMPT',
        default=None,
        help='Run one agent turn for PROMPT non-interactively, print the answer, and exit'
    )
    parser.add_argument(
        '--no-auto-compact',
        dest='auto_compact',
        action='store_false',
        default=None,
        help='Disable automatic context compaction when nearing the model context limit'
    )
    parser.add_argument(
        '--permission-mode',
        dest='permission_mode',
        metavar='MODE',
        choices=['default', 'acceptEdits', 'dontAsk', 'bypassPermissions'],
        default=None,
        help='Session permission mode: default, acceptEdits, dontAsk, or bypassPermissions (deny rules always win)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # login subcommand
    login_parser = subparsers.add_parser('login', help='Configure API keys')

    # config subcommand
    config_parser = subparsers.add_parser('config', help='Show current configuration')

    # web subcommand
    web_parser = subparsers.add_parser('web', help='Start the browser app for this workspace')
    web_parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='Port for the local-network web app (default: 8765; with --kill, only kill this port)',
    )
    web_parser.add_argument(
        '--kill',
        action='store_true',
        help='Kill running Clawd web app(s) tracked in ~/.clawd (all, or one with --port)',
    )
    web_parser.add_argument(
        '--reload',
        action='store_true',
        help='Restart the server and refresh connected browsers after Clawd source edits',
    )
    web_parser.add_argument('--_reload-worker', action='store_true', help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Handle --version
    if args.version:
        from src import __version__
        print(f"clawd-codex version {__version__} (Python)")
        return 0

    # Handle --config
    if args.config:
        return show_config()

    # Handle commands
    if args.command == 'login':
        return handle_login()
    elif args.command == 'config':
        return show_config()
    elif args.command == 'web':
        if args.kill:
            from src.web_app import kill_web_apps

            summary = kill_web_apps(port=args.port)
            if not (summary["killed"] or summary["not_running"] or summary["skipped"]):
                scope = f" on port {args.port}" if args.port else ""
                print(f"No tracked Clawd web apps running{scope}.")
            for entry in summary["killed"]:
                print(f"Killed Clawd web app on {entry}.")
            for entry in summary["not_running"]:
                print(f"{entry}.")
            for entry in summary["skipped"]:
                print(f"Could not kill Clawd web app on {entry}.")
            return 0
        from src.web_app import run_web_app

        return run_web_app(
            workspace=Path.cwd(),
            port=args.port or 8765,
            reload=args.reload,
            reload_worker=args._reload_worker,
        )

    # Script mode: one non-interactive turn (stays non-streamed unless asked)
    if args.print_:
        from src.repl.print_mode import run_print
        return run_print(
            args.print_,
            stream=bool(args.stream),
            permission_mode=args.permission_mode,
            auto_compact=args.auto_compact,
        )

    # Default: start REPL
    return start_repl(stream=args.stream, permission_mode=args.permission_mode, auto_compact=args.auto_compact)


def _show_provider_defaults_table() -> None:
    """Print a table showing available providers and their defaults."""
    from src.providers import PROVIDER_INFO

    console = Console()
    table = Table(title="Available Providers & Defaults", show_header=True, header_style="bold")
    table.add_column("Provider", style="cyan")
    table.add_column("Default Model", style="magenta")
    table.add_column("Base URL", style="green")

    for name, info in PROVIDER_INFO.items():
        table.add_row(
            f"{name} ({info['label']})",
            info["default_model"],
            info["default_base_url"],
        )

    console.print(table)
    console.print()


def handle_login():
    """Interactive API configuration."""
    console = Console()
    console.print("\n[bold blue]Clawd Codex - API Configuration[/bold blue]\n")

    # Show available providers and their defaults
    _show_provider_defaults_table()

    # Select provider
    from src.providers import PROVIDER_INFO
    provider_names = list(PROVIDER_INFO.keys())

    provider = Prompt.ask(
        "Select LLM provider",
        choices=provider_names,
        default="anthropic"
    )

    info = PROVIDER_INFO[provider]

    # Input API Key
    api_key = Prompt.ask(
        f"Enter {provider.upper()} API Key",
        password=True
    )

    if not api_key:
        console.print("\n[red]Error: API Key cannot be empty[/red]")
        return 1

    # Optional: Base URL (show default)
    console.print(f"\n[dim]Default:[/dim] {info['default_base_url']}")
    base_url = Prompt.ask(
        f"{provider.upper()} Base URL",
        default=info["default_base_url"]
    )

    # Optional: Default Model (show available options)
    console.print(f"\n[dim]Available models:[/dim] {', '.join(info['available_models'])}")
    console.print(f"[dim]Default:[/dim] [bold]{info['default_model']}[/bold]")
    default_model = Prompt.ask(
        f"{provider.upper()} Default Model",
        default=info["default_model"]
    )

    # Save configuration
    from src.config import set_api_key, set_default_provider

    set_api_key(provider, api_key=api_key, base_url=base_url, default_model=default_model)
    set_default_provider(provider)

    console.print(f"\n[green]✓ {provider.upper()} API Key saved successfully![/green]")
    console.print(f"[green]✓ Default provider set to: {provider}[/green]\n")
    return 0


def show_config():
    """Show current configuration."""
    console = Console()

    try:
        from src.config import load_config, get_config_path

        config = load_config()
        config_path = get_config_path()

        console.print(f"\n[bold]Configuration File:[/bold] {config_path}\n")
        console.print("[bold]Current Configuration:[/bold]\n")

        # Show default provider
        console.print(f"[cyan]Default Provider:[/cyan] {config.get('default_provider', 'Not set')}")

        # Show providers (without showing full API keys)
        console.print("\n[cyan]Configured Providers:[/cyan]")
        for provider_name, provider_config in config.get("providers", {}).items():
            api_key = provider_config.get("api_key", "")
            masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "Not set"

            console.print(f"\n  [yellow]{provider_name.upper()}:[/yellow]")
            console.print(f"    API Key: {masked_key}")
            console.print(f"    Base URL: {provider_config.get('base_url', 'Not set')}")
            console.print(f"    Default Model: {provider_config.get('default_model', 'Not set')}")

        console.print()

    except Exception as e:
        console.print(f"\n[red]Error loading configuration: {e}[/red]\n")
        return 1

    return 0


def start_repl(stream: bool | None = None, permission_mode: str | None = None, auto_compact: bool | None = None):
    """Start interactive REPL (streaming on by default; pass False to disable)."""
    from src.config import get_default_provider
    from src.repl import ClawdREPL

    provider = get_default_provider()
    if stream is None:
        repl = ClawdREPL(provider_name=provider, auto_compact=auto_compact)
    else:
        repl = ClawdREPL(provider_name=provider, stream=stream, auto_compact=auto_compact)
    if permission_mode is not None:
        from src.tool_system.permissions import PermissionMode
        repl.tool_context.permission_mode = PermissionMode.from_string(permission_mode)
    repl.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
