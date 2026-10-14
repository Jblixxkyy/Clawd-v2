"""Tests for the new local slash commands: /init, /model, /mcp, /resume.

Covers the pure logic helpers in src.command_system.builtins and the REPL
dispatch/handlers in src.repl.core.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.command_system.builtins import (
    add_mcp_server,
    generate_claude_md_draft,
    list_configured_providers,
    list_mcp_servers,
    list_saved_sessions,
    load_mcp_config,
)


class TestGenerateClaudeMdDraft(unittest.TestCase):
    def _project(self, tmp: Path, files: dict[str, str], dirs: tuple[str, ...] = ("src", "tests")) -> Path:
        root = tmp / "proj"
        root.mkdir()
        for d in dirs:
            (root / d).mkdir()
        for name, content in files.items():
            (root / name).write_text(content, encoding="utf-8")
        return root

    def test_pyproject_detected(self):
        with _tmpdir() as tmp:
            root = self._project(tmp, {
                "pyproject.toml": (
                    '[project]\nname = "demo"\ndescription = "Demo project"\n'
                    'requires-python = ">=3.11"\n'
                    'dependencies = ["rich", "click"]\n'
                ),
                "uv.lock": "",
            })
            draft = generate_claude_md_draft(root)
            self.assertTrue(draft.startswith("# CLAUDE.md\n"))
            self.assertIn("This file provides guidance to Claude Code", draft)
            self.assertIn("demo", draft)
            self.assertIn("Demo project", draft)
            self.assertIn("Python (>=3.11)", draft)
            self.assertIn("rich", draft)
            self.assertIn("uv run pytest", draft)
            self.assertIn("- `src/`", draft)

    def test_package_json_scripts(self):
        with _tmpdir() as tmp:
            root = self._project(tmp, {
                "package.json": json.dumps({
                    "name": "webapp",
                    "scripts": {"test": "vitest", "build": "vite build", "lint": "eslint ."},
                    "dependencies": {"react": "^19.0.0"},
                }),
            })
            draft = generate_claude_md_draft(root)
            self.assertIn("webapp", draft)
            self.assertIn("npm test", draft)
            self.assertIn("npm run build", draft)
            self.assertIn("npm run lint", draft)
            self.assertIn("react", draft)

    def test_go_mod_detected(self):
        with _tmpdir() as tmp:
            root = self._project(tmp, {"go.mod": "module example.com/tool\n\ngo 1.23\n"})
            draft = generate_claude_md_draft(root)
            self.assertIn("example.com/tool", draft)
            self.assertIn("Go 1.23", draft)
            self.assertIn("go test ./...", draft)

    def test_readme_description_fallback(self):
        with _tmpdir() as tmp:
            root = self._project(tmp, {"README.md": "# Title\n\nA tiny tool that does things well.\n\nMore docs.\n"})
            draft = generate_claude_md_draft(root)
            self.assertIn("A tiny tool that does things well.", draft)

    def test_empty_dir_still_returns_header(self):
        with _tmpdir() as tmp:
            root = tmp / "empty"
            root.mkdir()
            draft = generate_claude_md_draft(root)
            self.assertTrue(draft.startswith("# CLAUDE.md\n"))
            self.assertIn("This file provides guidance to Claude Code", draft)


class TestListConfiguredProviders(unittest.TestCase):
    def test_fields_and_flags(self):
        fake_info = {
            "glm": {
                "label": "GLM",
                "default_base_url": "https://x",
                "default_model": "glm-4.5",
                "available_models": ["glm-4.5", "glm-4"],
            },
            "openai": {
                "label": "OpenAI",
                "default_base_url": "https://y",
                "default_model": "gpt-5.4",
                "available_models": ["gpt-5.4"],
            },
        }
        fake_config = {
            "default_provider": "glm",
            "providers": {
                "glm": {"api_key": "secret", "base_url": "https://x", "default_model": "glm-4.5"},
                "openai": {"api_key": "", "base_url": "https://y", "default_model": "gpt-5.4"},
            },
        }
        with patch("src.providers.PROVIDER_INFO", fake_info):
            with patch("src.config.load_config", return_value=fake_config):
                providers = list_configured_providers()
        by_name = {p["name"]: p for p in providers}
        self.assertEqual(set(by_name), {"glm", "openai"})
        self.assertTrue(by_name["glm"]["api_key_configured"])
        self.assertTrue(by_name["glm"]["is_default"])
        self.assertFalse(by_name["openai"]["api_key_configured"])
        self.assertFalse(by_name["openai"]["is_default"])
        self.assertEqual(by_name["glm"]["available_models"], ["glm-4.5", "glm-4"])
        self.assertEqual(by_name["openai"]["default_model"], "gpt-5.4")


class TestMcpConfig(unittest.TestCase):
    def test_list_empty_when_missing(self):
        with _tmpdir() as tmp:
            with patch("pathlib.Path.home", return_value=tmp):
                self.assertEqual(load_mcp_config(), {"mcpServers": {}})
                self.assertEqual(list_mcp_servers(), [])

    def test_add_and_list_roundtrip(self):
        with _tmpdir() as tmp:
            with patch("pathlib.Path.home", return_value=tmp):
                entry = add_mcp_server("files", "/usr/bin/mcp-files", ["--root", "/tmp"])
                self.assertEqual(entry["name"], "files")
                add_mcp_server("aaa", "/bin/other")
                servers = list_mcp_servers()
                self.assertEqual([s["name"] for s in servers], ["aaa", "files"])
                files = next(s for s in servers if s["name"] == "files")
                self.assertEqual(files["command"], "/usr/bin/mcp-files")
                self.assertEqual(files["args"], ["--root", "/tmp"])
                # persisted to disk
                on_disk = json.loads((tmp / ".clawd" / "mcp.json").read_text())
                self.assertIn("files", on_disk["mcpServers"])

    def test_add_validation(self):
        with _tmpdir() as tmp:
            with patch("pathlib.Path.home", return_value=tmp):
                with self.assertRaises(ValueError):
                    add_mcp_server("", "/bin/x")
                with self.assertRaises(ValueError):
                    add_mcp_server("  ", "/bin/x")
                with self.assertRaises(ValueError):
                    add_mcp_server("srv", "")
                add_mcp_server("srv", "/bin/x")
                with self.assertRaises(ValueError):
                    add_mcp_server("srv", "/bin/y")

    def test_corrupt_config_tolerated(self):
        with _tmpdir() as tmp:
            cfg = tmp / ".clawd" / "mcp.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text("not json{{{")
            with patch("pathlib.Path.home", return_value=tmp):
                self.assertEqual(list_mcp_servers(), [])


class TestListSavedSessions(unittest.TestCase):
    def _write_session(self, sessions_dir: Path, sid: str, updated: str, msgs: int = 3):
        data = {
            "session_id": sid,
            "provider": "glm",
            "model": "glm-4.5",
            "conversation": {"messages": [{"role": "user", "content": "x"}] * msgs},
            "created_at": "2026-10-10T10:00:00",
            "updated_at": updated,
            "workspace": None,
        }
        (sessions_dir / f"{sid}.json").write_text(json.dumps(data))

    def test_sorted_newest_first(self):
        with _tmpdir() as tmp:
            sdir = tmp / ".clawd" / "sessions"
            sdir.mkdir(parents=True)
            self._write_session(sdir, "old", "2026-10-10T10:00:00", msgs=2)
            self._write_session(sdir, "new", "2026-10-11T19:00:00", msgs=5)
            (sdir / "corrupt.json").write_text("{{{")
            with patch("pathlib.Path.home", return_value=tmp):
                sessions = list_saved_sessions()
            self.assertEqual([s["session_id"] for s in sessions], ["new", "old"])
            self.assertEqual(sessions[0]["message_count"], 5)
            self.assertEqual(sessions[1]["provider"], "glm")

    def test_no_dir_returns_empty(self):
        with _tmpdir() as tmp:
            with patch("pathlib.Path.home", return_value=tmp):
                self.assertEqual(list_saved_sessions(), [])


class TestReplNewCommands(unittest.TestCase):
    """Integration tests driving ClawdREPL.handle_command."""

    def _make_repl(self, tmp: Path):
        """Build a REPL with provider/config/session mocked out (test_repl.py pattern)."""
        from src.repl import ClawdREPL

        config_file = tmp / ".clawd" / "config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(json.dumps({
            "default_provider": "glm",
            "providers": {
                "glm": {
                    "api_key": "test_api_key_12345678",
                    "base_url": "https://open.bigmodel.cn/api/paas/v4",
                    "default_model": "glm-4.5",
                }
            },
        }))

        patches = [
            patch("src.config.get_config_path", return_value=config_file),
            patch("src.repl.core.Session.create"),
            patch("src.repl.core.get_provider_class"),
            patch("pathlib.Path.home", return_value=tmp),
        ]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(patches)])

        import src.repl.core as core_mod
        mock_session = Mock()
        mock_session.session_id = "test-session"
        mock_session.conversation = Mock()
        mock_session.provider = "glm"
        mock_session.model = "glm-4.5"
        core_mod.Session.create.return_value = mock_session

        mock_provider = Mock()
        mock_provider.model = "glm-4.5"
        core_mod.get_provider_class.return_value = Mock(return_value=mock_provider)

        repl = ClawdREPL(provider_name="glm")
        repl.session = mock_session
        return repl

    def test_palette_includes_new_commands(self):
        with _tmpdir() as tmp:
            repl = self._make_repl(tmp)
            words = repl._get_slash_command_words()
            for cmd in ("/init", "/model", "/mcp", "/resume"):
                self.assertIn(cmd, words)

    def test_init_decline_does_not_write(self):
        with _tmpdir() as tmp:
            repl = self._make_repl(tmp)
            workdir = tmp / "work"
            workdir.mkdir()
            (workdir / "pyproject.toml").write_text('[project]\nname = "w"\n')
            repl.tool_context.workspace_root = workdir
            with patch("builtins.input", return_value=""):
                repl.handle_command("/init")
            self.assertFalse((workdir / "CLAUDE.md").exists())

    def test_init_confirm_writes_draft(self):
        with _tmpdir() as tmp:
            repl = self._make_repl(tmp)
            workdir = tmp / "work"
            workdir.mkdir()
            (workdir / "pyproject.toml").write_text('[project]\nname = "w"\n')
            repl.tool_context.workspace_root = workdir
            with patch("builtins.input", return_value="y"):
                repl.handle_command("/init")
            target = workdir / "CLAUDE.md"
            self.assertTrue(target.exists())
            self.assertIn("# CLAUDE.md", target.read_text())

    def test_mcp_add_via_command(self):
        with _tmpdir() as tmp:
            repl = self._make_repl(tmp)
            repl.handle_command("/mcp add files /usr/bin/mcp-files --root /tmp")
            servers = list_mcp_servers()
            self.assertEqual(len(servers), 1)
            self.assertEqual(servers[0]["name"], "files")
            self.assertEqual(servers[0]["args"], ["--root", "/tmp"])

    def test_mcp_add_invalid_shows_error(self):
        with _tmpdir() as tmp:
            repl = self._make_repl(tmp)
            with patch("builtins.input", side_effect=["", "/bin/x", ""]):
                repl.handle_command("/mcp add")
            self.assertEqual(list_mcp_servers(), [])

    def test_mcp_list_empty(self):
        with _tmpdir() as tmp:
            repl = self._make_repl(tmp)
            repl.handle_command("/mcp")  # must not raise

    def test_resume_no_sessions(self):
        with _tmpdir() as tmp:
            repl = self._make_repl(tmp)
            repl.handle_command("/resume")  # must not raise
            self.assertEqual(repl.session.session_id, "test-session")

    def test_resume_loads_selected_session(self):
        with _tmpdir() as tmp:
            from src.agent.session import Session as RealSession

            repl = self._make_repl(tmp)
            real = RealSession.create("glm", "glm-4.5")
            real.save()  # writes under patched Path.home
            with patch("builtins.input", return_value="1"):
                repl.handle_command("/resume")
            self.assertEqual(repl.session.session_id, real.session_id)

    def test_resume_invalid_selection_keeps_session(self):
        with _tmpdir() as tmp:
            from src.agent.session import Session as RealSession

            repl = self._make_repl(tmp)
            RealSession.create("glm", "glm-4.5").save()
            with patch("builtins.input", return_value="99"):
                repl.handle_command("/resume")
            self.assertEqual(repl.session.session_id, "test-session")

    def test_model_switch_mid_session(self):
        with _tmpdir() as tmp:
            repl = self._make_repl(tmp)
            fake_info = {
                "glm": {
                    "label": "GLM",
                    "default_base_url": "https://x",
                    "default_model": "glm-4.5",
                    "available_models": ["glm-4.5", "glm-4"],
                }
            }
            new_instance = Mock()
            new_instance.model = "glm-4"
            with patch("src.providers.PROVIDER_INFO", fake_info), \
                 patch("src.config.load_config", return_value={
                     "default_provider": "glm",
                     "providers": {
                         "glm": {"api_key": "k", "base_url": "https://x", "default_model": "glm-4.5"}
                     },
                 }), \
                 patch("src.config.get_provider_config",
                       return_value={"api_key": "k", "base_url": "https://x"}), \
                 patch("src.providers.get_provider_class",
                       return_value=Mock(return_value=new_instance)), \
                 patch("src.config.set_api_key"), \
                 patch("src.config.set_default_provider"), \
                 patch("builtins.input", side_effect=["1", "2"]):
                repl.handle_command("/model")
            self.assertEqual(repl.provider_name, "glm")
            self.assertIs(repl.provider, new_instance)
            self.assertEqual(repl.session.model, "glm-4")
            from src.repl.status_bar import render_status
            self.assertIn("glm-4", render_status(repl))

    def test_model_preselect_provider(self):
        with _tmpdir() as tmp:
            repl = self._make_repl(tmp)
            fake_info = {
                "glm": {
                    "label": "GLM",
                    "default_base_url": "https://x",
                    "default_model": "glm-4.5",
                    "available_models": ["glm-4.5"],
                }
            }
            new_instance = Mock()
            new_instance.model = "glm-4.5"
            with patch("src.providers.PROVIDER_INFO", fake_info), \
                 patch("src.config.load_config", return_value={
                     "default_provider": "glm",
                     "providers": {
                         "glm": {"api_key": "k", "base_url": "https://x", "default_model": "glm-4.5"}
                     },
                 }), \
                 patch("src.config.get_provider_config",
                       return_value={"api_key": "k", "base_url": "https://x"}), \
                 patch("src.providers.get_provider_class",
                       return_value=Mock(return_value=new_instance)), \
                 patch("src.config.set_api_key"), \
                 patch("src.config.set_default_provider"):
                repl.handle_command("/model glm")
            self.assertIs(repl.provider, new_instance)

    def test_model_unknown_provider(self):
        with _tmpdir() as tmp:
            repl = self._make_repl(tmp)
            repl.handle_command("/model nope")  # must not raise
            self.assertEqual(repl.provider_name, "glm")


class _tmpdir:
    """Context manager yielding a TemporaryDirectory Path (avoids nested withs)."""

    def __init__(self):
        self._td = None

    def __enter__(self) -> Path:
        import tempfile

        self._td = tempfile.TemporaryDirectory()
        return Path(self._td.name)

    def __exit__(self, *exc):
        self._td.cleanup()
        return False


if __name__ == "__main__":
    unittest.main()
