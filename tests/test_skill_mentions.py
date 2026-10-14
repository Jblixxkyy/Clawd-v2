"""Tests for skill $mentions in user input (harness brief 009)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.repl.skill_mentions import (
    MENTION_RE,
    find_skill_mentions,
    resolve_skill_mentions,
)
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry


SKILL_MD = """---
description: Read and summarize PDF files.
---

You are the PDF skill. Follow these steps:
1. Extract the text of the PDF.
2. Summarize it in three bullet points.
"""


def _make_skills_dir(tmp: Path, skills: dict[str, str]) -> Path:
    """Create a skills dir with ``name/SKILL.md`` for each entry."""
    skills_dir = tmp / "skills"
    for name, body in skills.items():
        skill_dir = skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
    skills_dir.mkdir(parents=True, exist_ok=True)
    return skills_dir


def _isolated_env(tmp: Path, skills_dir: Path) -> dict:
    """Env patch isolating skill resolution to ``skills_dir``.

    The loader also reads ``~/.clawd/skills`` and ``~/.claude/skills``;
    pointing HOME at an empty dir keeps real user skills out of tests.
    """
    home = tmp / "home"
    home.mkdir(parents=True, exist_ok=True)
    return patch.dict(
        os.environ, {"HOME": str(home), "CLAWD_SKILLS_DIR": str(skills_dir)}
    )


class TestParseSkillMentions(unittest.TestCase):
    """Unit tests for the $mention regex / parser."""

    def test_bare_dollar_name_parses(self):
        self.assertEqual(find_skill_mentions("summarize this $pdf"), ["pdf"])

    def test_skill_uri_form_parses(self):
        self.assertEqual(find_skill_mentions("$skill://pdf"), ["pdf"])

    def test_both_forms_parse_to_same_name(self):
        self.assertEqual(find_skill_mentions("$pdf"), ["pdf"])
        self.assertEqual(find_skill_mentions("$skill://pdf"), ["pdf"])

    def test_plain_dollar_ignored(self):
        self.assertEqual(find_skill_mentions("it costs $"), [])
        self.assertEqual(find_skill_mentions("it costs $ 5"), [])
        self.assertEqual(find_skill_mentions("$"), [])
        self.assertEqual(find_skill_mentions("$ "), [])

    def test_skill_uri_without_name_ignored(self):
        self.assertEqual(find_skill_mentions("$skill://"), [])
        self.assertEqual(find_skill_mentions("$skill:// "), [])

    def test_dollar_mid_word_still_matches_name(self):
        # Per the brief's regex \$([a-zA-Z0-9_-]+): name chars after $ parse.
        self.assertEqual(find_skill_mentions("$50"), ["50"])

    def test_multiple_mentions_in_order(self):
        self.assertEqual(
            find_skill_mentions("use $pdf then $skill://ocr please"),
            ["pdf", "ocr"],
        )

    def test_duplicate_mentions_deduped(self):
        self.assertEqual(find_skill_mentions("$pdf and $pdf"), ["pdf"])
        self.assertEqual(find_skill_mentions("$pdf and $skill://pdf"), ["pdf"])

    def test_name_characters(self):
        self.assertEqual(find_skill_mentions("$my-skill_2"), ["my-skill_2"])

    def test_no_mention(self):
        self.assertEqual(find_skill_mentions("just a normal message"), [])
        self.assertEqual(find_skill_mentions(""), [])

    def test_mention_regex_named_groups(self):
        m = MENTION_RE.search("$skill://pdf")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "pdf")
        m = MENTION_RE.search("$pdf")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), "pdf")


class TestResolveSkillMentions(unittest.TestCase):
    """Resolution against the skill index through the Skill tool path."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.skills_dir = _make_skills_dir(self.tmp, {"pdf": SKILL_MD})
        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.tmp)

    def _resolve(self, text, skills_dir=None):
        skills_dir = self.skills_dir if skills_dir is None else skills_dir
        with _isolated_env(self.tmp, skills_dir):
            return resolve_skill_mentions(
                text, tool_registry=self.registry, tool_context=self.context
            )

    def test_no_mention_passes_through_byte_identical(self):
        text = "just a normal message with $ and $ signs, nothing to resolve"
        res = self._resolve(text)
        self.assertIsNone(res.error)
        self.assertEqual(res.skills, [])
        self.assertEqual(res.input_text, text)

    def test_bare_form_injects_skill_content(self):
        res = self._resolve("summarize this $pdf")
        self.assertIsNone(res.error)
        self.assertEqual(res.skills, ["pdf"])
        # Original user text preserved verbatim...
        self.assertIn("summarize this $pdf", res.input_text)
        # ...the model is told the user explicitly requested it...
        self.assertIn("explicitly invoked skill 'pdf'", res.input_text)
        # ...and the SKILL.md content reaches the model context.
        self.assertIn("Extract the text of the PDF.", res.input_text)

    def test_skill_uri_form_injects_skill_content(self):
        res = self._resolve("summarize this $skill://pdf")
        self.assertIsNone(res.error)
        self.assertEqual(res.skills, ["pdf"])
        self.assertIn("Extract the text of the PDF.", res.input_text)

    def test_multiple_mentions_all_injected_in_order(self):
        _make_skills_dir(self.tmp, {"pdf": SKILL_MD, "ocr": SKILL_MD.replace("PDF", "image")})
        res = self._resolve("$pdf then $ocr")
        self.assertIsNone(res.error)
        self.assertEqual(res.skills, ["pdf", "ocr"])
        self.assertIn("Extract the text of the PDF.", res.input_text)
        self.assertIn("Extract the text of the image.", res.input_text)
        # pdf's block comes before ocr's block
        self.assertLess(
            res.input_text.index("Extract the text of the PDF."),
            res.input_text.index("Extract the text of the image."),
        )

    def test_unknown_mention_helpful_error(self):
        empty_dir = self.tmp / "empty-skills"
        empty_dir.mkdir(exist_ok=True)
        res = self._resolve("do the $unknown thing", skills_dir=empty_dir)
        self.assertEqual(res.skills, [])
        self.assertIsNotNone(res.error)
        self.assertIn("unknown skill", res.error)
        # Reuses the Skill tool's listing: available skills or the empty case.
        self.assertTrue(
            "Available skills" in res.error or "No skills are installed" in res.error,
            f"error did not list skills: {res.error}",
        )

    def test_unknown_mention_leaves_input_untouched(self):
        text = "do the $unknown thing"
        res = self._resolve(text)
        self.assertEqual(res.input_text, text)


class TestREPLMentionHook(unittest.TestCase):
    """The REPL pre-processes mentions before the agent loop."""

    def _make_repl(self):
        import json

        from src.repl import ClawdREPL

        config_dir = Path(tempfile.mkdtemp()) / ".clawd"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "config.json").write_text(
            json.dumps(
                {
                    "default_provider": "glm",
                    "providers": {
                        "glm": {
                            "api_key": "test_api_key_12345678",
                            "base_url": "https://example.invalid",
                            "default_model": "glm-4.5",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        with patch("src.config.get_config_path") as mock_path, patch(
            "src.repl.core.Session.create"
        ) as mock_session, patch(
            "src.repl.core.get_provider_class"
        ) as mock_provider_class:
            mock_path.return_value = config_dir / "config.json"
            mock_session.return_value = Mock()
            mock_provider = Mock()
            mock_provider.model = "glm-4.5"
            mock_provider_class.return_value = Mock(return_value=mock_provider)
            return ClawdREPL(provider_name="glm")

    def test_no_mention_returns_input_unchanged(self):
        repl = self._make_repl()
        text = "hello world"
        self.assertEqual(repl._expand_skill_mentions(text), text)

    def test_unknown_mention_aborts_turn(self):
        repl = self._make_repl()
        tmp = Path(tempfile.mkdtemp())
        repl.tool_registry = build_default_registry()
        repl.tool_context = ToolContext(workspace_root=tmp)
        empty_dir = tmp / "empty-skills"
        empty_dir.mkdir(exist_ok=True)
        with _isolated_env(tmp, empty_dir):
            with patch.object(repl.console, "print") as mock_print:
                result = repl._expand_skill_mentions("run $unknown")
        self.assertIsNone(result)
        printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list)
        self.assertIn("unknown skill", printed)

    def test_known_mention_expands_in_repl(self):
        repl = self._make_repl()
        tmp = Path(tempfile.mkdtemp())
        skills_dir = _make_skills_dir(tmp, {"pdf": SKILL_MD})
        repl.tool_registry = build_default_registry()
        repl.tool_context = ToolContext(workspace_root=tmp)
        with _isolated_env(tmp, skills_dir):
            result = repl._expand_skill_mentions("summarize this $pdf")
        self.assertIsNotNone(result)
        self.assertIn("summarize this $pdf", result)
        self.assertIn("Extract the text of the PDF.", result)


if __name__ == "__main__":
    unittest.main()
