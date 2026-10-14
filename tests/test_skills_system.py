from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.skills.create import create_skill
from src.skills.frontmatter import parse_frontmatter
from src.skills.loader import clear_skill_registry, get_all_skills
from src.tool_system.context import ToolContext
from src.tool_system.tools import SkillTool


class SkillSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self) -> None:
        clear_skill_registry()
        self.tmp.cleanup()


class TestSkillCreate(SkillSystemTests):
    def test_create_writes_skill_md(self) -> None:
        skills_dir = self.root / "skills"
        p = create_skill(
            directory=skills_dir,
            name="demo",
            description="demo skill",
            when_to_use="use it when testing",
            allowed_tools=["Read", "Grep"],
            arguments=["foo", "bar"],
            body="Hello $foo $1",
        )
        self.assertTrue(p.exists())
        parsed = parse_frontmatter(p.read_text(encoding="utf-8"))
        self.assertEqual(parsed.frontmatter["description"], "demo skill")
        self.assertEqual(parsed.frontmatter["when_to_use"], "use it when testing")

    def test_parse_frontmatter_supports_inline_lists(self) -> None:
        parsed = parse_frontmatter(
            "---\n"
            "arguments: [name]\n"
            "allowed-tools: [Read, Grep]\n"
            "---\n"
            "Hello $name\n"
        )
        self.assertEqual(parsed.frontmatter["arguments"], ["name"])
        self.assertEqual(parsed.frontmatter["allowed-tools"], ["Read", "Grep"])


class TestSkillRegister(SkillSystemTests):
    def test_register_loads_skill_from_dir(self) -> None:
        skills_dir = self.root / "skills"
        create_skill(
            directory=skills_dir,
            name="hello",
            description="say hello",
            body="Hello",
        )
        with patch.dict(os.environ, {"CLAWD_SKILLS_DIR": str(skills_dir)}):
            skills = get_all_skills(project_root=self.root)
            by_name = {s.name: s for s in skills}
            self.assertIn("hello", by_name)
            self.assertEqual(by_name["hello"].description, "say hello")
            self.assertEqual(by_name["hello"].loaded_from, "user")


class TestSkillUse(SkillSystemTests):
    def test_use_substitutes_arguments(self) -> None:
        skills_dir = self.root / "skills"
        create_skill(
            directory=skills_dir,
            name="hello",
            description="say hello",
            arguments=["name"],
            body="Hello $name ($0) / $ARGUMENTS",
        )
        ctx = ToolContext(workspace_root=self.root)
        with patch.dict(os.environ, {"CLAWD_SKILLS_DIR": str(skills_dir)}):
            out = SkillTool().run({"skill": "hello", "args": 'bob "the builder"'}, ctx).output
            self.assertTrue(out["success"])
            self.assertIn("Hello bob (bob)", out["prompt"])
            self.assertIn('bob "the builder"', out["prompt"])


class SkillPromptInjectionTests(unittest.TestCase):
    """Covers the merged prompt work: frontmatter parsing, skills section rendering,
    unknown-skill hints, and the base system prompt."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_skill(self, name: str, frontmatter: str, body: str = "Do the thing.") -> None:
        d = self.root / ".clawd" / "skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8")

    def test_frontmatter_description_with_commas_stays_a_string(self) -> None:
        parsed = parse_frontmatter("---\nname: x\ndescription: Use when A, B, or C\n---\nbody")
        self.assertEqual(parsed.frontmatter["description"], "Use when A, B, or C")

    def test_frontmatter_list_keys_still_split_on_commas(self) -> None:
        parsed = parse_frontmatter("---\nallowed-tools: Read, Grep\n---\nbody")
        self.assertEqual(parsed.frontmatter["allowed-tools"], ["Read", "Grep"])

    def test_frontmatter_strips_quotes(self) -> None:
        parsed = parse_frontmatter('---\nname: "quoted"\n---\nbody')
        self.assertEqual(parsed.frontmatter["name"], "quoted")

    def test_find_all_skills_thread_safe(self) -> None:
        from src.skills.loader import find_all_skills

        with patch.dict(os.environ, {"CLAWD_SKILLS_DIR": str(self.root / "none"), "HOME": str(self.root)}):
            self._write_skill("alpha", "description: Use when doing A")
            found = find_all_skills(project_root=self.root)
        self.assertEqual([s.name for s in found], ["alpha"])

    def test_skills_section_lists_name_and_description(self) -> None:
        from src.tool_system.system_prompt import build_skills_section
        from types import SimpleNamespace

        with patch.dict(os.environ, {"CLAWD_SKILLS_DIR": str(self.root / "none"), "HOME": str(self.root)}):
            self._write_skill("zeta", "description: Use when doing Z, carefully")
            self._write_skill("alpha", "description: Use when doing A")
            ctx = SimpleNamespace(cwd=self.root, workspace_root=self.root)
            text = build_skills_section(ctx)
        self.assertIn("## Available Skills", text)
        self.assertIn("- /alpha: Use when doing A", text)
        self.assertIn("- /zeta: Use when doing Z, carefully", text)
        self.assertLess(text.index("- /alpha:"), text.index("- /zeta:"))
        self.assertIn("## Installing Skills", text)

    def test_skills_section_includes_when_to_use(self) -> None:
        from src.tool_system.system_prompt import build_skills_section
        from types import SimpleNamespace

        with patch.dict(os.environ, {"CLAWD_SKILLS_DIR": str(self.root / "none"), "HOME": str(self.root)}):
            self._write_skill("helper", "description: Helps\nwhen_to_use: when stuck on X")
            ctx = SimpleNamespace(cwd=self.root, workspace_root=self.root)
            text = build_skills_section(ctx)
        self.assertIn("when stuck on X", text)

    def test_skills_section_hides_model_disabled_skills(self) -> None:
        from src.tool_system.system_prompt import build_skills_section
        from types import SimpleNamespace

        with patch.dict(os.environ, {"CLAWD_SKILLS_DIR": str(self.root / "none"), "HOME": str(self.root)}):
            self._write_skill("secret", "description: Hidden\ndisable-model-invocation: true")
            self._write_skill("shown", "description: Visible")
            ctx = SimpleNamespace(cwd=self.root, workspace_root=self.root)
            text = build_skills_section(ctx)
        self.assertIn("- /shown:", text)
        self.assertNotIn("secret", text)

    def test_skills_section_empty_when_no_skills(self) -> None:
        from src.tool_system.system_prompt import (
            NO_SKILLS_AVAILABLE,
            build_skills_section,
        )
        from types import SimpleNamespace

        with patch.dict(os.environ, {"CLAWD_SKILLS_DIR": str(self.root / "none"), "HOME": str(self.root)}):
            ctx = SimpleNamespace(cwd=self.root, workspace_root=self.root)
            # Opencode's explicit empty state: a sentence beats an absent section
            # for small models.
            self.assertEqual(build_skills_section(ctx), NO_SKILLS_AVAILABLE)
            self.assertEqual(NO_SKILLS_AVAILABLE, "No skills are currently available.")

    def test_unknown_skill_error_lists_available_names(self) -> None:
        with patch.dict(os.environ, {"CLAWD_SKILLS_DIR": str(self.root / "none"), "HOME": str(self.root)}):
            self._write_skill("alpha", "description: Use when doing A")
            ctx = ToolContext(workspace_root=self.root)
            result = SkillTool().run({"skill": "nope"}, ctx)
        self.assertTrue(result.is_error)
        self.assertIn("alpha", result.output["error"])

    def test_skill_output_lists_sibling_files(self) -> None:
        with patch.dict(os.environ, {"CLAWD_SKILLS_DIR": str(self.root / "none"), "HOME": str(self.root)}):
            self._write_skill("alpha", "description: Use when doing A")
            extra = self.root / ".clawd" / "skills" / "alpha" / "helper.py"
            extra.write_text("x = 1\n", encoding="utf-8")
            ctx = ToolContext(workspace_root=self.root)
            result = SkillTool().run({"skill": "alpha"}, ctx)
        self.assertTrue(result.output["success"])
        self.assertIn("helper.py", result.output["prompt"])
        self.assertIn("Base directory for this skill:", result.output["prompt"])


class BasePromptTests(unittest.TestCase):
    def test_default_style_is_the_full_coding_agent_prompt(self) -> None:
        from src.outputStyles.styles import BUILTIN_OUTPUT_STYLES
        from src.tool_system.system_prompt import build_tool_guidance

        prompt = BUILTIN_OUTPUT_STYLES["default"].prompt
        for needle in ("Read a file before you Edit", "TodoWrite", "never commit"):
            self.assertIn(needle.lower(), prompt.lower())
        # Tool-specific guidance moved out of the static core (prompt diet):
        # "Skill" renders only when the Skill tool is registered.
        self.assertNotIn("skill", prompt.lower())
        self.assertIn("skill", build_tool_guidance({"skill"}).lower())

    def test_system_reminder_convention_present(self) -> None:
        from src.outputStyles.styles import BUILTIN_OUTPUT_STYLES

        self.assertIn("<system-reminder>", BUILTIN_OUTPUT_STYLES["default"].prompt)

    def test_explanatory_style_extends_default(self) -> None:
        from src.outputStyles.styles import BUILTIN_OUTPUT_STYLES

        self.assertTrue(
            BUILTIN_OUTPUT_STYLES["explanatory"].prompt.startswith(BUILTIN_OUTPUT_STYLES["default"].prompt)
        )

    def test_prompt_only_names_registered_tools(self) -> None:
        import re

        from src.outputStyles.base_prompt import BASE_SYSTEM_PROMPT
        from src.tool_system.defaults import build_default_registry

        registered = {spec.name for spec in build_default_registry().list_specs()}
        mentioned = set(re.findall(r"\b(Read|Edit|Write|Bash|Grep|Glob|TodoWrite|Skill|WebSearch|WebFetch|AskUserQuestion)\b", BASE_SYSTEM_PROMPT))
        self.assertTrue(mentioned)
        self.assertLessEqual(mentioned, registered)
