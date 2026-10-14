"""Prompt-diet verification (opencode parity, brief 001).

Guards the merged-core architecture:
- static core (BASE_SYSTEM_PROMPT) stays under 3 KB
- no cross-layer duplicate sentences in the assembled prompt
- per-tool guidance renders only when the tool is registered
- Today's date line + <env> block + explicit no-skills sentence
"""

import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from src.outputStyles.base_prompt import BASE_SYSTEM_PROMPT
from src.tool_system.agent_loop import _build_effective_system_prompt
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.system_prompt import (
    NO_SKILLS_AVAILABLE,
    build_env_block,
    build_skills_section,
    build_tool_guidance,
    todays_date_line,
)


class TestPromptDiet(TestCase):
    def _ctx(self, tmp_path: Path) -> ToolContext:
        ctx = ToolContext(workspace_root=tmp_path)
        ctx.cwd = tmp_path
        return ctx

    # --- core byte budget --------------------------------------------------
    def test_core_under_3kb(self) -> None:
        self.assertLess(
            len(BASE_SYSTEM_PROMPT.encode("utf-8")),
            3 * 1024,
            "merged static core must stay under 3 KB",
        )

    def test_core_merged_single_layer(self) -> None:
        # The old duplicated two-layer stack is gone: no legacy section headers.
        for header in (
            "## Agent Capabilities",
            "## Tool-Use Judgment",
            "## Working Discipline",
            "## Ask vs Act",
            "## Error Recovery",
        ):
            self.assertNotIn(header, BASE_SYSTEM_PROMPT)

    # --- no cross-layer duplicate sentences -------------------------------
    def test_no_duplicate_sentences_assembled(self) -> None:
        import tempfile

        root = Path(tempfile.mkdtemp())
        ctx = self._ctx(root)
        names = {s.name for s in build_default_registry().list_specs()}
        with patch.dict(os.environ, {
            "CLAWD_SKILLS_DIR": str(root / "none"),
            "HOME": str(root / "none"),
        }):
            prompt = _build_effective_system_prompt(
                BASE_SYSTEM_PROMPT, ctx, registered_tools=names
            )
        lowered = prompt.lower()
        # The 6+ sentences the old base + guidelines layers duplicated.
        # Each must appear at most once now (in exactly one layer).
        once_or_absent = [
            "call the skill tool first",
            "several independent tool calls in one turn",
            "for tasks with three or more steps",
            "use websearch/webfetch for current or external information",
            "ask one short question",
            "never ask the user for permission",
            "read a file before you edit or overwrite it",
        ]
        for sentence in once_or_absent:
            self.assertLessEqual(
                lowered.count(sentence),
                1,
                f"duplicate sentence in assembled prompt: {sentence!r}",
            )

    # --- conditional tool guidance ----------------------------------------
    def test_bash_noise_rule_only_when_bash_registered(self) -> None:
        with_bash = build_tool_guidance({"bash", "read"})
        self.assertIn("echo", with_bash)
        without_bash = build_tool_guidance({"read"})
        self.assertNotIn("echo", without_bash)

    def test_guidance_dedupes_identical_lines(self) -> None:
        # WebSearch and WebFetch share one guidance sentence.
        guidance = build_tool_guidance({"websearch", "webfetch"})
        self.assertEqual(
            guidance.lower().count(
                "use websearch/webfetch for current or external information"
            ),
            1,
        )

    def test_guidance_empty_when_nothing_registered(self) -> None:
        self.assertEqual(build_tool_guidance(set()), "")
        self.assertEqual(build_tool_guidance(None), "")

    def test_script_mode_line_conditional(self) -> None:
        guidance = build_tool_guidance({"bash"}, script_mode=True)
        self.assertIn("auto-deny", guidance)
        self.assertNotIn("auto-deny", build_tool_guidance({"bash"}))

    def test_enterplanmode_guidance_conditional(self) -> None:
        self.assertIn("EnterPlanMode", build_tool_guidance({"enterplanmode"}))
        self.assertNotIn("EnterPlanMode", build_tool_guidance({"read"}))

    # --- date + env ---------------------------------------------------------
    def test_date_line_at_head(self) -> None:
        import tempfile

        root = Path(tempfile.mkdtemp())
        ctx = self._ctx(root)
        prompt = _build_effective_system_prompt(
            BASE_SYSTEM_PROMPT, ctx, registered_tools={"read"}
        )
        self.assertTrue(
            prompt.startswith(todays_date_line()),
            "assembled prompt must start with Today's date line",
        )

    def test_env_block_rendered(self) -> None:
        import tempfile

        root = Path(tempfile.mkdtemp())
        ctx = self._ctx(root)
        block = build_env_block(ctx)
        self.assertIn("<env>", block)
        self.assertIn("Working directory:", block)
        self.assertIn("Platform:", block)
        self.assertIn("Is directory a git repo:", block)

    # --- skills section ------------------------------------------------------
    def test_explicit_no_skills_state(self) -> None:
        import tempfile

        root = Path(tempfile.mkdtemp())
        with patch.dict(os.environ, {
            "CLAWD_SKILLS_DIR": str(root / "none"),
            "HOME": str(root / "none"),
        }):
            ctx = self._ctx(root)
            text = build_skills_section(ctx)
            self.assertEqual(text, NO_SKILLS_AVAILABLE)
            self.assertIn(
                NO_SKILLS_AVAILABLE,
                _build_effective_system_prompt(
                    BASE_SYSTEM_PROMPT, ctx, registered_tools={"read"}
                ),
            )

    def test_few_shot_is_shape_not_relist(self) -> None:
        import tempfile
        from types import SimpleNamespace

        root = Path(tempfile.mkdtemp())
        skills_dir = root / "skills"
        skills_dir.mkdir()
        (skills_dir / "alpha").mkdir()
        (skills_dir / "alpha" / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: Use when doing A\n---\nDo A."
        )
        env = {
            "CLAWD_SKILLS_DIR": str(skills_dir),
            "HOME": str(root / "none"),
        }
        with patch.dict(os.environ, env):
            ctx = SimpleNamespace(cwd=root, workspace_root=root)
            text = build_skills_section(ctx)
        # Inventory lists the real skill...
        self.assertIn("- /alpha:", text)
        # ...but the few-shot example uses generic placeholders, not a re-list.
        self.assertIn("Example:", text)
        example_tail = text.split("Example:")[1]
        self.assertNotIn("- /alpha:", example_tail)
        self.assertIn("/deploy", example_tail)
