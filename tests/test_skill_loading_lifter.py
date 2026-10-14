"""030 — Skill loading as a small-model lifter: triggers, markers,
progressive disclosure, context_cmd, model-family variants, pruning.

Expected tests from the brief:
- Repo with uv.lock -> uv skill auto-activates; repo without -> it doesn't.
- Keyword trigger in the user message -> matching skill loads its body;
  non-matching skills stay as one-liners.
- context_cmd output appears in the skill's injected context.
- Token count of the skill block with 50 installed skills stays under budget.
- Full test suite green before shipping.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.skills.activation import (
    ActivationContext,
    activate_skills,
    detect_model_family,
    load_prompt_variant,
    marker_present,
    prune_catalog,
    run_context_cmd,
    score_skill,
)
from src.skills.loader import clear_skill_registry


def _write_skill(skills_dir: Path, name: str, *, frontmatter: str, body: str) -> None:
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8")


UV_FRONTMATTER = """description: Manage Python projects with uv
triggers: uv, venv, virtualenv
markers: [uv.lock, pyproject.toml]
context_cmd: echo uv-context-marker-output"""

PDF_FRONTMATTER = """description: Convert and read PDF files
triggers: [pdf, acrobat]
markers: []"""

PLAIN_FRONTMATTER = """description: A plain skill with no triggers or markers"""


class SkillLoadingLifterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.project = self.root / "project"
        self.project.mkdir()
        self.skills_dir = self.root / "skills"
        self.skills_dir.mkdir()
        _write_skill(
            self.skills_dir, "uv-helper",
            frontmatter=UV_FRONTMATTER,
            body="# uv helper\nUse `uv sync` to install. Use `uv run` to execute.\n",
        )
        _write_skill(
            self.skills_dir, "pdf-tool",
            frontmatter=PDF_FRONTMATTER,
            body="# pdf tool\nExtract text from PDF files with this guide.\n",
        )
        _write_skill(
            self.skills_dir, "plain-skill",
            frontmatter=PLAIN_FRONTMATTER,
            body="# plain\nNothing special here.\n",
        )
    def tearDown(self) -> None:
        clear_skill_registry()
        self.tmp.cleanup()

    def _activate(self, **kw) -> "object":
        # user_skills_dir pins the loader to exactly this dir (the loader
        # otherwise also scans ~/.clawd/skills and ~/.claude/skills).
        ctx = ActivationContext(
            project_root=self.project, user_skills_dir=self.skills_dir, **kw
        )
        return activate_skills(ctx)

    # -- markers -----------------------------------------------------------
    def test_uv_lock_marker_activates_uv_skill(self) -> None:
        (self.project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        act = self._activate(text="")
        names = [s.name for s in act.active]
        self.assertIn("uv-helper", names)
        score, reasons = act.scores["uv-helper"]
        self.assertGreater(score, 0)
        self.assertTrue(any(r.startswith("marker:") for r in reasons))

    def test_no_uv_lock_no_activation(self) -> None:
        act = self._activate(text="")
        names = [s.name for s in act.active]
        self.assertNotIn("uv-helper", names)
        self.assertEqual(act.scores["uv-helper"][0], 0)

    def test_marker_bare_extension(self) -> None:
        (self.project / "deploy.ps1").write_text("Write-Host hi\n", encoding="utf-8")
        _write_skill(
            self.skills_dir, "ps-helper",
            frontmatter="description: PowerShell helper\nmarkers: [.ps1]",
            body="# ps\nPowerShell tips.\n",
        )
        act = self._activate(text="")
        self.assertIn("ps-helper", [s.name for s in act.active])

    def test_marker_present_unit(self) -> None:
        (self.project / "uv.lock").write_text("x", encoding="utf-8")
        self.assertTrue(marker_present("uv.lock", self.project))
        self.assertFalse(marker_present("package.json", self.project))
        self.assertFalse(marker_present("", self.project))

    # -- trigger keywords ---------------------------------------------------
    def test_trigger_keyword_loads_body_others_stay_one_liners(self) -> None:
        act = self._activate(text="please convert this pdf to text", event="user_message")
        names = [s.name for s in act.active]
        self.assertIn("pdf-tool", names)
        self.assertNotIn("plain-skill", names)
        rendered = act.render()
        # Activated skill: full body present.
        self.assertIn("Extract text from PDF files", rendered)
        # Non-activated skill: index line only, body absent.
        self.assertIn("- /plain-skill —", rendered)
        self.assertNotIn("Nothing special here.", rendered)

    def test_trigger_word_boundary_no_false_positive(self) -> None:
        # 'go' trigger must not fire on the word "going".
        _write_skill(
            self.skills_dir, "go-helper",
            frontmatter="description: Go helper\ntriggers: [go]",
            body="# go\nGo tips.\n",
        )
        act = self._activate(text="I am going to the store", event="user_message")
        self.assertNotIn("go-helper", [s.name for s in act.active])
        act2 = self._activate(text="help me with go modules", event="user_message")
        self.assertIn("go-helper", [s.name for s in act2.active])

    # -- context_cmd ---------------------------------------------------------
    def test_context_cmd_output_injected(self) -> None:
        (self.project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        act = self._activate(text="")
        rendered = act.render()
        self.assertIn("uv-context-marker-output", rendered)
        self.assertIn('<context-cmd name="uv-helper">', rendered)

    def test_run_context_cmd_failure_returns_none(self) -> None:
        from src.skills.model import PromptSkill

        bad = PromptSkill(
            name="bad", description="d", loaded_from="u",
            user_invocable=True, disable_model_invocation=False,
            content_length=0, is_hidden=False, skill_root=None,
            when_to_use=None, version=None, model=None, allowed_tools=[],
            arg_names=[], context=None, agent=None, effort=None, paths=None,
            markdown_content="x", context_cmd="exit 3",
        )
        ctx = ActivationContext(project_root=self.project)
        self.assertIsNone(run_context_cmd(bad, ctx))

    # -- progressive disclosure ----------------------------------------------
    def test_promote_loads_body_on_first_use(self) -> None:
        act = self._activate(text="")
        self.assertNotIn("Nothing special here.", act.render())
        self.assertTrue(act.promote("plain-skill"))
        self.assertIn("Nothing special here.", act.render())

    def test_auto_activate_off_keeps_everything_index_only(self) -> None:
        (self.project / "uv.lock").write_text("x", encoding="utf-8")
        act = self._activate(text="", auto_activate=False)
        self.assertEqual(act.active, [])
        rendered = act.render()
        self.assertIn("- /uv-helper —", rendered)
        self.assertNotIn("Use `uv sync`", rendered)

    def test_refresh_activates_on_later_event(self) -> None:
        act = self._activate(text="")
        self.assertNotIn("pdf-tool", [s.name for s in act.active])
        newly = act.refresh("user_message", "can you read this pdf for me?")
        self.assertIn("pdf-tool", newly)
        self.assertIn("Extract text from PDF files", act.render())

    # -- token budget ----------------------------------------------------------
    def test_50_skill_catalog_stays_under_budget(self) -> None:
        for i in range(47):  # 3 already exist -> 50 total
            _write_skill(
                self.skills_dir, f"filler-{i:02d}",
                frontmatter=f"description: Filler skill number {i}",
                body=("# filler\n" + "Some filler body text for token counting. " * 40),
            )
        act = self._activate(
            text="convert this pdf please", event="user_message", token_budget=8000
        )
        estimate = act.token_estimate()
        self.assertLessEqual(estimate, 8000)
        # Activated skill body present; fillers stay one-liners.
        rendered = act.render()
        self.assertIn("Extract text from PDF files", rendered)
        self.assertIn("- /filler-00 —", rendered)
        self.assertNotIn("Some filler body text", rendered)

    def test_tight_budget_demotes_lowest_score_bodies(self) -> None:
        (self.project / "uv.lock").write_text("x", encoding="utf-8")
        act = self._activate(
            text="convert this pdf please", event="user_message", token_budget=60
        )
        rendered = act.render()
        # Both bodies demoted to index-only: index lines survive, bodies gone.
        self.assertIn("- /uv-helper —", rendered)
        self.assertIn("- /pdf-tool —", rendered)
        self.assertNotIn("Use `uv sync`", rendered)
        self.assertNotIn("Extract text from PDF files", rendered)

    # -- model families + variants ----------------------------------------------
    def test_detect_model_family(self) -> None:
        self.assertEqual(detect_model_family("claude-sonnet-4-6"), "claude")
        self.assertEqual(detect_model_family("gpt-5.4"), "gpt")
        self.assertEqual(detect_model_family("gemini-2.5-pro"), "gemini")
        self.assertEqual(detect_model_family("qwen3:4b"), "qwen")
        self.assertEqual(detect_model_family(None), "default")
        self.assertEqual(detect_model_family("some-unknown-model-xyz"), "default")

    def test_prompt_variant_prepended(self) -> None:
        variant_dir = self.skills_dir / "prompt-variants"
        variant_dir.mkdir()
        (variant_dir / "qwen.md").write_text(
            "QWEN-VARIANT-MARKER: keep skill guidance terse.\n", encoding="utf-8"
        )
        ctx = ActivationContext(
            project_root=self.project,
            user_skills_dir=self.skills_dir,
            model_family="qwen",
            text="",
        )
        act = activate_skills(ctx)
        self.assertIn("QWEN-VARIANT-MARKER", act.render())

    def test_load_prompt_variant_missing_returns_none(self) -> None:
        self.assertIsNone(load_prompt_variant("nosuchfamily", [self.skills_dir]))
        self.assertIsNone(load_prompt_variant(None, [self.skills_dir]))

    # -- pruning (conservative default: none) ------------------------------------
    def test_prune_catalog_default_is_noop(self) -> None:
        tools = ["Read", "Grep", "Glob", "Bash"]
        self.assertEqual(prune_catalog(tools, "qwen"), tools)
        self.assertEqual(prune_catalog(tools, None, {"qwen": ["Grep"]}), tools)

    def test_prune_catalog_configured_family(self) -> None:
        tools = ["Read", "Grep", "Glob", "Bash"]
        pruned = prune_catalog(tools, "qwen", {"qwen": ["Grep", "Glob"]})
        self.assertEqual(pruned, ["Read", "Bash"])

    # -- scoring unit -------------------------------------------------------------
    def test_score_skill_reasons(self) -> None:
        (self.project / "uv.lock").write_text("x", encoding="utf-8")
        from src.skills.loader import find_all_skills

        skills = {
            s.name: s
            for s in find_all_skills(
                project_root=self.project, user_skills_dir=self.skills_dir
            )
        }
        ctx = ActivationContext(project_root=self.project, text="use uv sync")
        score, reasons = score_skill(skills["uv-helper"], ctx)
        self.assertGreaterEqual(score, 12)  # marker(10) + trigger(2)
        self.assertIn("marker:uv.lock", reasons)
        self.assertTrue(any(r == "trigger:uv" for r in reasons))


if __name__ == "__main__":
    unittest.main()
