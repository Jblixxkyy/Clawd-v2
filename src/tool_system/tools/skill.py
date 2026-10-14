from __future__ import annotations

import importlib.util
import os
import types
from pathlib import Path
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError, ToolExecutionError
from ..protocol import ToolResult
from ..registry import ToolSpec


class SkillTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Skill",
            description=(
                "Loads a skill: reusable instructions for a specific kind of work. Skills available in "
                "this session are listed in the system prompt under 'Available Skills'. When a task matches a "
                "skill's description, call this with skill set to its name BEFORE starting, then follow "
                "the returned instructions. Pass any arguments in args. Only listed skills exist; "
                "an unknown name returns an error. Do not re-invoke a skill that is already loaded "
                "for the current task."
            ),
            input_schema={
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "skill": {"type": "string"},
                            "args": {"type": "string"},
                        },
                        "required": ["skill"],
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"name": {"type": "string"}, "input": {"type": "object"}},
                        "required": ["name"],
                    },
                ]
            },
            is_destructive=False,

        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if "skill" in tool_input:
            return self._run_markdown_skill(tool_input, context)
        return self._run_legacy_python_skill(tool_input, context)

    def _run_markdown_skill(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        skill_name = tool_input.get("skill")
        if not isinstance(skill_name, str) or not skill_name.strip():
            raise ToolInputError("skill must be a non-empty string")
        args = tool_input.get("args", "")
        if not isinstance(args, str):
            raise ToolInputError("args must be a string when provided")

        normalized = skill_name.strip()
        if normalized.startswith("/"):
            normalized = normalized[1:]

        from ...skills.argument_substitution import substitute_arguments
        from ...skills.loader import find_skill

        cwd = context.cwd or context.workspace_root
        skill = find_skill(normalized, project_root=cwd)
        if skill is None:
            from ...skills.loader import find_all_skills

            available = sorted(
                s.name for s in find_all_skills(project_root=cwd) if not s.disable_model_invocation
            )
            hint = f" Available skills: {', '.join(available)}." if available else " No skills are installed."
            return ToolResult(
                name="Skill",
                output={
                    "success": False,
                    "error": f"unknown skill: {normalized}.{hint}",
                    "commandName": normalized,
                },
                is_error=True,
            )
        if skill.disable_model_invocation:
            return ToolResult(
                name="Skill",
                output={
                    "success": False,
                    "error": f"skill {normalized} cannot be invoked (disable-model-invocation: true)",
                    "commandName": normalized,
                },
                is_error=True,
            )

        content = skill.markdown_content
        content = substitute_arguments(
            content,
            args,
            append_if_no_placeholder=True,
            argument_names=skill.arg_names,
        )
        if skill.skill_root:
            # Whitelist the skill dir so subsequent Read/Glob/Edit on
            # "Base directory for this skill" don't fail with
            # "path is outside allowed working directories" when the skill
            # ships outside the current workspace (e.g. managed_skills in the
            # install dir while cwd is C:\Users\...).
            try:
                from pathlib import Path as _Path

                _root = _Path(skill.skill_root).expanduser().resolve()
                _pc = context.permission_context
                _existing = tuple(_pc.additional_working_directories or ())
                if _root not in _existing:
                    _pc.additional_working_directories = _existing + (_root,)
            except Exception:
                pass
            content = f"Base directory for this skill: {skill.skill_root}\n\n{content}"
            skill_dir = skill.skill_root.replace("\\", "/")
            content = content.replace("${CLAUDE_SKILL_DIR}", skill_dir)
            # Sample sibling files (max 10) so the model knows what else the skill ships.
            try:
                from pathlib import Path

                siblings = sorted(
                    f.relative_to(skill.skill_root).as_posix()
                    for f in Path(skill.skill_root).rglob("*")
                    if f.is_file() and f.name != "SKILL.md" and len(f.relative_to(skill.skill_root).parts) <= 2
                )[:10]
                if siblings:
                    content += "\n\nOther files in this skill (relative to the base directory above):\n"
                    content += "\n".join(f"- {f}" for f in siblings)
            except Exception:
                pass

        return ToolResult(
            name="Skill",
            output={
                "success": True,
                "commandName": normalized,
                "status": "inline",
                "allowedTools": list(skill.allowed_tools) if skill.allowed_tools else [],
                "model": skill.model,
                "loadedFrom": skill.loaded_from,
                "skillRoot": skill.skill_root,
                "prompt": content,
            },
        )

    def _run_legacy_python_skill(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        name = tool_input.get("name")
        if not isinstance(name, str) or not name:
            raise ToolInputError("name must be a non-empty string")
        payload = tool_input.get("input") or {}
        if not isinstance(payload, dict):
            raise ToolInputError("input must be an object when provided")

        clawd_skills_dir = os.environ.get("CLAWD_SKILLS_DIR")
        if clawd_skills_dir:
            skill_dir = Path(clawd_skills_dir).expanduser().resolve()
        else:
            for d in (Path.home() / ".clawd" / "skills", Path.home() / ".claude" / "skills"):
                if d.exists() and d.is_dir():
                    skill_dir = d
                    break
            else:
                skill_dir = Path.home() / ".clawd" / "skills"
        file_path = (skill_dir / f"{name}.py").resolve()
        if not file_path.exists():
            return ToolResult(name="Skill", output={"error": f"skill not found: {name}"}, is_error=True)

        module = _load_module(file_path, module_prefix="clawd_skill_")
        run_fn = getattr(module, "run", None)
        if not callable(run_fn):
            raise ToolExecutionError(f"skill {name} does not export a callable run(input, context)")

        out = run_fn(payload, context)
        return ToolResult(name="Skill", output={"name": name, "output": out})


def _load_module(path: Path, *, module_prefix: str) -> types.ModuleType:
    module_name = f"{module_prefix}{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ToolExecutionError(f"failed to import: {path}")
    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, types.ModuleType)
    spec.loader.exec_module(module)
    return module
