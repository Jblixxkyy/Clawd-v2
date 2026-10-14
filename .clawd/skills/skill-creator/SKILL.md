---
name: "skill_creator"
description: "Create or update a Clawd-Code skill: its description, structure, instructions, and supporting files. Use when the user asks to save a workflow as a reusable skill, or to author a skill from scratch."
---

# Skill Creator

## Purpose
Create or update a skill that is easy to trigger, concise to load, and backed by references or helper code only when they materially improve reliability.

## How skills work here
- A skill is a folder containing `SKILL.md` with `name:` / `description:` frontmatter.
- User skills live in `~/.clawd/skills/<name>/`; project skills in `<project>/.clawd/skills/<name>/`. Either location is discovered automatically — no restart needed, a skill is usable the moment its folder exists.
- Invoking a skill injects its `SKILL.md` as a prompt and the agent follows it with its normal tools. Helper scripts in the skill folder are run via the Bash tool; reference `${CLAUDE_SKILL_DIR}` inside `SKILL.md` and it resolves to the skill's own folder.
- Optional frontmatter: `allowed_tools` (restrict which tools the agent may use while the skill is active), `disable_model_invocation: true` (documentation-only skill).

## Workflow
1. Clarify the capability, likely trigger phrases, and where the skill should live (user `~/.clawd/skills/<name>/` vs project `.clawd/skills/<name>/`).
2. Choose a narrow scope. Prefer one clear job per skill. Split unrelated jobs into separate skills.
3. Plan the file layout before writing:
   - Keep only the operational core in `SKILL.md`.
   - Put bulky docs, examples, or schemas in `references/`.
   - Put helper scripts in `bin/` — prefer checked-in helpers over long auth/setup prose.
4. Write frontmatter. Required: `name`, `description`. The description is what the agent matches against, so say what the skill does AND when to use it.
5. Write the body:
   - Tool-backed skills: `Purpose`, `Tooling`, `Auth`, `Operating Rules`
   - Workflow-only skills: `Purpose`, `Workflow`, `Output Contract`, `Operating Rules`
   - Keep examples short and directly executable. Write for the agent that will read this, not for a human.
6. Trim aggressively. Move anything not needed on every trigger into `references/`.
7. Sanity-check the result:
   - The description says what the skill does and when it should trigger.
   - The body tells the agent what to do next, not the whole domain.
   - Commands, paths, and auth flows match real behavior on this machine — never invent binaries or paths.
8. Verify: list it with the `/skills` command or ask the agent what skills it has, then invoke it once on a real task.

## Operating Rules
1. Preserve working commands and repo conventions; do not invent binaries, paths, or auth flows.
2. Prefer minimal frontmatter and on-demand loading. Only add metadata the skill actually needs.
3. Give auth its own section instead of burying it in operating rules.
4. Never put secrets, API keys, or tokens in a skill. Reference where the credential lives (env var, config file, keychain) and how the agent should read it.
5. If the skill wraps work you just did with the user, distill the steps — don't paste the whole transcript.
