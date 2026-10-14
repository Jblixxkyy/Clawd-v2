---
name: "install-skill"
description: "Download and install a skill from GitHub (or another source) into the local skills folders. Use when the user asks to download, install, or add a new skill."
---

# Install Skill

## Purpose
Fetch a skill from a remote source (usually GitHub) and install it where Clawd-Code will discover it. Skills take effect immediately — no restart.

## Where skills live
- User skills (available everywhere): `~/.clawd/skills/<name>/`
- Project skills (this project only): `<project>/.clawd/skills/<name>/`

A skill is a folder containing a `SKILL.md` file with `name:` and `description:` frontmatter. If the folder lacks `SKILL.md`, it is not a valid skill.

## Workflow
1. Clarify which skill and where it should live (user vs project). Default to user (`~/.clawd/skills/`) unless the user says otherwise.
2. Download it. For GitHub repos, prefer a shallow clone:
   `git clone --depth 1 <repo-url> /tmp/skill-dl`
   For a single skill inside a large repo (e.g. anthropics/skills), clone then copy just that skill's folder.
3. Verify the folder contains `SKILL.md` with valid `name:` / `description:` frontmatter.
4. Copy the skill folder to the target location, e.g.:
   `cp -r /tmp/skill-dl/<skill-name> ~/.clawd/skills/`
5. Confirm it is discovered (it is usable immediately — list it back to the user).

## Operating Rules
1. Never claim you have no mechanism to install skills — this workflow IS the mechanism.
2. Do not overwrite an existing skill folder without asking the user first.
3. Never put secrets, API keys, or tokens into an installed skill.
4. Clean up the temp download dir afterwards.
