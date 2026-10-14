---
description: "Customize Clawd by creating, installing, and managing skills, plugins, agents, and configurations"
user-invocable: true
disable-model-invocation: false
when_to_use: |
  Use this skill when the user wants to:
  - Create a new custom skill for Clawd
  - Install a skill from a remote source or local directory
  - Manage existing skills (list, update, remove)
  - Create agent configurations
  - Set up MCP servers
  - Configure provider settings and API keys
  - Customize the web UI or REPL behavior
version: "1.0.0"
model: null
allowed-tools:
  - Glob
  - Grep
  - Read
  - Write
  - Edit
  - Bash
  - WebFetch
  - WebSearch
  - Task
arguments:
  - action: "Action to perform: create, install, list, remove, info, configure"
  - name: "Skill/agent/config name"
  - source: "Source URL, path, or git repo for install"
  - description: "Description for new skill"
  - allowed-tools: "Comma-separated list of tools for new skill"
arguments_optional: true
context: "inline"
---

# Customize Clawd

This skill helps you customize Clawd by creating, installing, and managing skills, plugins, agents, and configurations.

## Quick Start

```bash
# Create a new skill
/customize-clawd create my-skill "My custom skill" --tools "Read,Write,Bash"

# Install a skill from GitHub
/customize-clawd install https://github.com/user/skill-repo

# List all installed skills
/customize-clawd list

# Configure providers/API keys
/customize-clawd configure provider
```

## Actions

### `create` - Create a new skill
Creates a new skill directory with SKILL.md template.

**Arguments:**
- `name` (required): Skill name (kebab-case)
- `description` (required): Skill description
- `allowed-tools` (optional): Comma-separated list of allowed tools
- `when-to-use` (optional): When this skill should be invoked
- `arguments` (optional): Comma-separated argument names

**Example:**
```
/customize-clawd create my-analyzer "Analyzes code patterns" --tools "Read,Grep,Glob" --when-to-use "Use when user wants code analysis" --arguments "path,pattern"
```

### `install` - Install a skill
Installs a skill from a remote source (GitHub, ZIP, tar.gz) or local directory.

**Arguments:**
- `source` (required): GitHub URL (user/repo), direct ZIP/tar.gz URL, or local path
- `name` (optional): Override skill name
- `force` (optional): Force reinstall if exists

**Examples:**
```
/customize-clawd install user/repo
/customize-clawd install https://github.com/user/skill/archive/main.zip
/customize-clawd install ./local/skill --name my-skill
```

### `list` - List installed skills
Shows all installed skills with their source and version.

**Example:**
```
/customize-clawd list
```

### `remove` - Remove a skill
Removes an installed skill.

**Arguments:**
- `name` (required): Skill name to remove

**Example:**
```
/customize-clawd remove old-skill
```

### `info` - Show skill details
Displays detailed information about a skill.

**Arguments:**
- `name` (required): Skill name

**Example:**
```
/customize-clawd info my-skill
```

### `configure` - Configure Clawd settings
Interactive configuration for providers, API keys, MCP servers, etc.

**Sub-commands:**
- `provider` - Configure API providers and keys
- `mcp` - Configure MCP servers
- `agent` - Create/edit agent configurations
- `web` - Configure web UI settings
- `repl` - Configure REPL settings

**Examples:**
```
/customize-clawd configure provider
/customize-clawd configure mcp
/customize-clawd configure agent
```

## Skill Structure

A Clawd skill is a directory containing a `SKILL.md` file:

```
my-skill/
├── SKILL.md          # Required: Frontmatter + markdown content
├── scripts/          # Optional: Helper scripts
└── templates/        # Optional: Template files
```

### SKILL.md Frontmatter

```yaml
---
description: "Brief description of what this skill does"
user-invocable: true          # Can user invoke via /skill-name?
disable-model-invocation: false  # Should model auto-invoke?
when_to_use: "When to use this skill"
version: "1.0.0"
model: "claude-sonnet-4"      # Optional: preferred model
allowed-tools:                # Optional: restrict tools
  - Read
  - Write
  - Bash
arguments:                    # Optional: expected arguments
  - name
  - path
context: "inline"             # or "fork"
agent: null                   # Optional: specific agent
effort: null                  # Optional: thinking effort
paths: null                   # Optional: file path restrictions
---
```

## Managed Skills Directory

Built-in skills are stored in the managed skills directory (set via `CLAWD_MANAGED_SKILLS_DIR`):

```
managed_skills/
├── customize-clawd/    # This skill
├── browser-automation/
├── game-development/
└── ...
```

## Related Commands

- `/skill-create` - Create a new skill (shorthand)
- `/skill-install` - Install a skill (shorthand)
- `/skill-list` - List skills (shorthand)
- `/skill-remove` - Remove a skill (shorthand)
- `/settings` - Open settings UI (web)
- `/mcp` - MCP server management