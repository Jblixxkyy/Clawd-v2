# 013 — MCP server inventory awareness

## What
Inject the connected MCP servers and their tool/resource inventories into the model context at session start (and refresh on change), instead of the current one-line generic MCP tool description.

## Why
The model gets an `MCP` tool — "Call a tool exposed by a connected MCP server." — with zero inventory: no server list, no per-server tool list, nothing in the system prompt (`src/tool_system/system_prompt.py` has no MCP mention at all). The client protocol has `list_tools()`, but nothing model-facing exposes it, so the model cannot discover what a server offers; it must already know exact server and tool names. Same blindness class as skills pre-009, and the Oct 11 Q&A test proved models ignore capabilities they can't see. The config side is fine (`~/.clawd/mcp.json`, `/mcp add <name> <command>`, doctor check in `builtins.py:doctor_check_mcp_servers`); the model side is blind.

## Suggested approach
- At session start (context build / SessionStart hook), enumerate `context.mcp_clients`; call `list_tools()` per server; inject a compact block: `server → tools (one-line descriptions each)` plus resource URIs where cheap.
- Keep it prompt-diet-friendly (brief 001): only inject when ≥1 server is connected; terse `server.tool — description` lines, no boilerplate.
- Refresh the inventory when servers change mid-session (`/mcp add`/`remove`).
- Unknown server/tool names in MCP calls → error that points at the inventory (same pattern as the Skill tool's unknown-name error).
- Later (not this brief): `$mcp:server` user-side mentions, as noted in 009.

## Expected tests
- Inventory block present when ≥1 server connected; lists each tool with its description.
- Empty state (no servers): no block injected, no error, MCP tool still present.
- MCP call with unknown server/tool → helpful error referencing available names.
- Adding a server mid-session refreshes the inventory for subsequent turns.
- Full test suite green before shipping.
