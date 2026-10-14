# 002 — Delta instruction system

What: opencode's biggest architectural win we haven't built. Instead of resending the full system prompt every turn, send only the changed instructions per turn (delta updates). Study opencode's implementation in ~/workspace/opencode-ref (packages/core/src/session/) and adapt it to our agent loop in src/tool_system/agent_loop.py.

Why: token savings compound every turn; smaller per-turn prompts also behave better on small models.

Tests: suite green + a live two-turn session proving turn 2 carries only the delta, with token counts logged as evidence.
