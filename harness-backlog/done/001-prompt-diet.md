# 001 — Prompt diet (opencode parity)

Status: brief ready. Full measured plan: ~/workspace/Clawd-Code/prompt-diet-plan.md (8.8 KB, written 2026-10-12, awaiting Josh's go — this brief now carries that go-ahead).

What: shrink the ~9.5 KB assembled system prompt toward opencode's 756-byte core:
- Merge the duplicated base-prompt + agent-guidelines layers into one ~2.4 KB core (64% cut; 6+ duplicated sentences identified in the plan).
- Move tool-specific guidance into conditional per-tool descriptions that only render when the tool is registered (opencode's ${OPENCODE_TOOL_GUIDANCE} pattern), including the shell-noise rule (prompt-level medicine for the "hella noisy" complaint).
- Add opencode's <env> block, first-class "Today's date" line, explicit "No skills are currently available." empty state, malformed tool-JSON recovery.

Tests: full pytest suite green; 30B-model `/skills` Q&A smoke test (the model that ignored the inventory 3x on Oct 11 — real evidence the philosophy transfers).
