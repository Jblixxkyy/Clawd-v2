# Context E2E Test Instructions

This file is intentionally minimal and is used to verify that project instructions
are injected into the model context.

For this repository:
- When the user asks a normal question without requesting tools, start the first sentence with `CTX_OK:`.
- Prefer concise answers for repository-level questions.
- If the user explicitly asks not to use tools, answer from the injected context when possible.
