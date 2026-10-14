# 003 — Quiet tool-call display

What: Josh called the terminal renderer "hella noisy" — every tool call draws TWO bordered panels (a "running…" card + a "done" card); one Read = ~8 lines of boxes. Replace with ChatGPT-style quiet: one dim line per call (`✓ Read src/foo.py`), red-green diffs only for Write/Edit where they matter, red only on errors.

Tests: renderer unit tests green; PTY before/after screenshot as proof (save under ~/workspace/your_files/).
