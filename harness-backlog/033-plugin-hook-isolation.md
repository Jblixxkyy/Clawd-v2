# 033 — Plugin/hook isolation: never run plugin code on the host's main thread, fail loudly

## What
Run plugin/hook code in an isolated execution context (worker thread or subprocess), never on the agent loop's main thread. A plugin timeout or crash terminates only the plugin; the loop continues and the plugin is marked failed. Delete silent fallbacks: a plugin that cannot load must report a named, actionable error (exact missing bootstrap/config) surfaced in `/doctor` and session-start warnings — never a quiet skip.

## Why
From cline/cline@`789f79e4` (https://github.com/cline/cline/commit/789f79e4.patch): Cline removed the in-process plugin execution mode entirely — plugin code now only runs in worker threads (fresh import per session), never on the host's main thread. The silent fallback to in-process when no worker bootstrap shipped was deleted; every load now fails with `PluginBootstrapMissingError` naming the exact env var/path, surfaced in `cline config tools`, `cline hub plugins`, and session-start warnings. Their docs explicitly call workers "crash isolation, not a security boundary."

Qualifies as a single-harness gem, model-agnostic: a crashing plugin can no longer take down the agent loop, and failures are loud with a named cause instead of silent. Ours: hooks are shell commands (already out-of-process) but fail-open, and the plugin story (PARITY watch item "plugin-asserted permissions") has no isolation or loud-failure contract. Passes the fitness test — robustness engineering that helps every model equally.

## Suggested approach
- Give hook/plugin execution an isolated context: worker thread or subprocess per plugin with a per-plugin timeout (fail-open guard stays: hook times out → hook skipped, turn continues, plugin marked failed).
- The loop must be provably immune: a hook that segfaults/exits non-zero inside the isolated context cannot poison loop state; record the failure in the audit log (`~/.clawd/hooks-audit.log` from brief 031's builder work).
- Kill silent fallbacks: if a plugin cannot load (missing bootstrap, bad config), surface a named error with the exact missing piece in `/doctor` output and a session-start warning. Never skip quietly.
- Follow Cline's framing: document as "crash isolation, not a security boundary" so nobody mistakes it for a sandbox.

## Expected tests
- A crashing hook cannot kill the agent loop: loop continues, plugin marked failed, failure logged.
- Missing plugin bootstrap/config surfaces the exact cause in `/doctor` diagnostics (not a silent skip).
- Hook timeout kills the plugin execution, turn continues normally.
- Full test suite green before shipping.
