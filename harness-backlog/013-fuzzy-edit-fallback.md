# 013 — Fuzzy edit fallback chain (opencode parity)

Status: **implemented + tested 2026-10-14.** Full pytest suite green (baseline
49 pre-existing failures unchanged — verified by swapping HEAD versions back in
and diffing the failure sets; zero new failures, one flaky file-watcher test
passes 3/3 in isolation).

Shipped:
- `src/tool_system/tools/edit_fuzzy.py` — the chain (new, ~300 lines, stdlib only)
- `src/tool_system/tools/edit.py` — `_resolve_replacement` / `_apply_spans`, span-based
  splicing for both single- and multi-hunk mode, `match` + `notice` result fields
- `src/tool_system/system_prompt.py` — one-sentence edit guidance
- `tests/test_edit_fuzzy.py` — 18 tests

Deviations from this brief (design decisions made during implementation):
1. **`indent` fires less often than assumed.** A dedented `old_string` is still a
   plain *substring* of the indented file, so the exact path wins and already
   produces the right result. `indent` covers the real misses: tab-vs-space,
   over-indented, and multi-line indent drift.
2. **End-anchoring added.** When `old` ends with `\n` the match must also end a
   line — otherwise `old="foo\n"` would match the `foo` inside `foo bar`.
3. **Whitespace rule.** The span absorbs trailing whitespace only when `old`
   itself ended in whitespace/newline; otherwise the file's own tail survives.
4. **`replace_all` with fuzzy is allowed** across N spans — they are one
   strategy by construction (the chain stops at the first strategy that hits),
   so the "mixed-strategy bulk replace" the brief refused cannot occur.

Source: opencode `packages/opencode/src/tool/edit.ts` — 9-stage match chain (exact →
line-trimmed → whitespace-normalized → indentation-flexible → escape-normalized →
block-anchor + Levenshtein).

## The problem (measured)

`src/tool_system/tools/edit.py` is exact-match only:

- `_validate_replacement()` → `content.count(old)` (line 94). Zero matches raises
  immediately with `_nearest_match_hints()` — which *names the right line* but cannot
  fix it. The model then burns a turn re-reading the file and retrying, or gives up.
- Failure modes that hit constantly with small models:
  1. **Trailing whitespace / final newline** — file has `foo  \n`, model sends `foo\n`.
  2. **Indentation drift** — model sends dedented text for an indented block (or vice
     versa; tabs vs spaces on the Windows path).
  3. **Internal whitespace collapse** — model squeezes `a    b` to `a b`.
  4. **Near-miss quoting/escaping** — model paraphrases 1-2 chars of a long line.
- Current cost per failure: 1 tool error + ≥1 re-read + ≥1 retry, and on non-unique
  near-misses the model historically abandons the edit entirely.

## What: a confidence-gated fallback chain, exact-first

New module `src/tool_system/tools/edit_fuzzy.py` (pure functions, no deps beyond
stdlib `difflib`), called from `_validate_replacement()` **only when `count == 0`**.
Exact matches keep byte-identical behavior — no regression path.

Chain (stops at first strategy with a unique match):

| # | Strategy | Handles | Match function |
|---|---|---|---|
| 0 | `exact` | (existing) | `content.count(old)` |
| 1 | `line-trim` | trailing WS, missing final newline | `.rstrip()` each line of both sides |
| 2 | `indent` | leading-WS / tab-vs-space drift | `textwrap.dedent`-style strip of leading WS per line |
| 3 | `collapse-ws` | run-length whitespace differences | `re.sub(r"\s+", " ", ...)` per line, both sides |
| 4 | `near` | 1-2 char drift in a long block | sliding-window `SequenceMatcher.ratio()` over blocks of the same line count, **cutoff 0.90** and min(old) ≥ 24 chars |

Every strategy returns the **actual matched span** in the file. Replacement is applied
to that span, never to a re-normalized reconstruction (so the file keeps its real
indentation/whitespace — we fix the *match*, not the *content*).

### Safety rules (non-negotiable)

- **Never fuzzy on short needles.** Strategies 1-3 require `len(old) >= 12` and
  `>= 1` non-blank line of substance; strategy 4 requires `>= 24` chars. Short needles
  (`"foo"`) keep failing exactly as today.
- **Ambiguity → error, never a guess.** If a strategy matches ≥2 *distinct* spans,
  raise `ToolInputError` listing both line numbers (reuse `_format_lines`). A fuzzy
  match is only auto-applied when it is unique across the whole chain prefix.
- **`expected_occurrences` / `replace_all` still gate it**: counts are computed on the
  matched spans; `replace_all` with fuzzy must match ≥2 spans via the *same* strategy,
  else error (mixed-strategy bulk replace is refused).
- **Multi-hunk stays atomic**: fuzzy resolution happens in Phase 1 (validate against
  ORIGINAL); any hunk failing leaves the file untouched.
- **`## Safety` unaffected**: fuzzy runs *after* `check_permissions` and after the
  read-first gate (`_load_for_edit`), inside `run()` only.

### Reporting (so the model learns instead of silently drifting)

Add to the Edit result output:

```
"match": {"kind": "exact" | "fuzzy", "strategy": "exact|line-trim|indent|collapse-ws|near",
          "foundLine": 42, "found": "<the exact text matched>"}
```

When `kind == "fuzzy"`, prepend one line to the human-readable result:
`Matched fuzzily (indent, line 42) — file text was used as-is.`

Optional `_TOOL_GUIDANCE["edit"]` addition (1 sentence): fuzzy fallback exists, but
exact old_string is still preferred — do not advertise it as license to be sloppy.

## Out of scope (deliberately)

- The other 4 opencode strategies (escape-normalized, block-anchor, full Levenshtein
  distance search) — add only if the 4 above measurably miss.
- Post-edit LSP diagnostics, output truncation, incremental instruction discovery
  (separate backlog items — 014+).
- Changing `_nearest_match_hints()` — still used for the zero-fuzzy-match path.

## Tests — `tests/test_edit_fuzzy.py`

1. trailing-whitespace old_string → succeeds, `strategy == "line-trim"`, file text
   preserved (indentation unchanged).
2. dedented old_string against indented block → `strategy == "indent"`, output keeps
   original indentation.
3. collapsed internal whitespace → `strategy == "collapse-ws"`.
4. 1-char drift on a ≥24-char line → `strategy == "near"`; 1-char drift on a short
   line → still `ToolInputError`.
5. **ambiguity**: fuzzy match appears twice → raises, file untouched.
6. **exact wins**: content that matches exactly AND fuzzily elsewhere → exact behavior,
   no fuzzy flag.
7. short needle (`"foo"` in `"foo\nbar"`) → unchanged exact error message (guards
   existing `test_edit_hardening.py` expectations).
8. multi-hunk: hunk 2 fails fuzzy → file byte-identical to before.
9. `replace_all` refused when fuzzy spans use differing strategies.
10. existing full pytest suite green (esp. `test_edit_hardening.py`,
    `test_claude_code_tool_parity.py`, `test_tool_system_tools.py`).

## Acceptance

- Full pytest suite green.
- Measured: the 4 built-in failure modes above go from (error + re-read + retry) to a
  single successful Edit.
- Zero behavior change for exact matches (asserted by test 6) and for short needles
  (test 7).
