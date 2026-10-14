# 011 — WebFetch offset + unread reporting

## What
When a fetched page is truncated, report how much was unread and accept an `offset` to read on — instead of a dead-end truncation notice.

## Why
Claude Code 2.1.290 fixed WebFetch silently dropping page text past 100k chars: it now states how much was unread and takes an `offset` to continue reading. Our `WebFetchTool` (`src/tool_system/tools/web_fetch.py`) truncates at 100k chars with `"... [truncated] ..."` and offers no way to read the rest — the model either guesses or re-fetches (same truncation). Small, concrete, model-agnostic.

## Suggested approach
- Add optional `offset` (chars) parameter to the WebFetch input schema.
- Cache the fetched text per URL in the session (bounded cache, e.g. last 5 URLs, 1MB cap) so offset reads don't re-fetch.
- Truncation message becomes: `"... [truncated: showing 0–100000 of 342118 chars; pass offset=100000 to continue]"`.

## Expected tests
- Unit test with a stubbed 250k-char page: first call truncates with accurate counts; `offset=100000` returns the next chunk.
- Test cache hit doesn't re-fetch (count stub calls).
- Test offset beyond end returns empty with a clean message.
