"""Fuzzy match fallback for the Edit tool — harness brief 013 (opencode parity).

The exact path (``content.count(old)``) is untouched: everything in this module
runs ONLY after an exact search has already found nothing. Exact matches keep
byte-identical behavior, and a fuzzy match is applied only when it is unique —
ambiguity raises instead of guessing.

Strategies, strictest first; each is cumulative over the previous:

============ ============ ==================================================
strategy     handles      what it ignores
============ ============ ==================================================
line-trim    trailing WS  spaces/tabs before a newline, one final newline
indent       indent drift leading spaces/tabs on every line
collapse-ws  WS runs      runs of spaces/tabs inside a line -> one space
near         1-2 char     line blocks with SequenceMatcher ratio >= 0.90
             drift        (needles >= 24 chars only)
============ ============ ==================================================

Needles shorter than 12 chars (24 for ``near``) never enter the chain, so short
searches such as ``"foo"`` fail exactly as they did before.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

__all__ = [
    "Match",
    "resolve_match",
    "STRATEGIES",
    "MIN_NEEDLE_LEN",
    "NEAR_MIN_NEEDLE_LEN",
    "NEAR_CUTOFF",
]

STRATEGIES = ("line-trim", "indent", "collapse-ws", "near")

#: Strategies 1-3 are skipped for needles shorter than this.
MIN_NEEDLE_LEN = 12
#: The ``near`` strategy is skipped for needles shorter than this.
NEAR_MIN_NEEDLE_LEN = 24
#: Minimum ``SequenceMatcher`` ratio for the ``near`` strategy.
NEAR_CUTOFF = 0.90

_WS = " \t"


@dataclass(frozen=True)
class Match:
    """A resolved fuzzy match.

    ``spans`` are ``(start, end)`` offsets into the original file content.
    ``adjust_indent`` is True when the model's ``old_string`` indentation does
    not match the file's, meaning ``new_string``'s leading whitespace must be
    dropped so the file keeps the indentation it already had.
    """

    strategy: str
    spans: tuple[tuple[int, int], ...]
    adjust_indent: bool = False

    @property
    def unique(self) -> bool:
        return len(self.spans) == 1


# ---------------------------------------------------------------------------
# normalisation with an offset map
# ---------------------------------------------------------------------------

def _normalize(text: str, level: int) -> tuple[str, list[int], list[int]]:
    """Normalise *text* at *level* (1=line-trim, 2=+indent, 3=+collapse).

    Returns ``(normalized, starts, ends)`` where ``starts[i]`` / ``ends[i]`` is
    the original ``[start, end)`` span that produced ``normalized[i]``, so a
    match found in the normalised string maps straight back to real file
    offsets. Groups (a collapsed whitespace run) map to a wider original span.
    """
    trim = level >= 1
    dedent = level >= 2
    collapse = level >= 3

    groups: list[tuple[int, int, str]] = []  # (start, end, char)
    i = 0
    n = len(text)
    while i <= n:
        j = text.find("\n", i)
        has_nl = j != -1
        if not has_nl:
            j = n
        a, b = i, j
        if dedent:
            while a < b and text[a] in _WS:
                a += 1
        if trim:
            while b > a and text[b - 1] in _WS:
                b -= 1
        k = a
        while k < b:
            if collapse and text[k] in _WS:
                e = k
                while e < b and text[e] in _WS:
                    e += 1
                groups.append((k, e, " "))
                k = e
            else:
                groups.append((k, k + 1, text[k]))
                k += 1
        if has_nl:
            groups.append((j, j + 1, "\n"))
        if not has_nl:
            break
        i = j + 1

    # Both sides drop one final newline: "foo\\n" and "foo" are the same block
    # for matching purposes (the file simply has no trailing newline).
    if groups and groups[-1][2] == "\n":
        groups.pop()

    norm = "".join(g[2] for g in groups)
    starts = [g[0] for g in groups]
    ends = [g[1] for g in groups]
    return norm, starts, ends


def _line_spans(text: str) -> list[tuple[int, int]]:
    """``[start, end)`` of every line in *text*, excluding the newline."""
    out: list[tuple[int, int]] = []
    start = 0
    while True:
        i = text.find("\n", start)
        if i == -1:
            if start < len(text):
                out.append((start, len(text)))
            break
        out.append((start, i))
        start = i + 1
    return out


def _line_indent(text: str, line_start: int) -> str:
    """Leading spaces/tabs of the line beginning at *line_start*."""
    i = line_start
    n = len(text)
    while i < n and text[i] in _WS:
        i += 1
    return text[line_start:i]


def _first_line_indent(text: str) -> str:
    i = 0
    while i < len(text) and text[i] in _WS:
        i += 1
    return text[:i]


def _dedupe(spans: list[tuple[int, int, bool]]) -> list[tuple[int, int, bool]]:
    out: list[tuple[int, int, bool]] = []
    for span in spans:
        key = (span[0], span[1])
        if not any((o[0], o[1]) == key for o in out):
            out.append(span)
    return out


# ---------------------------------------------------------------------------
# span anchoring
# ---------------------------------------------------------------------------

def _anchor(content: str, old: str, s: int, e: int) -> tuple[int, int, bool]:
    """Fix up one mapped span so old/new indentation stays consistent.

    Returns ``(start, end, adjust_indent)``. When the match sits at the start
    of a line's code (everything before it is indentation):

    - indentation **matches** the file -> span pulls back to the line start so
      ``new`` replaces it (the model saw the real indentation, so honour it);
    - indentation **differs** -> span stops after the file's indentation and
      ``adjust_indent`` tells the caller to drop ``new``'s leading whitespace,
      so the file keeps the indentation it already had.

    End: mirror exactly the whitespace ``old``'s tail lost to normalisation —
    when ``old`` ended with a newline, swallow the line's trailing whitespace
    and that newline; when it ended with spaces, swallow the run; otherwise
    leave the rest of the line (and its newline) alone.
    """
    line_start = content.rfind("\n", 0, s) + 1
    adjust = False
    if content[line_start:s].strip() == "":
        file_ind = _line_indent(content, line_start)
        if _first_line_indent(old) != file_ind:
            s = line_start + len(file_ind)
            adjust = True
        else:
            s = line_start

    n = len(content)
    tail = old[-1:]
    if tail == "\n" or tail in _WS:
        t = e
        while t < n and content[t] in _WS:
            t += 1
        e = t
        if tail == "\n" and e < n and content[e] == "\n":
            e += 1
    return s, e, adjust


# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------

def _anchor_ok(nc: str, i: int, level: int) -> bool:
    """Is position *i* an acceptable place for a match to begin?

    ``line-trim`` may only match at a line start — otherwise every indentation
    difference would masquerade as a trailing-whitespace fix. Deeper levels
    relax that to "after whitespace", which is what makes an indentation-diff
    or a mid-line fragment eligible without letting ``foo`` match inside
    ``barfoo``.
    """
    if i == 0:
        return True
    prev = nc[i - 1]
    if prev == "\n":
        return True
    return level >= 2 and prev in _WS


def _end_anchor_ok(nc: str, me: int, old: str) -> bool:
    """When *old* claims to end a line, the match must actually end one.

    Without this, ``old="foo\\n"`` would happily match the ``foo`` inside
    ``foo bar`` and eat the rest of the line's prefix. *old* without a trailing
    newline is a legitimate mid-line fragment, so it stays unconstrained.
    """
    if not old.endswith("\n"):
        return True
    return me >= len(nc) or nc[me] == "\n"


def _normalized_matches(content: str, old: str, level: int) -> list[tuple[int, int, bool]]:
    nc, cstarts, cends = _normalize(content, level)
    no, _, _ = _normalize(old, level)
    if not no or not nc:
        return []

    found: list[tuple[int, int, bool]] = []
    start = 0
    while True:
        i = nc.find(no, start)
        if i == -1:
            break
        start = i + 1
        me = i + len(no)
        if i >= len(cstarts) or me - 1 >= len(cends):
            continue
        if not _anchor_ok(nc, i, level) or not _end_anchor_ok(nc, me, old):
            continue
        s, e, adjust = _anchor(content, old, cstarts[i], cends[me - 1])
        if e > s:
            found.append((s, e, adjust))
    return _dedupe(sorted(found))


def _near_matches(content: str, old: str) -> list[tuple[int, int, bool]]:
    """Sliding line-block comparison for 1-2 character drift."""
    if len(old.strip()) < NEAR_MIN_NEEDLE_LEN:
        return []
    nc, cstarts, cends = _normalize(content, 1)
    no, _, _ = _normalize(old, 1)
    if not no.strip() or not nc.strip():
        return []

    c_lines = _line_spans(nc)
    o_lines = no.split("\n")
    k = len(o_lines)
    if k == 0 or len(c_lines) < k:
        return []

    found: list[tuple[int, int, bool]] = []
    for i in range(len(c_lines) - k + 1):
        block = "\n".join(nc[a:b] for a, b in c_lines[i : i + k])
        if not block.strip():
            continue
        if SequenceMatcher(None, block, no).ratio() < NEAR_CUTOFF:
            continue
        ws = c_lines[i][0]
        we = c_lines[i + k - 1][1]
        if we <= ws:
            we = ws + 1
        if ws >= len(cstarts) or we - 1 >= len(cends):
            continue
        s, e, adjust = _anchor(content, old, cstarts[ws], cends[we - 1])
        if e > s:
            found.append((s, e, adjust))
    return _dedupe(sorted(found))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def resolve_match(content: str, old: str) -> Match | None:
    """Find *old* in *content* via the fuzzy chain, or return ``None``.

    Returns the FIRST strategy that produces any match — even an ambiguous one,
    so the caller can refuse to guess rather than falling through to a looser
    strategy that may land on a different spot entirely.
    """
    if not isinstance(old, str) or not old or not isinstance(content, str):
        return None
    if not content:
        return None

    if len(old) >= MIN_NEEDLE_LEN:
        for level, name in ((1, "line-trim"), (2, "indent"), (3, "collapse-ws")):
            found = _normalized_matches(content, old, level)
            if found:
                return Match(
                    name,
                    tuple((s, e) for s, e, _ in found),
                    any(adj for _, _, adj in found),
                )

    if len(old) >= NEAR_MIN_NEEDLE_LEN:
        found = _near_matches(content, old)
        if found:
            return Match(
                "near",
                tuple((s, e) for s, e, _ in found),
                any(adj for _, _, adj in found),
            )

    return None
