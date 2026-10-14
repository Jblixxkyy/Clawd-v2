"""check_for_file_mentions: scan user messages for filenames.

Mirrors aider's check_for_file_mentions(): find tokens in the message that
look like filenames and resolve them against the repo's known files, so
mentioned files can be seeded into the PageRank personalization vector.
"""

from __future__ import annotations

import keyword
import re

_FILE_TOKEN_RE = re.compile(r"[A-Za-z0-9_~\-][A-Za-z0-9_\-./~]*\.[A-Za-z0-9]{1,6}\b")
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")

_STOP_IDENTS = set(keyword.kwlist) | {
    "self", "cls", "True", "False", "None", "the", "and", "with", "from",
    "that", "this", "have", "what", "when", "where", "which", "your",
}


def check_for_file_mentions(text: str, known_files: list[str]) -> set[str]:
    """Return repo-relative paths mentioned in *text*.

    Matches on basename or path suffix, e.g. "utils.py" matches
    "pkg/utils.py". Case-sensitive.
    """
    if not text or not known_files:
        return set()
    by_base: dict[str, list[str]] = {}
    for rel in known_files:
        base = rel.rsplit("/", 1)[-1]
        by_base.setdefault(base, []).append(rel)

    found: set[str] = set()
    for tok in _FILE_TOKEN_RE.findall(text):
        tok = tok.strip("./~")
        if tok in by_base:
            found.update(by_base[tok])
            continue
        # suffix match: "pkg/utils.py" or "./pkg/utils.py"
        for rel in known_files:
            if rel.endswith("/" + tok) or rel == tok:
                found.add(rel)
    return found


def extract_mentioned_idents(text: str, known_idents: set[str] | None = None) -> set[str]:
    """Identifiers in *text* that look intentional (not prose).

    If known_idents is given, only idents defined in the repo are returned.
    """
    if not text:
        return set()
    out: set[str] = set()
    for tok in _IDENT_RE.findall(text):
        if tok in _STOP_IDENTS:
            continue
        if tok.lower() == tok and "_" not in tok and len(tok) < 5:
            continue  # plain lowercase prose word
        if known_idents is not None and tok not in known_idents:
            continue
        out.add(tok)
    return out
