"""Token-budgeted rendering of ranked tags.

Binary-search the ranked tag list for the largest prefix whose rendered
text fits in max_map_tokens. Signatures only — bodies are elided.
"""

from __future__ import annotations


def _count_tokens(text: str) -> int:
    try:
        from ..token_estimation import count_tokens

        return count_tokens(text)
    except Exception:
        return max(1, len(text) // 4)


def _format_tag(tag) -> str:
    sig = (tag.signature or "").strip()
    if tag.kind == "class":
        line = f"class {tag.name}"
        if "(" in sig:
            line += sig[sig.index("("):].rstrip(":")
    elif tag.kind == "def":
        line = f"def {tag.qualname}{sig[sig.index('('):] if '(' in sig else '()'}"
        line = line.rstrip(":")
    elif tag.kind == "var":
        line = f"{tag.name} = ..."
    else:
        line = tag.name
    return line


def render_lines(ranked: list[tuple], count: int) -> list[str]:
    """Render the top-*count* ranked tags, grouped by file.

    Files are ordered by their best tag's rank; tags within a file keep
    rank order. File headers appear only for files with rendered tags.
    """
    selected = [tag for tag, _ in ranked[:count]]
    # file order = first appearance in ranked order
    file_order: list[str] = []
    by_file: dict[str, list] = {}
    for tag in selected:
        if tag.rel not in by_file:
            by_file[tag.rel] = []
            file_order.append(tag.rel)
        by_file[tag.rel].append(tag)
    lines: list[str] = []
    for rel in file_order:
        lines.append(f"{rel}:")
        for tag in by_file[rel]:
            lines.append(f"  {_format_tag(tag)}")
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def render_map(ranked: list[tuple], *, max_map_tokens: int = 1024) -> str:
    """Render ranked tags into at most max_map_tokens of map text.

    Binary search for the largest tag count that fits the budget.
    """
    if not ranked or max_map_tokens <= 0:
        return ""

    def tokens_for(count: int) -> int:
        return _count_tokens("\n".join(render_lines(ranked, count)))

    total = tokens_for(len(ranked))
    if total <= max_map_tokens:
        return "\n".join(render_lines(ranked, len(ranked)))

    lo, hi = 0, len(ranked)
    # invariant: tokens_for(lo) <= budget < tokens_for(hi)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if tokens_for(mid) <= max_map_tokens:
            lo = mid
        else:
            hi = mid
    return "\n".join(render_lines(ranked, lo))
