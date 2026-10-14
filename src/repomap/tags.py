"""Tag extraction: regex/heuristic tagger (Python-focused).

tree-sitter is not installed in this environment and pip is blocked by
PEP 668 (externally-managed environment), so tags come from a careful
regex pass instead of a tree-sitter query. If tree_sitter ever becomes
importable, TREE_SITTER_AVAILABLE flips True, but the regex path remains
the default (hermetic, no native deps).

A Tag is either a definition (class / def / module-level var) or a
reference: (file, identifier) pairs with a count of how often the file
mentions the identifier.
"""

from __future__ import annotations

import fnmatch
import keyword
import re
from dataclasses import dataclass, field
from pathlib import Path

TREE_SITTER_AVAILABLE = False
try:  # optional; regex path is used regardless
    import tree_sitter  # noqa: F401

    TREE_SITTER_AVAILABLE = True
except Exception:
    pass

# -- file discovery ------------------------------------------------------

DEFAULT_IGNORE_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build", "egg-info",
    ".cache",
}
SOURCE_EXTENSIONS = {
    ".py", ".pyi",
    ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".c", ".h", ".cpp", ".hpp", ".cs",
    ".rb", ".php", ".swift", ".kt",
}
BINARY_CHECK_BYTES = 8192


def _load_gitignore(root: Path) -> list[str]:
    pats = []
    gi = root / ".gitignore"
    if not gi.is_file():
        return pats
    try:
        for line in gi.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pats.append(line)
    except OSError:
        pass
    return pats


def _ignored(rel: str, patterns: list[str]) -> bool:
    for pat in patterns:
        p = pat
        if p.startswith("/"):
            p = p[1:]
        if p.endswith("/"):
            if rel == p[:-1] or rel.startswith(p):
                return True
            continue
        if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(Path(rel).name, p):
            return True
        # match pattern against any suffix of the path (git-ish behavior)
        parts = rel.split("/")
        for i in range(len(parts)):
            if fnmatch.fnmatch("/".join(parts[i:]), p):
                return True
    return False


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(BINARY_CHECK_BYTES)
        return b"\x00" in chunk
    except OSError:
        return True


def collect_source_files(root: Path, *, max_files: int = 5000) -> list[str]:
    """Repo-relative source files, respecting .gitignore and skipping binaries."""
    root = Path(root)
    patterns = _load_gitignore(root)
    out: list[str] = []
    for path in sorted(root.rglob("*")):
        if len(out) >= max_files:
            break
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        parts = rel.split("/")
        if any(p in DEFAULT_IGNORE_DIRS or p.startswith(".") and p in (".git",) for p in parts):
            continue
        if any(p in DEFAULT_IGNORE_DIRS for p in parts):
            continue
        if _ignored(rel, patterns):
            continue
        if path.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        if _is_binary(path):
            continue
        out.append(rel)
    return out


# -- tags ----------------------------------------------------------------

@dataclass
class Tag:
    """A definition or a reference found in a source file."""

    rel: str            # repo-relative file path
    name: str           # bare identifier (no qualification)
    qualname: str       # qualified name (Class.method)
    kind: str           # "def" | "class" | "var" | "ref"
    line: int           # 1-based line number
    signature: str = ""  # source signature line (defs only)
    count: int = 1      # reference count (refs only)
    extra: dict = field(default_factory=dict)


_DEF_RE = re.compile(r"^(\s*)(?:async\s+)?def\s+([A-Za-z_]\w*)\s*(\(.*)?\s*:?\s*(?:#.*)?$")
_CLASS_RE = re.compile(r"^(\s*)class\s+([A-Za-z_]\w*)\s*(\(.*\))?\s*:")
_VAR_RE = re.compile(r"^([A-Za-z_]\w*)\s*(:\s*[^=]+)?=\s*(?![=])")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TRIPLE_RE = re.compile(r'("""|\'\'\')(.*?)(\1)', re.DOTALL)
_STRING_RE = re.compile(r"(?P<q>\"\"\"|'''|\"|')(?:\\.|(?!(?P=q)).)*?(?P=q)", re.DOTALL)

_PY_KEYWORDS = set(keyword.kwlist) | {
    "self", "cls", "True", "False", "None", "NotImplemented", "Ellipsis",
}


def _strip_strings_and_comments(line: str) -> str:
    line = _STRING_RE.sub("", line)
    # naive comment strip: cut at # not inside anything (strings already gone)
    idx = line.find("#")
    if idx != -1:
        line = line[:idx]
    return line


def _extract_python_defs(path: Path, rel: str, lines: list[str]) -> list[Tag]:
    tags: list[Tag] = []
    stack: list[tuple[int, str]] = []  # (indent, qualname)
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        indent = len(line) - len(line.lstrip(" "))
        # pop stack to current scope
        while stack and indent <= stack[-1][0]:
            stack.pop()
        scope = ".".join(s for _, s in stack)

        m = _CLASS_RE.match(line)
        if m:
            name = m.group(2)
            qual = f"{scope}.{name}" if scope else name
            sig = line.strip()
            # swallow base-class continuation lines for the signature
            j = i
            while not sig.rstrip().endswith(":") and j + 1 < n and j - i < 4:
                j += 1
                sig += " " + lines[j].strip()
            tags.append(Tag(rel, name, qual, "class", i + 1, sig[:160]))
            stack.append((indent, name))
            i = j + 1
            continue

        m = _DEF_RE.match(line)
        if m:
            name = m.group(2)
            qual = f"{scope}.{name}" if scope else name
            sig = line.strip()
            # multi-line signature: join until parens balance
            depth = sig.count("(") - sig.count(")")
            j = i
            while depth > 0 and j + 1 < n and j - i < 8:
                j += 1
                sig += " " + lines[j].strip()
                depth = sig.count("(") - sig.count(")")
            sig = re.sub(r"\s+", " ", sig)
            if len(sig) > 160:
                sig = sig[:157] + "..."
            tags.append(Tag(rel, name, qual, "def", i + 1, sig))
            stack.append((indent, name))
            i = j + 1
            continue

        if indent == 0:
            m = _VAR_RE.match(line)
            if m and not _DEF_RE.match(line) and not _CLASS_RE.match(line):
                name = m.group(1)
                if name not in _PY_KEYWORDS and not name.startswith("_"):
                    tags.append(Tag(rel, name, name, "var", i + 1, line.strip()[:120]))
        i += 1
    return tags


def _extract_python_refs(path: Path, rel: str, text: str, def_names: set[str]) -> list[Tag]:
    """Reference tags: (file, ident) -> count, for idents defined somewhere."""
    if not def_names:
        return []
    # drop triple-quoted strings wholesale to cut docstring noise
    scrubbed = _TRIPLE_RE.sub("", text)
    counts: dict[str, int] = {}
    for line in scrubbed.splitlines():
        code = _strip_strings_and_comments(line)
        if "import " in code and re.match(r"\s*(import|from)\b", code):
            continue
        for tok in _IDENT_RE.findall(code):
            if tok in def_names and tok not in _PY_KEYWORDS:
                counts[tok] = counts.get(tok, 0) + 1
    return [
        Tag(rel, name, name, "ref", 0, "", count=c)
        for name, c in counts.items()
    ]


def _extract_generic_defs(rel: str, lines: list[str]) -> list[Tag]:
    """Fallback for non-Python files: function-ish definitions only."""
    tags: list[Tag] = []
    pat = re.compile(
        r"^\s*(?:function\s+([A-Za-z_]\w*)|([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{|"
        r"func\s+([A-Za-z_]\w*)\s*\(|fn\s+([A-Za-z_]\w*)\s*\()"
    )
    for i, line in enumerate(lines, 1):
        m = pat.match(line)
        if m:
            name = next(g for g in m.groups() if g)
            tags.append(Tag(rel, name, name, "def", i, line.strip()[:160]))
    return tags


def extract_refs(path: Path, rel: str, def_names: set[str]) -> list[Tag]:
    """Reference tags for one file against repo-wide definition names."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if path.suffix.lower() in (".py", ".pyi"):
        return _extract_python_refs(path, rel, text, def_names)
    return []


def extract_tags(path: Path, rel: str, repo_def_names: set[str] | None = None) -> list[Tag]:
    """Extract def + ref tags for one file.

    repo_def_names: definition names across the whole repo (for refs).
    Defaults to the file's own defs when omitted.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    if path.suffix.lower() in (".py", ".pyi"):
        defs = _extract_python_defs(path, rel, lines)
    else:
        defs = _extract_generic_defs(rel, lines)
    if repo_def_names is None:
        repo_def_names = {t.name for t in defs if t.kind in ("def", "class", "var")}
    refs = extract_refs(path, rel, repo_def_names)
    return defs + refs
