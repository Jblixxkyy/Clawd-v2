"""Definition/reference graph + personalized PageRank (stdlib only).

Graph model (mirrors aider's repomap.py):
  - nodes: ("file", rel) for files and ("tag", rel, name) for definitions.
  - edges:
      file -> its def tags (weight 1)
      def tag -> its file (weight 1)
      ref_file -> def tag, for every identifier defined somewhere that the
        file references, weighted by:
          base = reference count
          x IN_CHAT_WEIGHT (50) if the referencing file is in the chat
          x MENTIONED_WEIGHT (10) if the identifier was mentioned by the user
          x LONG_SNAKE_WEIGHT (10) if the identifier is long snake_case
  - personalization seeds: chat files, mentioned files, mentioned
    identifiers' def nodes, and identifiers defined in chat files.

PageRank: power iteration with damping, weighted edges, and a
personalization vector; dangling nodes redistribute via personalization.
"""

from __future__ import annotations

from .tags import Tag

IN_CHAT_WEIGHT = 50.0
MENTIONED_WEIGHT = 10.0
LONG_SNAKE_WEIGHT = 10.0
DAMPING = 0.85
MAX_ITER = 100
TOL = 1e-6

# aider treats long snake_case idents mentioned by the user as more salient
LONG_SNAKE_MIN_LEN = 10


def is_long_snake(name: str) -> bool:
    return (
        len(name) >= LONG_SNAKE_MIN_LEN
        and "_" in name
        and name == name.lower()
        and name.replace("_", "").isalnum()
    )


def build_graph(tags: list[Tag], *, chat_fnames=(), mentioned_idents=()):
    """Build weighted adjacency: {src: {dst: weight}} plus node list."""
    chat = set(chat_fnames)
    mentioned = set(mentioned_idents)

    adj: dict[tuple, dict[tuple, float]] = {}

    def add_edge(src, dst, w):
        adj.setdefault(src, {})
        adj[src][dst] = adj[src].get(dst, 0.0) + w
        adj.setdefault(dst, {})

    defs = [t for t in tags if t.kind in ("def", "class", "var")]
    refs = [t for t in tags if t.kind == "ref"]

    # def name -> def nodes (bare-name matching, like aider)
    defs_by_name: dict[str, list[tuple]] = {}
    for t in defs:
        node = ("tag", t.rel, t.name)
        defs_by_name.setdefault(t.name, []).append(node)
        fnode = ("file", t.rel)
        add_edge(fnode, node, 1.0)
        add_edge(node, fnode, 1.0)

    for r in refs:
        src = ("file", r.rel)
        mult = float(r.count)
        if r.rel in chat:
            mult *= IN_CHAT_WEIGHT
        if r.name in mentioned:
            mult *= MENTIONED_WEIGHT
        if is_long_snake(r.name):
            mult *= LONG_SNAKE_WEIGHT
        for dst in defs_by_name.get(r.name, []):
            if dst[1] == r.rel:
                continue  # skip self-file ref edges; def<->file covers it
            add_edge(src, dst, mult)

    return adj


def personalized_pagerank(adj, personalization, *, damping=DAMPING,
                          max_iter=MAX_ITER, tol=TOL):
    """Power-iteration PageRank over a weighted adjacency dict."""
    nodes = list(adj.keys())
    n = len(nodes)
    if n == 0:
        return {}
    idx = {node: i for i, node in enumerate(nodes)}

    out_w = [0.0] * n
    for i, node in enumerate(nodes):
        out_w[i] = sum(adj[node].values())

    pers = [0.0] * n
    for node, w in personalization.items():
        if node in idx:
            pers[idx[node]] = w
    total = sum(pers)
    if total <= 0:
        pers = [1.0 / n] * n
    else:
        pers = [p / total for p in pers]

    rank = pers[:]
    for _ in range(max_iter):
        new = [(1.0 - damping) * p for p in pers]
        dangling = 0.0
        for i, node in enumerate(nodes):
            r = rank[i]
            if r == 0.0:
                continue
            if out_w[i] <= 0:
                dangling += r
                continue
            share = damping * r / out_w[i]
            for dst, w in adj[node].items():
                new[idx[dst]] += share * w
        if dangling:
            add = damping * dangling
            for i in range(n):
                new[i] += add * pers[i]
        err = sum(abs(new[i] - rank[i]) for i in range(n))
        rank = new
        if err < tol:
            break
    return {node: rank[i] for i, node in enumerate(nodes)}


def rank_tags(tags: list[Tag], *, chat_fnames=(), mentioned_idents=(),
              mentioned_fnames=()) -> list[tuple[Tag, float]]:
    """Rank definition tags by personalized PageRank score (desc)."""
    chat = set(chat_fnames)
    mentioned_files = set(mentioned_fnames)
    mentioned = set(mentioned_idents)

    defs = [t for t in tags if t.kind in ("def", "class", "var")]
    if not defs:
        return []

    adj = build_graph(tags, chat_fnames=chat, mentioned_idents=mentioned)

    personalization: dict[tuple, float] = {}
    for t in defs:
        node = ("tag", t.rel, t.name)
        fnode = ("file", t.rel)
        if t.rel in chat or t.rel in mentioned_files:
            personalization[fnode] = personalization.get(fnode, 0.0) + 1.0
        if t.name in mentioned:
            personalization[node] = personalization.get(node, 0.0) + 1.0
        if t.rel in chat:
            # idents defined in chat files (aider's chat_idents seed)
            personalization[node] = personalization.get(node, 0.0) + 1.0
    for rel in mentioned_files:
        fnode = ("file", rel)
        personalization[fnode] = personalization.get(fnode, 0.0) + 1.0

    scores = personalized_pagerank(adj, personalization)
    ranked = []
    for t in defs:
        node = ("tag", t.rel, t.name)
        ranked.append((t, scores.get(node, 0.0)))
    # stable: score desc, then file, then line — deterministic
    ranked.sort(key=lambda tr: (-tr[1], tr[0].rel, tr[0].line))
    return ranked
