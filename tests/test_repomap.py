"""Tests for the repomap package (brief 021).

Fast and hermetic: every test builds its own repo in a tmp dir, uses a
tmp cache dir, and touches no network.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.repomap import (
    RepoMap,
    check_for_file_mentions,
    extract_mentioned_idents,
)
from src.repomap.render import render_map
from src.token_estimation import count_tokens


def write_repo(files: dict[str, str]) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="repomap-test-"))
    for rel, content in files.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return tmp


def make_rm(root: Path, **kwargs) -> RepoMap:
    cache_dir = root / ".repocache"
    return RepoMap(root, cache_dir=cache_dir, **kwargs)


SAMPLE_REPO = {
    "alpha.py": (
        "def alpha_main():\n"
        "    return beta_helper(1)\n"
        "\n"
        "def alpha_util(x):\n"
        "    return x + 1\n"
    ),
    "beta.py": (
        "def beta_helper(v):\n"
        "    return v * 2\n"
        "\n"
        "class BetaWidget:\n"
        "    def render(self):\n"
        "        return beta_helper(0)\n"
    ),
    "gamma.py": (
        "def gamma_lonely_function():\n"
        "    return 'nobody calls me'\n"
    ),
}


class TestPersonalization(unittest.TestCase):
    def test_chat_files_rank_first(self):
        root = write_repo(SAMPLE_REPO)
        rm = make_rm(root, refresh="off")
        ranked = rm.ranked_tags(chat_fnames=["alpha.py"])
        top = ranked[0][0]
        # alpha.py is in chat: its own symbols or what it touches should lead
        self.assertIn(top.rel, ("alpha.py", "beta.py"))
        # the unreferenced lonely function should rank below beta_helper
        order = [(t.rel, t.name) for t, _ in ranked]
        self.assertLess(
            order.index(("beta.py", "beta_helper")),
            order.index(("gamma.py", "gamma_lonely_function")),
        )

    def test_mentioned_ident_boost(self):
        root = write_repo(SAMPLE_REPO)
        rm = make_rm(root, refresh="off")
        plain = [(t.rel, t.name) for t, _ in rm.ranked_tags()]
        boosted = [(t.rel, t.name) for t, _ in
                   rm.ranked_tags(mentioned_idents=["gamma_lonely_function"])]
        self.assertLess(
            boosted.index(("gamma.py", "gamma_lonely_function")),
            plain.index(("gamma.py", "gamma_lonely_function")),
        )

    def test_long_snake_case_weight(self):
        from src.repomap.pagerank import is_long_snake

        self.assertTrue(is_long_snake("compute_total_price"))
        self.assertTrue(is_long_snake("a_very_long_snake_case_name"))
        self.assertFalse(is_long_snake("tiny_name"))
        self.assertFalse(is_long_snake("CamelCaseName"))
        self.assertFalse(is_long_snake("nosnake"))


class TestBudget(unittest.TestCase):
    def test_budget_bounded_within_tolerance(self):
        files = {}
        for i in range(8):
            funcs = "\n\n".join(
                f"def module{i}_function{j}(arg1, arg2):\n    return arg1 + arg2 + {j}"
                for j in range(10)
            )
            files[f"mod{i}.py"] = funcs + "\n"
        root = write_repo(files)
        budget = 300
        rm = make_rm(root, refresh="off", max_map_tokens=budget)
        text = rm.get_map()
        tokens = count_tokens(text)
        full = render_map(rm.ranked_tags(), max_map_tokens=10**9)
        self.assertGreater(count_tokens(full), budget,
                           "fixture too small to test budget pressure")
        self.assertLessEqual(tokens, budget)
        self.assertGreaterEqual(tokens, int(budget * 0.85))

    def test_small_repo_under_budget_renders_fully(self):
        root = write_repo(SAMPLE_REPO)
        rm = make_rm(root, refresh="off", max_map_tokens=100000)
        text = rm.get_map()
        self.assertIn("alpha_main", text)
        self.assertIn("gamma_lonely_function", text)

    def test_empty_repo(self):
        root = write_repo({})
        rm = make_rm(root, refresh="off")
        self.assertEqual(rm.get_map().strip(), "")


class TestCache(unittest.TestCase):
    def test_disk_cache_hit_no_reparse(self):
        root = write_repo(SAMPLE_REPO)
        rm = make_rm(root, refresh="auto")
        first = rm.get_tags()
        cache_files = list((root / ".repocache").glob("*.json"))
        self.assertEqual(len(cache_files), 1)
        mtime_before = cache_files[0].stat().st_mtime
        time.sleep(0.02)
        second = rm.get_tags()
        mtime_after = cache_files[0].stat().st_mtime
        self.assertEqual(mtime_before, mtime_after)
        self.assertEqual(
            [(t.rel, t.name, t.kind) for t in first],
            [(t.rel, t.name, t.kind) for t in second],
        )

    def test_mtime_change_refreshes(self):
        root = write_repo(SAMPLE_REPO)
        rm = make_rm(root, refresh="auto")
        self.assertNotIn(
            ("alpha.py", "brand_new_function"),
            [(t.rel, t.name) for t in rm.get_tags()],
        )
        p = root / "alpha.py"
        p.write_text(p.read_text() + "\ndef brand_new_function():\n    pass\n")
        # force a distinct mtime (some filesystems have coarse granularity)
        new_mtime = p.stat().st_mtime + 5
        os.utime(p, (new_mtime, new_mtime))
        names = [(t.rel, t.name) for t in rm.get_tags()]
        self.assertIn(("alpha.py", "brand_new_function"), names)

    def test_on_demand_mode_stale_until_refresh(self):
        root = write_repo(SAMPLE_REPO)
        rm = make_rm(root, refresh="on_demand")
        before = {(t.rel, t.name) for t in rm.get_tags()}
        p = root / "alpha.py"
        p.write_text(p.read_text() + "\ndef on_demand_new():\n    pass\n")
        os.utime(p, (p.stat().st_mtime + 5,) * 2)
        stale = {(t.rel, t.name) for t in rm.get_tags()}
        self.assertEqual(before, stale)
        rm.refresh()
        fresh = {(t.rel, t.name) for t in rm.get_tags()}
        self.assertIn(("alpha.py", "on_demand_new"), fresh)

    def test_off_mode_writes_no_cache(self):
        root = write_repo(SAMPLE_REPO)
        rm = make_rm(root, refresh="off")
        rm.get_tags()
        rm.get_map()
        self.assertEqual(list((root / ".repocache").glob("*.json")), [])


class TestMentions(unittest.TestCase):
    def test_check_for_file_mentions(self):
        known = ["pkg/utils.py", "pkg/helpers.py", "main.py"]
        found = check_for_file_mentions(
            "The bug is in utils.py, and helpers.py too. See ./pkg/utils.py!", known
        )
        self.assertEqual(found, {"pkg/utils.py", "pkg/helpers.py"})

    def test_mentioned_filename_boosts_rank(self):
        root = write_repo(SAMPLE_REPO)
        rm = make_rm(root, refresh="off")
        plain = [(t.rel, t.name) for t, _ in rm.ranked_tags()]
        msg = "can you look at gamma.py? gamma_lonely_function seems off"
        mentioned_files = check_for_file_mentions(msg, rm.source_files())
        self.assertEqual(mentioned_files, {"gamma.py"})
        known_idents = {t.name for t in rm.get_tags() if t.kind != "ref"}
        mentioned_idents = extract_mentioned_idents(msg, known_idents)
        self.assertIn("gamma_lonely_function", mentioned_idents)
        boosted = [(t.rel, t.name) for t, _ in rm.ranked_tags(
            mentioned_fnames=mentioned_files, mentioned_idents=mentioned_idents)]
        self.assertLess(
            boosted.index(("gamma.py", "gamma_lonely_function")),
            plain.index(("gamma.py", "gamma_lonely_function")),
        )


class TestRetrievalBenchmark(unittest.TestCase):
    """Map-guided ranking beats unpersonalized ranking at finding the
    function relevant to the files already in context."""

    def test_map_wins_find_the_function(self):
        files = {
            "handlers.py": (
                "from services import process_payment\n"
                "\n"
                "def handle_checkout(cart):\n"
                "    return process_payment(cart)\n"
            ),
            "services.py": (
                "def process_payment(cart):\n"
                "    total = sum(cart)\n"
                "    return charge_gateway(total)\n"
                "\n"
                "def charge_gateway(amount):\n"
                "    return amount\n"
            ),
        }
        for i in range(6):
            files[f"unrelated{i}.py"] = (
                f"def unrelated{i}_task():\n    return {i}\n"
            )
        root = write_repo(files)
        rm = make_rm(root, refresh="off")

        def rank_of(name, **kw):
            order = [t.name for t, _ in rm.ranked_tags(**kw)]
            return order.index(name)

        # "user is working in handlers.py" -> process_payment should surface
        guided = rank_of("process_payment", chat_fnames=["handlers.py"])
        unguided = rank_of("process_payment")
        self.assertLessEqual(guided, unguided)
        self.assertLess(guided, 4,
                        "relevant function should be near the top with the map")


class TestIgnores(unittest.TestCase):
    def test_gitignore_and_binary_respected(self):
        root = write_repo(dict(SAMPLE_REPO))
        (root / ".gitignore").write_text("ignored.py\n")
        (root / "ignored.py").write_text("def should_not_appear():\n    pass\n")
        (root / "blob.py").write_bytes(b"\x00\x01\x02def fake():\n    pass\n")
        rm = make_rm(root, refresh="off")
        rels = set(rm.source_files())
        self.assertNotIn("ignored.py", rels)
        self.assertNotIn("blob.py", rels)
        self.assertIn("alpha.py", rels)


if __name__ == "__main__":
    unittest.main()
