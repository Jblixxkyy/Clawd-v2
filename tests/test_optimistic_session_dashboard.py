"""Tests for harness brief 034: optimistic session dashboard.

The optimistic session-lifecycle state machine lives in src/web_app.py as a
self-contained, DOM-free JS block (createSessionDashboard), delimited by
/*__OPTIMISTIC_SESSION_DASHBOARD_BEGIN__*/ / END markers. These tests:

  1. Extract that block from the served HTML and run behavioral scenarios
     against it under node (skipped if node is unavailable).
  2. Assert the block is actually served and wired into the page:
     progress-line/error CSS, renderSessionList integration, and the
     conflicting-action gates (switch/create/leave).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BEGIN_MARKER = "/*__OPTIMISTIC_SESSION_DASHBOARD_BEGIN__*/"
END_MARKER = "/*__OPTIMISTIC_SESSION_DASHBOARD_END__*/"
NODE = shutil.which("node")

DRIVER = r"""
const assert = require("assert");
const results = {};
function S(id, title) { return {session_id: id, title: title}; }

/* 1. deleting a session keeps the list browsable during the async op */
{
  const d = createSessionDashboard();
  d.setSessions([S("a", "Alpha"), S("b", "Beta"), S("c", "Gamma")]);
  const begun = d.begin("delete", "b");
  assert.strictEqual(begun.ok, true);
  assert.deepStrictEqual(d.visible().map(s => s.session_id), ["a", "c"]);
  assert.strictEqual(d.isPending(), true);
  const prog = d.progress();
  assert.ok(prog && prog.text.indexOf("Deleting") === 0);
  assert.ok(prog.text.indexOf("keep browsing") >= 0);
  /* navigation + search keep working over the visible list */
  const searched = d.visible().filter(s => (s.title || "").toLowerCase().includes("gam"))
    .map(s => s.session_id);
  assert.deepStrictEqual(searched, ["c"]);
  results.browsable_during_pending = true;
}

/* 2. conflicting actions are blocked with an inline message */
{
  const d = createSessionDashboard();
  d.setSessions([S("a", "Alpha"), S("b", "Beta")]);
  d.begin("delete", "a");
  const sw = d.gateSwitch("a");
  assert.strictEqual(sw.ok, false); assert.ok(sw.reason);
  assert.strictEqual(d.gateSwitch("b").ok, true); /* other chats still switchable */
  const cr = d.gateCreate();
  assert.strictEqual(cr.ok, false); assert.ok(cr.reason);
  const lv = d.gateLeave();
  assert.strictEqual(lv.ok, false); assert.ok(lv.reason);
  results.conflicts_blocked = true;
}

/* 3. selection index is preserved after the removal completes */
{
  const d = createSessionDashboard();
  d.setSessions([S("a", "Alpha"), S("b", "Beta"), S("c", "Gamma")]);
  d.begin("delete", "b");
  const done = d.complete();
  assert.strictEqual(done.selectId, "c"); /* item below becomes selected */
  assert.deepStrictEqual(d.visible().map(s => s.session_id), ["a", "c"]);

  const d2 = createSessionDashboard();
  d2.setSessions([S("a", "Alpha"), S("b", "Beta")]);
  d2.begin("delete", "b");
  assert.strictEqual(d2.complete().selectId, "a"); /* last item: select above */

  const d3 = createSessionDashboard();
  d3.setSessions([S("a", "Alpha")]);
  d3.begin("delete", "a");
  assert.strictEqual(d3.complete().selectId, null); /* only item: nothing left */
  results.selection_stable = true;
}

/* 4. failed deletion restores the row with an error message */
{
  const d = createSessionDashboard();
  d.setSessions([S("a", "Alpha"), S("b", "Beta"), S("c", "Gamma")]);
  d.begin("delete", "b");
  d.fail("boom");
  assert.strictEqual(d.isPending(), false);
  const vis = d.visible();
  assert.deepStrictEqual(vis.map(s => s.session_id), ["a", "b", "c"]);
  const row = vis.find(s => s.session_id === "b");
  assert.ok(row._optimisticError && row._optimisticError.indexOf("boom") >= 0);
  assert.strictEqual(row._optimisticRetry, true); /* retry is actionable */
  results.failure_restores_row = true;
}

/* 5. rename applies the new title optimistically, row stays in place */
{
  const d = createSessionDashboard();
  d.setSessions([S("a", "Alpha"), S("b", "Beta")]);
  assert.strictEqual(d.begin("rename", "a", {title: "Alpha 2"}).ok, true);
  const vis = d.visible();
  assert.strictEqual(vis.length, 2);
  assert.strictEqual(vis.find(s => s.session_id === "a").title, "Alpha 2");
  assert.ok(d.progress().text.indexOf("Renaming") === 0);
  /* rename is non-destructive: switching/leaving still allowed, creating not */
  assert.strictEqual(d.gateSwitch("a").ok, true);
  assert.strictEqual(d.gateLeave().ok, true);
  assert.strictEqual(d.gateCreate().ok, false);
  d.complete();
  assert.strictEqual(d.visible().find(s => s.session_id === "a").title, "Alpha 2");
  results.rename_optimistic = true;
}

/* 6. only one op may be pending at a time */
{
  const d = createSessionDashboard();
  d.setSessions([S("a", "Alpha"), S("b", "Beta")]);
  d.begin("delete", "a");
  const r = d.begin("delete", "b");
  assert.strictEqual(r.ok, false); assert.ok(r.reason);
  results.single_pending_op = true;
}

/* 7. failed rename restores the old title with an error state */
{
  const d = createSessionDashboard();
  d.setSessions([S("a", "Alpha")]);
  d.begin("rename", "a", {title: "Alpha 2"});
  d.fail("nope");
  const row = d.visible()[0];
  assert.strictEqual(row.title, "Alpha");
  assert.ok(row._optimisticError);
  assert.strictEqual(row._optimisticRetry, false); /* rename retry = edit again */
  results.rename_failure_restores = true;
}

console.log("DASHBOARD_RESULTS:" + JSON.stringify(results));
"""


def _extract_dashboard_js() -> str:
    src = (REPO / "src" / "web_app.py").read_text()
    m = re.search(re.escape(BEGIN_MARKER) + r"(.*?)" + re.escape(END_MARKER), src, re.S)
    assert m, "optimistic dashboard block markers missing from src/web_app.py"
    return m.group(1)


@unittest.skipIf(NODE is None, "node not available")
class OptimisticDashboardNodeTests(unittest.TestCase):
    def test_dashboard_scenarios(self) -> None:
        block = _extract_dashboard_js()
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(block + "\n" + DRIVER)
            path = f.name
        proc = subprocess.run([NODE, path], capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, f"node driver failed:\n{proc.stderr}")
        line = next(
            (ln for ln in proc.stdout.splitlines() if ln.startswith("DASHBOARD_RESULTS:")),
            "",
        )
        self.assertTrue(line, "node driver produced no results")
        results = json.loads(line.split("DASHBOARD_RESULTS:", 1)[1])
        for name, passed in sorted(results.items()):
            with self.subTest(scenario=name):
                self.assertTrue(passed, f"scenario {name} failed")


class OptimisticDashboardWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        from src.web_app import _HTML

        self.html: str = _HTML

    def test_block_is_served(self) -> None:
        self.assertIn(BEGIN_MARKER, self.html)
        self.assertIn(END_MARKER, self.html)
        self.assertIn("function createSessionDashboard()", self.html)

    def test_progress_and_error_css(self) -> None:
        self.assertIn(".session-progress{", self.html)
        self.assertIn(".session-error-row{", self.html)
        self.assertIn(".session-error-line{", self.html)

    def test_render_uses_dashboard(self) -> None:
        self.assertIn("sessionDashboard.setSessions(lastSessions)", self.html)
        self.assertIn("sessionDashboard.visible()", self.html)
        self.assertIn("sessionDashboard.progress()", self.html)
        self.assertIn("renderSessionErrorLine(list,session)", self.html)

    def test_delete_is_optimistic(self) -> None:
        self.assertIn('sessionDashboard.begin("delete",sessionId)', self.html)
        self.assertIn("sessionDashboard.complete()", self.html)
        self.assertIn("sessionDashboard.fail(e.message)", self.html)

    def test_rename_is_optimistic(self) -> None:
        self.assertIn('sessionDashboard.begin("rename",session.session_id,{title:value})', self.html)

    def test_conflicting_action_gates(self) -> None:
        self.assertIn("sessionDashboard.gateSwitch(sessionId)", self.html)
        self.assertIn("sessionDashboard.gateCreate()", self.html)
        self.assertIn("guardedCloseSidebar", self.html)
        self.assertIn("sessionDashboard.gateLeave()", self.html)

    def test_progress_copy(self) -> None:
        block = _extract_dashboard_js()
        self.assertIn("you can keep browsing", block)


if __name__ == "__main__":
    unittest.main()
