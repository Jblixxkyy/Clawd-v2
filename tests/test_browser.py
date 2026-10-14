"""Focused tests for the dual-control managed browser panel."""

from __future__ import annotations

import concurrent.futures
import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.agent import Session
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.tools.browser_manager import (
    BrowserManager,
    get_browser_manager,
    reset_browser_manager,
)
from src.web_app import _AppState, _HTML, _make_handler


class FakeBackend:
    """In-memory stand-in for the Playwright backend (no browser needed)."""

    def __init__(self) -> None:
        self.url = "about:blank"
        self.title = "Fake"
        self.calls: list[tuple] = []
        self.text = "hello fake page"
        self.closed = False

    def open(self, url: str):
        self.calls.append(("open", url))
        self.url = url
        self.closed = False
        return {"ok": True, "action": "open", "url": url, "title": self.title}

    def navigate(self, url: str):
        self.calls.append(("navigate", url))
        self.url = url
        return {"ok": True, "action": "navigate", "url": url, "title": self.title}

    def snapshot(self):
        self.calls.append(("snapshot",))
        if self.closed:
            return {"ok": False, "action": "snapshot", "error": "no open page"}
        return {"ok": True, "action": "snapshot", "url": self.url,
                "title": self.title, "text": self.text,
                "refs": [{"ref": 0, "tag": "button", "name": "Go", "selector": "#go"}],
                "screenshot": "ZmFrZQ=="}

    def click(self, selector: str):
        self.calls.append(("click", selector))
        if self.closed:
            return {"ok": False, "action": "click", "error": "no open page"}
        return {"ok": True, "action": "click", "selector": selector, "url": self.url}

    def click_at(self, x: float, y: float):
        self.calls.append(("click_at", x, y))
        return {"ok": True, "action": "click", "x": x, "y": y, "url": self.url}

    def type(self, selector: str, text: str):
        self.calls.append(("type", selector, text))
        return {"ok": True, "action": "type", "selector": selector, "url": self.url}

    def screenshot(self):
        self.calls.append(("screenshot",))
        return {"ok": True, "action": "screenshot", "url": self.url, "screenshot": "ZmFrZQ=="}

    def back(self):
        self.calls.append(("back",))
        return {"ok": True, "action": "back", "url": self.url}

    def forward(self):
        self.calls.append(("forward",))
        return {"ok": True, "action": "forward", "url": self.url}

    def reload(self):
        self.calls.append(("reload",))
        return {"ok": True, "action": "reload", "url": self.url}

    def close(self):
        self.calls.append(("close",))
        self.closed = True
        return {"ok": True, "action": "close", "closed": True}

    def hover_at(self, x: float, y: float):
        return {"ok": True, "action": "hover",
                "element": {"selector": "#go", "tag": "button",
                            "box": {"x": x, "y": y, "w": 0.1, "h": 0.05}}}

    def pick_at(self, x: float, y: float):
        return {"ok": True, "action": "pick", "url": self.url,
                "selector": "#go", "tag": "button",
                "outerHTML": '<button id="go">Go</button>',
                "box": {"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.05},
                "screenshot": "ZmFrZQ=="}


def _ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace_root=workspace)


class BrowserToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name).resolve()
        self.backend = FakeBackend()
        reset_browser_manager(self.backend)

    def tearDown(self) -> None:
        reset_browser_manager(None)
        self.temp.cleanup()

    def test_tools_registered(self) -> None:
        registry = build_default_registry(include_user_tools=False)
        names = {t.spec().name for t in registry.list_tools()}
        for expected in ("browser_open", "browser_navigate", "browser_snapshot",
                         "browser_click", "browser_type", "browser_screenshot",
                         "browser_back", "browser_forward", "browser_reload",
                         "browser_close"):
            self.assertIn(expected, names)

    def test_open_snapshot_click_type(self) -> None:
        manager = get_browser_manager()
        opened = manager.open("https://example.com", actor="agent")
        self.assertTrue(opened["ok"])
        self.assertEqual(opened["url"], "https://example.com")
        snap = manager.snapshot(actor="agent")
        self.assertTrue(snap["ok"])
        self.assertEqual(snap["url"], "https://example.com")
        self.assertIn("refs", snap)
        self.assertIn("screenshot", snap)
        clicked = manager.click("#go", actor="agent")
        self.assertTrue(clicked["ok"])
        typed = manager.type("#q", "hello", actor="agent")
        self.assertTrue(typed["ok"])
        shot = manager.screenshot(actor="agent")
        self.assertTrue(shot["ok"])
        self.assertIn("screenshot", shot)

    def test_errors_do_not_raise(self) -> None:
        manager = get_browser_manager()
        manager.close(actor="agent")
        snap = manager.snapshot(actor="agent")
        self.assertIsInstance(snap, dict)
        self.assertIn("ok", snap)

    def test_user_action_ordering_and_generation(self) -> None:
        manager = get_browser_manager()
        gen0 = manager.generation
        manager.open("https://a.example", actor="agent")
        self.assertEqual(manager.generation, gen0)  # agent actions don't bump generation
        manager.navigate("https://b.example", actor="user")
        self.assertEqual(manager.generation, gen0 + 1)
        self.assertEqual(manager.last_actor, "user")
        snap = manager.snapshot(actor="agent")
        self.assertEqual(snap["url"], "https://b.example")
        self.assertEqual(snap["generation"], manager.generation)
        actors = [e["actor"] for e in manager.action_log]
        self.assertEqual(actors, ["agent", "user", "agent"])

    def test_take_control_blocks_agent(self) -> None:
        manager = get_browser_manager()
        manager.open("https://example.com", actor="agent")
        manager.set_control("user", actor="user")
        blocked = manager.click("#go", actor="agent")
        self.assertFalse(blocked["ok"])
        self.assertIn("user has control", blocked["error"])
        # user actions still work while holding control
        ok = manager.click_at(0.5, 0.5, actor="user")
        self.assertTrue(ok["ok"])
        snap = manager.snapshot(actor="agent")
        self.assertFalse(snap["ok"])
        manager.set_control("shared", actor="user")
        self.assertTrue(manager.click("#go", actor="agent")["ok"])

    def test_picker_payload(self) -> None:
        manager = get_browser_manager()
        manager.open("https://example.com", actor="user")
        picked = manager.pick_at(0.15, 0.22, actor="user")
        self.assertTrue(picked["ok"])
        self.assertEqual(picked["selector"], "#go")
        result = manager.submit_comment({
            "comment": "fix this button",
            "selector": picked["selector"],
            "outerHTML": picked["outerHTML"],
            "url": picked["url"],
            "screenshot": picked["screenshot"],
        })
        self.assertTrue(result["ok"])
        entry = result["comment"]
        self.assertEqual(entry["comment"], "fix this button")
        self.assertEqual(entry["selector"], "#go")
        self.assertIn("button", entry["outerHTML"])
        self.assertTrue(entry["screenshot"])

    def test_backend_confines_playwright_to_one_thread(self) -> None:
        # The sync API breaks ("connection closed") when shared across the
        # web server's per-request threads, so every public method must run
        # on the backend's dedicated worker thread. Needs no browser.
        from src.tool_system.tools.browser_manager import _PlaywrightBackend

        backend = _PlaywrightBackend()
        try:
            worker_ids: set[int] = set()

            def probe() -> bool:
                worker_ids.add(threading.get_ident())
                return True

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: backend._dispatch(probe), range(16)))
            self.assertTrue(all(results))
            self.assertEqual(len(worker_ids), 1)
            self.assertNotIn(threading.get_ident(), worker_ids)
        finally:
            backend._drop()


class BrowserPanelEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name).resolve()
        self.backend = FakeBackend()
        reset_browser_manager(self.backend)
        self.state = _AppState(
            workspace=self.workspace,
            provider_name="nvidia",
            provider=SimpleNamespace(model="test-model"),
            session=Session.create("nvidia", "test-model"),
            registry=object(),
            tool_context=ToolContext(workspace_root=self.workspace),
        )
        self.state.sessions[self.state.session.session_id] = self.state.session
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self.state))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        reset_browser_manager(None)
        self.temp.cleanup()

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self.url}{path}", timeout=2) as response:
            return json.load(response)

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            return json.load(response)

    def test_state_endpoint(self) -> None:
        payload = self._get("/api/browser/state")
        for key in ("url", "control", "generation", "seq", "last_actor", "log"):
            self.assertIn(key, payload)

    def test_user_navigate_and_click_flow(self) -> None:
        nav = self._post("/api/browser/navigate", {"url": "https://example.com"})
        self.assertTrue(nav["ok"])
        clicked = self._post("/api/browser/click", {"x": 0.5, "y": 0.3})
        self.assertTrue(clicked["ok"])
        state = self._get("/api/browser/state")
        self.assertEqual(state["url"], "https://example.com")
        self.assertEqual(state["last_actor"], "user")
        self.assertGreaterEqual(state["generation"], 2)

    def test_control_toggle(self) -> None:
        self._post("/api/browser/navigate", {"url": "https://example.com"})
        held = self._post("/api/browser/control", {"mode": "user"})
        self.assertEqual(held["control"], "user")
        manager = get_browser_manager()
        blocked = manager.click("#go", actor="agent")
        self.assertIn("user has control", blocked["error"])
        released = self._post("/api/browser/control", {"mode": "shared"})
        self.assertEqual(released["control"], "shared")

    def test_comment_reaches_session(self) -> None:
        self._post("/api/browser/navigate", {"url": "https://example.com"})
        result = self._post("/api/browser/comment", {
            "comment": "fix this button",
            "selector": "#go",
            "outerHTML": '<button id="go">Go</button>',
            "url": "https://example.com",
            "session_id": self.state.session.session_id,
        })
        self.assertTrue(result["ok"])
        texts = [m.content for m in self.state.session.conversation.messages
                 if isinstance(m.content, str)]
        self.assertTrue(any("fix this button" in t and "#go" in t for t in texts))

    def test_panel_markup_present(self) -> None:
        with urllib.request.urlopen(f"{self.url}/", timeout=2) as response:
            body = response.read().decode("utf-8")
        for marker in ("browserPanel", "browserUrl", "browserShot", "browserBack",
                       "browserControlToggle", "browserPicker", "browserComment",
                       "browserCollapse", "browserResize", "browserEmpty",
                       "browser-head", "browser-statusbar",
                       "Select an element to comment on - While the page has focus Ctrl+Shift+C",
                       "/api/browser/state", "/api/browser/comment"):
            self.assertIn(marker, body)
        # screenshot streaming, not iframe
        self.assertNotIn("<iframe", body)


if __name__ == "__main__":
    unittest.main()
