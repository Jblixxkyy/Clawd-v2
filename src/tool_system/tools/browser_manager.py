"""Shared managed-browser session for dual user/agent control.

Single Playwright Chromium page per web-app instance. Both user actions
(HTTP endpoints in web_app.py) and agent tool calls (browser.py) drive the
SAME page through the module-level singleton from get_browser_manager().

Isolated profile: ~/.clawd/browser-profile (never the user's personal
Chrome profile). No saved logins, no password autofill.
"""

from __future__ import annotations

import base64
import concurrent.futures
import functools
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _profile_dir() -> Path:
    return Path.home() / ".clawd" / "browser-profile"


def _on_browser_thread(fn):
    """Run the method on the backend's dedicated Playwright thread."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        return self._dispatch(lambda: fn(self, *args, **kwargs))
    return wrapper


class _UnavailableBackend:
    """Fallback when Playwright is not installed."""

    reason = "playwright is not installed (pip install playwright && playwright install chromium)"

    def _err(self, action: str) -> dict[str, Any]:
        return {"ok": False, "action": action, "error": self.reason}

    def open(self, url: str) -> dict[str, Any]:
        return self._err("open")

    def navigate(self, url: str) -> dict[str, Any]:
        return self._err("navigate")

    def snapshot(self) -> dict[str, Any]:
        return self._err("snapshot")

    def click(self, selector: str) -> dict[str, Any]:
        return self._err("click")

    def click_at(self, x: float, y: float) -> dict[str, Any]:
        return self._err("click_at")

    def type(self, selector: str, text: str) -> dict[str, Any]:
        return self._err("type")

    def screenshot(self) -> dict[str, Any]:
        return self._err("screenshot")

    def back(self) -> dict[str, Any]:
        return self._err("back")

    def forward(self) -> dict[str, Any]:
        return self._err("forward")

    def reload(self) -> dict[str, Any]:
        return self._err("reload")

    def close(self) -> dict[str, Any]:
        return {"ok": True, "action": "close", "closed": True}

    def hover_at(self, x: float, y: float) -> dict[str, Any]:
        return self._err("hover")

    def pick_at(self, x: float, y: float) -> dict[str, Any]:
        return self._err("pick")

    def evaluate(self, js: str) -> Any:
        raise RuntimeError(self.reason)


class _PlaywrightBackend:
    """Lazy Playwright Chromium backend with an isolated profile.

    All Playwright calls run on ONE dedicated thread. The sync API is not
    safe to share across the web server's per-request threads (doing so
    breaks the driver connection with "connection closed" errors), so every
    public method dispatches to the worker and blocks for the result.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._pump, daemon=True,
                                        name="clawd-managed-browser")
        self._worker.start()

    def _pump(self) -> None:
        while True:
            fn, future = self._queue.get()
            try:
                future.set_result(fn())
            except BaseException as exc:  # never kill the worker on page errors
                future.set_exception(exc)

    def _dispatch(self, fn, timeout: float = 120.0):  # noqa: ANN202
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((fn, future))
        return future.result(timeout=timeout)

    def _drop(self) -> None:
        """Tear down the whole stack so the next call starts fresh.

        Prevents a half-dead Playwright connection from poisoning the
        singleton after a transient launch/navigation failure.
        """
        with self._lock:
            self._page = None
            self._context = None
            self._browser = None
            pw, self._pw = self._pw, None
        if pw is not None:
            try:
                pw.stop()
            except Exception:
                pass

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return ("connection closed" in msg or "target closed" in msg
                or "browser closed" in msg or "browser has been closed" in msg)

    def _ensure_page(self, url: str | None = None):  # noqa: ANN202
        from playwright.sync_api import sync_playwright

        with self._lock:
            if self._page is not None:
                try:
                    if not self._page.is_closed():
                        return self._page
                except Exception:
                    pass
                self._page = None
            try:
                if self._pw is None:
                    self._pw = sync_playwright().start()
                if self._context is None:
                    _profile_dir().mkdir(parents=True, exist_ok=True)
                    self._browser = self._pw.chromium.launch(headless=True)
                    self._context = self._browser.new_context(
                        viewport={"width": 1280, "height": 800},
                        user_agent="clawd-managed-browser/1.0",
                    )
                self._page = self._context.new_page()
            except Exception:
                # Never keep a half-initialized stack around (RLock: re-entrant).
                self._drop()
                raise
            if url:
                try:
                    self._page.goto(url, timeout=30000, wait_until="domcontentloaded")
                except Exception:
                    pass
            return self._page

    def _page_or_none(self):  # noqa: ANN202
        with self._lock:
            page = self._page
            try:
                if page is None or page.is_closed():
                    if page is None and self._context is not None:
                        # Dead stack with no page: reset so the next
                        # open/navigate starts fresh instead of lingering.
                        self._drop()
                    return None
            except Exception:
                self._drop()
                return None
            return page

    @_on_browser_thread
    def open(self, url: str) -> dict[str, Any]:
        try:
            page = self._ensure_page(url)
            return {"ok": True, "action": "open", "url": page.url, "title": self._safe_title(page)}
        except Exception as exc:
            if self._is_connection_error(exc):
                # Transient launch failure: reset and retry once with a fresh stack.
                self._drop()
                try:
                    page = self._ensure_page(url)
                    return {"ok": True, "action": "open", "url": page.url,
                            "title": self._safe_title(page)}
                except Exception as exc2:
                    return {"ok": False, "action": "open", "error": f"open failed: {exc2}"}
            return {"ok": False, "action": "open", "error": f"open failed: {exc}"}

    @_on_browser_thread
    def navigate(self, url: str) -> dict[str, Any]:
        try:
            page = self._ensure_page()
            try:
                page.goto(url, timeout=30000, wait_until="domcontentloaded")
            except Exception as exc:
                if self._is_connection_error(exc):
                    self._drop()
                    try:
                        page = self._ensure_page()
                        page.goto(url, timeout=30000, wait_until="domcontentloaded")
                    except Exception as exc2:
                        return {"ok": False, "action": "navigate", "url": url,
                                "error": f"navigation failed: {exc2}"}
                else:
                    return {"ok": False, "action": "navigate", "url": url,
                            "error": f"navigation failed: {exc}"}
            return {"ok": True, "action": "navigate", "url": page.url, "title": self._safe_title(page)}
        except Exception as exc:
            if self._is_connection_error(exc):
                self._drop()
                try:
                    page = self._ensure_page()
                    page.goto(url, timeout=30000, wait_until="domcontentloaded")
                    return {"ok": True, "action": "navigate", "url": page.url,
                            "title": self._safe_title(page)}
                except Exception as exc2:
                    return {"ok": False, "action": "navigate", "url": url,
                            "error": f"navigation failed: {exc2}"}
            return {"ok": False, "action": "navigate", "url": url, "error": str(exc)}

    @_on_browser_thread
    def snapshot(self) -> dict[str, Any]:
        page = self._page_or_none()
        if page is None:
            return {"ok": False, "action": "snapshot", "error": "no open page"}
        try:
            url = page.url
            title = self._safe_title(page)
            text = ""
            try:
                text = page.evaluate(
                    "() => document.body ? document.body.innerText.slice(0, 8000) : ''"
                ) or ""
            except Exception:
                pass
            refs = self._refs(page)
            shot = self._shot(page)
            out: dict[str, Any] = {"ok": True, "action": "snapshot", "url": url, "title": title,
                                   "text": text, "refs": refs}
            if shot:
                out["screenshot"] = shot
            return out
        except Exception as exc:
            return {"ok": False, "action": "snapshot", "error": str(exc)}

    @_on_browser_thread
    def click(self, selector: str) -> dict[str, Any]:
        page = self._page_or_none()
        if page is None:
            return {"ok": False, "action": "click", "error": "no open page"}
        try:
            page.click(selector, timeout=10000)
            return {"ok": True, "action": "click", "selector": selector,
                    "url": page.url, "title": self._safe_title(page)}
        except Exception as exc:
            return {"ok": False, "action": "click", "selector": selector, "error": str(exc)}

    @_on_browser_thread
    def click_at(self, x: float, y: float) -> dict[str, Any]:
        page = self._page_or_none()
        if page is None:
            return {"ok": False, "action": "click_at", "error": "no open page"}
        try:
            vp = page.viewport_size or {"width": 1280, "height": 800}
            px = max(0, min(int(x * vp["width"]), vp["width"] - 1))
            py = max(0, min(int(y * vp["height"]), vp["height"] - 1))
            page.mouse.click(px, py)
            return {"ok": True, "action": "click", "x": x, "y": y,
                    "url": page.url, "title": self._safe_title(page)}
        except Exception as exc:
            return {"ok": False, "action": "click", "error": str(exc)}

    @_on_browser_thread
    def type(self, selector: str, text: str) -> dict[str, Any]:
        page = self._page_or_none()
        if page is None:
            return {"ok": False, "action": "type", "error": "no open page"}
        try:
            page.fill(selector, text, timeout=10000)
            return {"ok": True, "action": "type", "selector": selector, "url": page.url}
        except Exception as exc:
            return {"ok": False, "action": "type", "selector": selector, "error": str(exc)}

    @_on_browser_thread
    def screenshot(self) -> dict[str, Any]:
        page = self._page_or_none()
        if page is None:
            return {"ok": False, "action": "screenshot", "error": "no open page"}
        try:
            shot = self._shot(page)
            if not shot:
                return {"ok": False, "action": "screenshot", "error": "capture failed"}
            return {"ok": True, "action": "screenshot", "url": page.url, "screenshot": shot}
        except Exception as exc:
            return {"ok": False, "action": "screenshot", "error": str(exc)}

    @_on_browser_thread
    def back(self) -> dict[str, Any]:
        return self._nav_action("back")

    @_on_browser_thread
    def forward(self) -> dict[str, Any]:
        return self._nav_action("forward")

    @_on_browser_thread
    def reload(self) -> dict[str, Any]:
        return self._nav_action("reload")

    def _nav_action(self, action: str) -> dict[str, Any]:
        page = self._page_or_none()
        if page is None:
            return {"ok": False, "action": action, "error": "no open page"}
        try:
            if action == "back":
                page.go_back(timeout=15000)
            elif action == "forward":
                page.go_forward(timeout=15000)
            else:
                page.reload(timeout=15000, wait_until="domcontentloaded")
            return {"ok": True, "action": action, "url": page.url, "title": self._safe_title(page)}
        except Exception as exc:
            return {"ok": False, "action": action, "error": str(exc)}

    @_on_browser_thread
    def close(self) -> dict[str, Any]:
        with self._lock:
            try:
                if self._page is not None:
                    try:
                        self._page.close()
                    except Exception:
                        pass
                    self._page = None
            except Exception as exc:
                return {"ok": False, "action": "close", "error": str(exc)}
            return {"ok": True, "action": "close", "closed": True}

    @_on_browser_thread
    def hover_at(self, x: float, y: float) -> dict[str, Any]:
        page = self._page_or_none()
        if page is None:
            return {"ok": False, "action": "hover", "error": "no open page"}
        try:
            vp = page.viewport_size or {"width": 1280, "height": 800}
            info = page.evaluate(
                """({fx, fy, vw, vh}) => {
                  const x = fx * vw, y = fy * vh;
                  const el = document.elementFromPoint(x, y);
                  if (!el) return null;
                  const r = el.getBoundingClientRect();
                  return {selector: (el.tagName.toLowerCase()
                    + (el.id ? '#' + el.id : '')
                    + (el.className && typeof el.className === 'string'
                       ? '.' + el.className.trim().split(/\\s+/).slice(0,2).join('.') : '')),
                    tag: el.tagName.toLowerCase(),
                    box: {x: r.x / vw, y: r.y / vh, w: r.width / vw, h: r.height / vh}};
                }""",
                {"fx": x, "fy": y, "vw": vp["width"], "vh": vp["height"]},
            )
            return {"ok": True, "action": "hover", "element": info}
        except Exception as exc:
            return {"ok": False, "action": "hover", "error": str(exc)}

    @_on_browser_thread
    def pick_at(self, x: float, y: float) -> dict[str, Any]:
        page = self._page_or_none()
        if page is None:
            return {"ok": False, "action": "pick", "error": "no open page"}
        try:
            vp = page.viewport_size or {"width": 1280, "height": 800}
            info = page.evaluate(
                """({fx, fy, vw, vh}) => {
                  const x = fx * vw, y = fy * vh;
                  const el = document.elementFromPoint(x, y);
                  if (!el) return null;
                  const r = el.getBoundingClientRect();
                  const esc = s => s.replace(/[^a-zA-Z0-9_-]/g, c => '\\\\' + c);
                  let sel = el.tagName.toLowerCase();
                  if (el.id) sel += '#' + esc(el.id);
                  else if (el.className && typeof el.className === 'string' && el.className.trim())
                    sel += '.' + el.className.trim().split(/\\s+/).slice(0,2).map(esc).join('.');
                  else if (el.getAttribute('data-testid')) sel += '[data-testid="' + el.getAttribute('data-testid') + '"]';
                  else if (el.name) sel += '[name="' + el.name + '"]';
                  const html = el.outerHTML ? el.outerHTML.slice(0, 2000) : '';
                  // highlight for the confirmation screenshot
                  const prev = el.style.outline;
                  el.style.outline = '3px solid #ff4444';
                  const box = {x: r.x / vw, y: r.y / vh, w: r.width / vw, h: r.height / vh};
                  return {selector: sel, tag: el.tagName.toLowerCase(),
                          outerHTML: html, box: box, url: location.href, prevOutline: prev};
                }""",
                {"fx": x, "fy": y, "vw": vp["width"], "vh": vp["height"]},
            )
            if not info:
                return {"ok": False, "action": "pick", "error": "no element at point"}
            shot = self._shot(page)
            # remove highlight again so later screenshots are clean
            try:
                page.evaluate(
                    "() => { const els = document.querySelectorAll('[style*=\"outline: 3px solid rgb(255, 68, 68)\"]');"
                    " els.forEach(e => e.style.outline = ''); }"
                )
            except Exception:
                pass
            out: dict[str, Any] = {"ok": True, "action": "pick", "url": page.url,
                                   "selector": info.get("selector"),
                                   "outerHTML": (info.get("outerHTML") or "")[:2000],
                                   "box": info.get("box"), "tag": info.get("tag")}
            if shot:
                out["screenshot"] = shot
            return out
        except Exception as exc:
            return {"ok": False, "action": "pick", "error": str(exc)}

    @_on_browser_thread
    def evaluate(self, js: str) -> Any:
        page = self._page_or_none()
        if page is None:
            raise RuntimeError("no open page")
        return page.evaluate(js)

    @staticmethod
    def _safe_title(page) -> str:  # noqa: ANN205
        try:
            return page.title() or ""
        except Exception:
            return ""

    def _shot(self, page) -> str | None:  # noqa: ANN202
        try:
            data = page.screenshot(type="png")
            return base64.b64encode(data).decode("ascii")
        except Exception:
            return None

    @staticmethod
    def _refs(page) -> list[dict[str, Any]]:  # noqa: ANN205
        try:
            items = page.evaluate(
                """() => Array.from(
                  document.querySelectorAll('a, button, input, select, textarea, [role=button], [onclick]'))
                  .slice(0, 100).map((el, i) => {
                    const r = el.getBoundingClientRect();
                    if (r.width === 0 && r.height === 0) return null;
                    let name = (el.innerText || el.value || el.getAttribute('aria-label')
                      || el.getAttribute('placeholder') || el.tagName) + '';
                    return {ref: i, tag: el.tagName.toLowerCase(),
                      role: el.getAttribute('role') || '',
                      name: name.trim().slice(0, 80),
                      selector: el.id ? '#' + el.id
                        : (el.name ? el.tagName.toLowerCase() + '[name="' + el.name + '"]' : null)};
                  }).filter(Boolean)"""
            )
            return items or []
        except Exception:
            return []


@dataclass
class BrowserManager:
    """Single shared page. actor is 'user' or 'agent' for logging."""

    _backend: Any = None
    control: str = "shared"  # "shared" | "user"
    generation: int = 0
    last_actor: str | None = None
    last_action: str | None = None
    current_url: str = ""
    current_title: str = ""
    last_screenshot: str | None = None
    action_log: list[dict[str, Any]] = field(default_factory=list)
    pending_comments: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def _backend_or_default(self) -> Any:
        if self._backend is not None:
            return self._backend
        try:
            import playwright  # noqa: F401
        except ImportError:
            self._backend = _UnavailableBackend()
            return self._backend
        self._backend = _PlaywrightBackend()
        return self._backend

    def _log(self, actor: str, action: str, result: dict[str, Any]) -> None:
        entry = {"seq": len(self.action_log) + 1, "actor": actor,
                 "action": action, "ok": bool(result.get("ok")),
                 "ts": time.time()}
        if result.get("url"):
            entry["url"] = result["url"]
        if result.get("error"):
            entry["error"] = str(result["error"])[:300]
        self.action_log.append(entry)
        if len(self.action_log) > 200:
            del self.action_log[: len(self.action_log) - 200]
        self.last_actor = actor
        self.last_action = action
        if result.get("url"):
            self.current_url = result["url"]
        if result.get("title"):
            self.current_title = result["title"]
        if result.get("screenshot"):
            self.last_screenshot = result["screenshot"]

    def _agent_blocked(self) -> dict[str, Any] | None:
        if self.control == "user":
            return {"ok": False, "error": "user has control, ask user to release"}
        return None

    def _run(self, actor: str, action: str, fn) -> dict[str, Any]:  # noqa: ANN202
        with self.lock:
            if actor == "agent":
                blocked = self._agent_blocked()
                if blocked:
                    blocked["action"] = action
                    self._log(actor, action, blocked)
                    return blocked
            backend = self._backend_or_default()
            try:
                result = fn(backend)
            except Exception as exc:
                result = {"ok": False, "action": action, "error": str(exc)[:500]}
            if not isinstance(result, dict):
                result = {"ok": False, "action": action, "error": "bad backend result"}
            result.setdefault("action", action)
            # Keep the panel fresh: capture a screenshot after page-changing
            # actions so user and agent always see the latest state.
            if result.get("ok") and action in {"open", "navigate", "click",
                                               "type", "back", "forward", "reload"}:
                try:
                    shot = backend.screenshot()
                    if isinstance(shot, dict) and shot.get("screenshot"):
                        result["screenshot"] = shot["screenshot"]
                        result.setdefault("url", shot.get("url", ""))
                except Exception:
                    pass
            if actor == "user":
                self.generation += 1
                result["generation"] = self.generation
            else:
                result["generation"] = self.generation
            result["seq"] = len(self.action_log) + 1
            self._log(actor, action, result)
            return result

    # -- public API (actor defaults to agent for tool calls) --
    def open(self, url: str, actor: str = "agent") -> dict[str, Any]:
        return self._run(actor, "open", lambda b: b.open(url))

    def navigate(self, url: str, actor: str = "agent") -> dict[str, Any]:
        return self._run(actor, "navigate", lambda b: b.navigate(url))

    def snapshot(self, actor: str = "agent") -> dict[str, Any]:
        with self.lock:
            if actor == "agent" and self.control == "user":
                out: dict[str, Any] = {"ok": False, "action": "snapshot",
                                       "error": "user has control, ask user to release",
                                       "generation": self.generation}
                self._log(actor, "snapshot", out)
                return out
            try:
                result = self._backend_or_default().snapshot()
            except Exception as exc:
                result = {"ok": False, "action": "snapshot", "error": str(exc)[:500]}
            if not isinstance(result, dict):
                result = {"ok": False, "action": "snapshot", "error": "bad backend result"}
            result.setdefault("action", "snapshot")
            result["generation"] = self.generation
            result["control"] = self.control
            result["seq"] = len(self.action_log) + 1
            self._log(actor, "snapshot", result)
            return result

    def click(self, selector: str, actor: str = "agent") -> dict[str, Any]:
        return self._run(actor, "click", lambda b: b.click(selector))

    def click_at(self, x: float, y: float, actor: str = "agent") -> dict[str, Any]:
        return self._run(actor, "click", lambda b: b.click_at(x, y))

    def type(self, selector: str, text: str, actor: str = "agent") -> dict[str, Any]:
        return self._run(actor, "type", lambda b: b.type(selector, text))

    def screenshot(self, actor: str = "agent") -> dict[str, Any]:
        return self._run(actor, "screenshot", lambda b: b.screenshot())

    def back(self, actor: str = "agent") -> dict[str, Any]:
        return self._run(actor, "back", lambda b: b.back())

    def forward(self, actor: str = "agent") -> dict[str, Any]:
        return self._run(actor, "forward", lambda b: b.forward())

    def reload(self, actor: str = "agent") -> dict[str, Any]:
        return self._run(actor, "reload", lambda b: b.reload())

    def close(self, actor: str = "agent") -> dict[str, Any]:
        return self._run(actor, "close", lambda b: b.close())

    def hover_at(self, x: float, y: float, actor: str = "user") -> dict[str, Any]:
        with self.lock:
            try:
                result = self._backend_or_default().hover_at(x, y)
            except Exception as exc:
                result = {"ok": False, "action": "hover", "error": str(exc)[:500]}
            return result

    def pick_at(self, x: float, y: float, actor: str = "user") -> dict[str, Any]:
        with self.lock:
            try:
                result = self._backend_or_default().pick_at(x, y)
            except Exception as exc:
                result = {"ok": False, "action": "pick", "error": str(exc)[:500]}
            if isinstance(result, dict) and result.get("url"):
                self.current_url = result["url"]
            if isinstance(result, dict) and result.get("screenshot"):
                self.last_screenshot = result["screenshot"]
            return result if isinstance(result, dict) else {"ok": False, "error": "bad result"}

    def set_control(self, mode: str, actor: str = "user") -> dict[str, Any]:
        with self.lock:
            if mode not in ("shared", "user"):
                return {"ok": False, "error": "mode must be 'shared' or 'user'"}
            self.control = mode
            self._log(actor, f"control:{mode}", {"ok": True, "url": self.current_url})
            return {"ok": True, "control": self.control, "generation": self.generation}

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "url": self.current_url,
                "title": self.current_title,
                "screenshot": self.last_screenshot,
                "control": self.control,
                "generation": self.generation,
                "seq": len(self.action_log),
                "last_actor": self.last_actor,
                "last_action": self.last_action,
                "log": list(self.action_log[-20:]),
                "pending_comments": list(self.pending_comments[-10:]),
            }

    def submit_comment(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Store an element-picker comment. Returns payload the agent can consume."""
        with self.lock:
            comment = str(payload.get("comment", ""))[:2000]
            selector = str(payload.get("selector", ""))[:500]
            outer = str(payload.get("outerHTML", ""))[:2000]
            url = str(payload.get("url", self.current_url))[:2000]
            entry = {"comment": comment, "selector": selector,
                     "outerHTML": outer, "url": url,
                     "screenshot": payload.get("screenshot") or self.last_screenshot,
                     "ts": time.time()}
            self.pending_comments.append(entry)
            if len(self.pending_comments) > 50:
                del self.pending_comments[: len(self.pending_comments) - 50]
            self._log("user", "comment", {"ok": True, "url": url})
            return {"ok": True, "comment": entry}


_MANAGER: BrowserManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_browser_manager() -> BrowserManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = BrowserManager()
        return _MANAGER


def reset_browser_manager(backend: Any = None) -> BrowserManager:
    """Test helper: replace the singleton (pass a fake backend)."""
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = BrowserManager(_backend=backend)
        return _MANAGER
