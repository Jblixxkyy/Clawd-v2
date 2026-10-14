"""Managed-browser agent tools (dual-control with the web panel user).

All tools drive the same shared page via get_browser_manager(). When the
user toggles "Take control", agent calls return
"user has control, ask user to release" instead of acting.
"""

from __future__ import annotations

from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec
from .browser_manager import get_browser_manager


def _out(name: str, result: dict[str, Any]) -> ToolResult:
    # Screenshot blobs are large; keep a marker + size instead of full data
    # when the result would blow the tool-result budget. Snapshot keeps the
    # screenshot field (panel needs it) but the registry truncates anyway.
    return ToolResult(name=name, output=result, is_error=not result.get("ok"))


def _need_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ToolInputError(f"{field} must be a non-empty string")
    return value


class BrowserOpenTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser_open",
            description="Open a URL in the shared managed browser (isolated Chromium profile).",
            input_schema={"type": "object", "additionalProperties": False,
                          "properties": {"url": {"type": "string"}}, "required": ["url"]},
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        url = _need_str(tool_input.get("url"), "url")
        return _out("browser_open", get_browser_manager().open(url, actor="agent"))


class BrowserNavigateTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser_navigate",
            description="Navigate the shared managed browser to a URL.",
            input_schema={"type": "object", "additionalProperties": False,
                          "properties": {"url": {"type": "string"}}, "required": ["url"]},
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        url = _need_str(tool_input.get("url"), "url")
        return _out("browser_navigate", get_browser_manager().navigate(url, actor="agent"))


class BrowserSnapshotTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser_snapshot",
            description=("Snapshot the shared browser: url, title, visible text, "
                         "element refs, and screenshot. Call after user actions to see new state."),
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            is_read_only=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        return _out("browser_snapshot", get_browser_manager().snapshot(actor="agent"))


class BrowserClickTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser_click",
            description="Click an element in the shared browser by CSS selector or snapshot ref.",
            input_schema={"type": "object", "additionalProperties": False,
                          "properties": {"selector": {"type": "string"},
                                         "ref": {"type": "string"}},
                          "required": ["selector"]},
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        selector = tool_input.get("selector") or tool_input.get("ref")
        selector = _need_str(selector, "selector")
        return _out("browser_click", get_browser_manager().click(selector, actor="agent"))


class BrowserTypeTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser_type",
            description="Type (fill) text into an element in the shared browser.",
            input_schema={"type": "object", "additionalProperties": False,
                          "properties": {"selector": {"type": "string"},
                                         "text": {"type": "string"}},
                          "required": ["selector", "text"]},
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        selector = _need_str(tool_input.get("selector"), "selector")
        text = tool_input.get("text")
        if not isinstance(text, str):
            raise ToolInputError("text must be a string")
        return _out("browser_type", get_browser_manager().type(selector, text, actor="agent"))


class BrowserScreenshotTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser_screenshot",
            description="Capture a screenshot of the shared browser page.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            is_read_only=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        return _out("browser_screenshot", get_browser_manager().screenshot(actor="agent"))


class BrowserBackTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser_back",
            description="Go back in the shared browser history.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        return _out("browser_back", get_browser_manager().back(actor="agent"))


class BrowserForwardTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser_forward",
            description="Go forward in the shared browser history.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        return _out("browser_forward", get_browser_manager().forward(actor="agent"))


class BrowserReloadTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser_reload",
            description="Reload the shared browser page.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        return _out("browser_reload", get_browser_manager().reload(actor="agent"))


class BrowserCloseTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser_close",
            description="Close the shared browser page.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        return _out("browser_close", get_browser_manager().close(actor="agent"))


BROWSER_TOOLS = (
    BrowserOpenTool(),
    BrowserNavigateTool(),
    BrowserSnapshotTool(),
    BrowserClickTool(),
    BrowserTypeTool(),
    BrowserScreenshotTool(),
    BrowserBackTool(),
    BrowserForwardTool(),
    BrowserReloadTool(),
    BrowserCloseTool(),
)
