"""MCP server inventory awareness tests (brief 013).

Verifies that connected MCP servers and their tool inventories are injected
into the model context at session start, refresh on mid-session changes,
stay silent when no servers are connected, and produce helpful errors for
unknown server/tool names (mirroring the Skill tool pattern).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.outputStyles.base_prompt import BASE_SYSTEM_PROMPT
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.delta_instructions import DeltaInstructionTracker
from src.tool_system.system_prompt import build_mcp_section
from src.tool_system.tools.mcp import MCPTool


class FakeMcpClient:
    """Minimal client matching the MCPClient protocol."""

    def __init__(self, tools: list[str], resources: list[dict[str, Any]] | None = None):
        self._tools = tools
        self._resources = resources

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        return {"tool": tool_name, "args": args}

    def list_tools(self) -> list[str]:
        return self._tools

    def list_resources(self) -> list[dict[str, Any]]:
        return self._resources if self._resources is not None else []


class ExplodingMcpClient:
    """Client whose tool listing fails; must not break the inventory."""

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        return {"ok": True}

    def list_tools(self) -> list[str]:
        raise RuntimeError("server exploded")


def _ctx(*, servers: dict[str, Any] | None = None) -> ToolContext:
    ctx = ToolContext(workspace_root=Path.cwd())
    if servers:
        ctx.mcp_clients.update(servers)
    return ctx


class TestBuildMcpSection(unittest.TestCase):
    def test_empty_state_returns_empty_string(self) -> None:
        # Prompt diet: no block, no error when no servers are connected.
        self.assertEqual(build_mcp_section(_ctx()), "")

    def test_inventory_present_when_server_connected(self) -> None:
        section = build_mcp_section(
            _ctx(servers={"files": FakeMcpClient(["read_file", "write_file"])})
        )
        self.assertIn("## MCP Servers", section)
        self.assertIn("- files: read_file, write_file", section)
        self.assertIn('"server": "<name>"', section)

    def test_servers_sorted_and_all_tools_listed(self) -> None:
        section = build_mcp_section(
            _ctx(
                servers={
                    "zebra": FakeMcpClient(["z_tool"]),
                    "alpha": FakeMcpClient(["a_tool", "b_tool"]),
                }
            )
        )
        self.assertLess(section.index("- alpha:"), section.index("- zebra:"))
        self.assertIn("a_tool, b_tool", section)

    def test_exploding_client_does_not_break_block(self) -> None:
        section = build_mcp_section(
            _ctx(
                servers={
                    "bad": ExplodingMcpClient(),
                    "good": FakeMcpClient(["works"]),
                }
            )
        )
        self.assertIn("## MCP Servers", section)
        self.assertIn("- bad: (tool list unavailable)", section)
        self.assertIn("- good: works", section)

    def test_resources_listed_where_cheap(self) -> None:
        section = build_mcp_section(
            _ctx(
                servers={
                    "res": FakeMcpClient(
                        ["q"],
                        resources=[{"uri": "file:///a.txt"}, {"uri": "file:///b.txt"}],
                    )
                }
            )
        )
        self.assertIn("resources: file:///a.txt, file:///b.txt", section)


class TestMCPToolStillPresent(unittest.TestCase):
    def test_mcp_tool_registered_in_default_registry(self) -> None:
        # Empty state: the MCP tool is still available even with no servers.
        registry = build_default_registry()
        names = {s.name.lower() for s in registry.list_specs()}
        self.assertIn("mcp", names)

    def test_mcp_call_with_servers_connected(self) -> None:
        ctx = _ctx(servers={"srv": FakeMcpClient(["x"])})
        out = MCPTool().run({"server": "srv", "tool": "x", "input": {"a": 1}}, ctx).output
        self.assertEqual(out["output"]["args"]["a"], 1)


class TestMCPToolUnknownNameErrors(unittest.TestCase):
    def test_unknown_server_lists_available(self) -> None:
        ctx = _ctx(servers={"srv": FakeMcpClient(["x"])})
        out = MCPTool().run({"server": "nope", "tool": "x"}, ctx)
        self.assertTrue(out.is_error)
        self.assertIn("mcp server not connected: nope", out.output["error"])
        self.assertIn("Available MCP servers: srv", out.output["error"])

    def test_unknown_server_with_none_connected(self) -> None:
        out = MCPTool().run({"server": "nope", "tool": "x"}, _ctx())
        self.assertTrue(out.is_error)
        self.assertIn("No MCP servers are connected", out.output["error"])

    def test_unknown_tool_lists_available(self) -> None:
        ctx = _ctx(servers={"srv": FakeMcpClient(["x", "y"])})
        out = MCPTool().run({"server": "srv", "tool": "bogus"}, ctx)
        self.assertTrue(out.is_error)
        self.assertIn("unknown mcp tool: srv.bogus", out.output["error"])
        self.assertIn("Available tools on srv: x, y", out.output["error"])

    def test_exploding_client_still_callable(self) -> None:
        # When list_tools() fails, the call goes through instead of erroring.
        ctx = _ctx(servers={"srv": ExplodingMcpClient()})
        out = MCPTool().run({"server": "srv", "tool": "anything"}, ctx)
        self.assertFalse(out.is_error)


class TestMcpSessionStartAndRefresh(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _tracker(self, ctx: ToolContext) -> DeltaInstructionTracker:
        return DeltaInstructionTracker(BASE_SYSTEM_PROMPT, ctx, {"read"})

    def test_session_start_injects_inventory(self) -> None:
        ctx = _ctx(servers={"files": FakeMcpClient(["read_file"])})
        ctx.cwd = self.root
        full = self._tracker(ctx).full_prompt()
        self.assertIn("## MCP Servers", full)
        self.assertIn("- files: read_file", full)

    def test_session_start_empty_state_has_no_block(self) -> None:
        ctx = _ctx()
        ctx.cwd = self.root
        full = self._tracker(ctx).full_prompt()
        self.assertNotIn("## MCP Servers", full)

    def test_mid_session_add_refreshes_inventory(self) -> None:
        ctx = _ctx()
        ctx.cwd = self.root
        tracker = self._tracker(ctx)
        tracker.full_prompt()
        # A server connects mid-session (e.g. via /mcp add + connect).
        ctx.mcp_clients["new"] = FakeMcpClient(["do_thing"])
        system2, report = tracker.next_system(turn=1)
        self.assertTrue(report.changed)
        names = [d.name for d in report.deltas]
        self.assertIn("mcp", names)
        delta = next(d for d in report.deltas if d.name == "mcp")
        self.assertEqual(delta.kind, "added")
        self.assertIn("## mcp (added)", system2)
        self.assertIn("- new: do_thing", system2)

    def test_mid_session_remove_refreshes_inventory(self) -> None:
        ctx = _ctx(servers={"files": FakeMcpClient(["read_file"])})
        ctx.cwd = self.root
        tracker = self._tracker(ctx)
        tracker.full_prompt()
        ctx.mcp_clients.clear()
        system2, report = tracker.next_system(turn=1)
        self.assertTrue(report.changed)
        delta = next(d for d in report.deltas if d.name == "mcp")
        self.assertEqual(delta.kind, "removed")
        self.assertIn("## mcp (removed -- no longer applies)", system2)

    def test_no_change_no_delta(self) -> None:
        ctx = _ctx(servers={"files": FakeMcpClient(["read_file"])})
        ctx.cwd = self.root
        tracker = self._tracker(ctx)
        tracker.full_prompt()
        system2, report = tracker.next_system(turn=1)
        self.assertFalse(report.changed)
        self.assertNotIn("## mcp", system2)


if __name__ == "__main__":
    unittest.main()
