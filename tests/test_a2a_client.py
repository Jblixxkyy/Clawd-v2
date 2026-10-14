"""Tests for the A2A client (brief 010): AgentCard fetch/parse, JSON-RPC
delegation, SSE artifact reassembly, timeout, and malformed-card handling.

A mock A2A server (http.server in-test) stands in for the remote agent:
GET /card -> AgentCard, POST /rpc -> message/send (JSON Task) or
message/stream (SSE events).
"""

import http.server
import json
import threading
import time
import unittest

from src.tool_system.a2a import (
    AgentCard,
    AgentCardError,
    A2AClient,
    DelegateAgentTool,
    delegate_task,
    fetch_agent_card,
)
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.protocol import ToolCall


def _sse_event(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


class MockA2AHandler(http.server.BaseHTTPRequestHandler):
    """Mock remote A2A agent. Behaviour is driven by server.mode."""

    server_version = "MockA2A/1.0"

    def log_message(self, *args):  # keep test output clean
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        mode = self.server.mode
        if self.path == "/card":
            if mode == "malformed-card":
                self._send_json({"name": "NoUrlAgent"})  # missing required "url"
                return
            if mode == "invalid-json-card":
                body = b"<html>not json</html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            card = {
                "name": "MockAgent",
                "description": "A mock A2A agent for tests",
                "url": f"http://127.0.0.1:{self.server.server_port}/rpc",
                "version": "1.0.0",
                "capabilities": {"streaming": mode == "stream"},
            }
            self._send_json(card)
            return
        self.send_error(404)

    def do_POST(self):
        mode = self.server.mode
        if self.path != "/rpc":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        try:
            request = json.loads(body.decode())
        except Exception:
            request = {}
        method = request.get("method")
        request_id = request.get("id")

        if mode == "slow":
            time.sleep(5)  # longer than the test's timeout
            self._send_json({"jsonrpc": "2.0", "id": request_id, "result": {}})
            return

        if mode == "jsonrpc-error":
            self._send_json(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "Method not found"},
                }
            )
            return

        if method == "message/stream" or mode == "stream":
            events = [
                {
                    "kind": "status-update",
                    "status": {
                        "state": "working",
                        "message": {
                            "parts": [{"kind": "text", "text": "Starting research"}]
                        },
                    },
                },
                {
                    "kind": "artifact-update",
                    "artifact": {
                        "artifactId": "a1",
                        "name": "answer",
                        "parts": [{"kind": "text", "text": "Hello "}],
                        "append": True,
                    },
                },
                {
                    "kind": "artifact-update",
                    "artifact": {
                        "artifactId": "a1",
                        "name": "answer",
                        "parts": [{"kind": "text", "text": "world"}],
                        "append": True,
                    },
                },
                {
                    "kind": "status-update",
                    "status": {
                        "state": "completed",
                        "message": {"parts": [{"kind": "text", "text": "Done"}]},
                    },
                },
            ]
            payload = b"".join(_sse_event(e) for e in events)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        # Default: blocking message/send -> final Task object
        self._send_json(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "id": "task-123",
                    "status": {"state": "completed"},
                    "artifacts": [
                        {
                            "artifactId": "a1",
                            "name": "answer",
                            "parts": [{"kind": "text", "text": "done!"}],
                        }
                    ],
                },
            }
        )


class A2ATestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), MockA2AHandler
        )
        self.server.mode = "stream"
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, name="mock-a2a", daemon=True
        )
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


class TestAgentCardFetch(A2ATestCase):
    def test_fetch_valid_card(self) -> None:
        card = fetch_agent_card(f"{self.base}/card")
        self.assertIsInstance(card, AgentCard)
        self.assertEqual(card.name, "MockAgent")
        self.assertEqual(card.url, f"{self.base}/rpc")
        self.assertTrue(card.supports_streaming)

    def test_malformed_card_missing_url_raises_clear_error(self) -> None:
        self.server.mode = "malformed-card"
        with self.assertRaises(AgentCardError) as ctx:
            fetch_agent_card(f"{self.base}/card")
        message = str(ctx.exception)
        self.assertIn("url", message)
        self.assertIn("missing required", message)

    def test_invalid_json_card_raises_clear_error(self) -> None:
        self.server.mode = "invalid-json-card"
        with self.assertRaises(AgentCardError) as ctx:
            fetch_agent_card(f"{self.base}/card")
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_unreachable_card_url_raises_clear_error(self) -> None:
        with self.assertRaises(AgentCardError) as ctx:
            fetch_agent_card("http://127.0.0.1:1/card", timeout=1)
        self.assertIn("failed to fetch", str(ctx.exception))


class TestStreamReassembly(A2ATestCase):
    def test_card_fetch_then_stream_reassembles_artifacts(self) -> None:
        self.server.mode = "stream"
        card = fetch_agent_card(f"{self.base}/card")
        result = delegate_task(card, "Do the thing", timeout=10)
        self.assertEqual(result.state, "completed")
        self.assertEqual(result.result_text.strip(), "Hello world")
        self.assertEqual(len(result.artifacts), 1)
        self.assertEqual(result.artifacts[0]["artifactId"], "a1")
        self.assertEqual(result.artifacts[0]["text"], "Hello world")
        self.assertIn("working", result.status_updates)
        self.assertIn("completed", result.status_updates)


class TestBlockingSend(A2ATestCase):
    def test_message_send_returns_assembled_result(self) -> None:
        self.server.mode = "blocking"
        card = fetch_agent_card(f"{self.base}/card")
        self.assertFalse(card.supports_streaming)
        result = delegate_task(card, "Do the thing", timeout=10)
        self.assertEqual(result.state, "completed")
        self.assertEqual(result.result_text, "done!")


class TestTimeout(A2ATestCase):
    def test_timeout_returns_clean_error_not_hang(self) -> None:
        self.server.mode = "slow"
        card = fetch_agent_card(f"{self.base}/card", timeout=5)
        client = A2AClient(card)
        started = time.monotonic()
        with self.assertRaises(Exception) as ctx:
            delegate_task(card, "Hurry", timeout=1.0)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 4.0, "delegation hung past the timeout")
        self.assertIn("timed out", str(ctx.exception))


class TestJsonRpcError(A2ATestCase):
    def test_remote_jsonrpc_error_becomes_clean_error(self) -> None:
        self.server.mode = "jsonrpc-error"
        card = fetch_agent_card(f"{self.base}/card", timeout=5)
        with self.assertRaises(Exception) as ctx:
            delegate_task(card, "Do the thing", timeout=10)
        self.assertIn("-32601", str(ctx.exception))
        self.assertIn("Method not found", str(ctx.exception))


class TestDelegateAgentTool(A2ATestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tool = DelegateAgentTool()
        self.context = ToolContext(workspace_root="/tmp")

    def test_full_flow_card_fetch_message_artifact_stream_reassembled(self) -> None:
        self.server.mode = "stream"
        result = self.tool.run(
            {
                "agent_card_url": f"{self.base}/card",
                "task": "Summarise the repo",
                "timeout": 10,
            },
            self.context,
        )
        self.assertFalse(result.is_error)
        output = result.output
        self.assertEqual(output["agent_name"], "MockAgent")
        self.assertEqual(output["state"], "completed")
        self.assertEqual(output["result_text"].strip(), "Hello world")
        self.assertEqual(output["artifacts"][0]["text"], "Hello world")

    def test_timeout_path_returns_clean_error(self) -> None:
        self.server.mode = "slow"
        result = self.tool.run(
            {
                "agent_card_url": f"{self.base}/card",
                "task": "Hurry",
                "timeout": 1.0,
            },
            self.context,
        )
        self.assertTrue(result.is_error)
        self.assertIn("timed out", result.output["error"])

    def test_malformed_card_returns_clear_validation_error(self) -> None:
        self.server.mode = "malformed-card"
        result = self.tool.run(
            {
                "agent_card_url": f"{self.base}/card",
                "task": "Do the thing",
                "timeout": 10,
            },
            self.context,
        )
        self.assertTrue(result.is_error)
        self.assertIn("url", result.output["error"])
        self.assertIn("missing required", result.output["error"])

    def test_registered_in_default_registry(self) -> None:
        registry = build_default_registry()
        tool = registry.get("DelegateAgent")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.spec().name, "DelegateAgent")
        self.assertIn("agent_card_url", tool.spec().input_schema["properties"])
        self.assertIn("task", tool.spec().input_schema["properties"])
        self.assertIn("timeout", tool.spec().input_schema["properties"])

    def test_dispatch_through_registry(self) -> None:
        self.server.mode = "blocking"
        registry = build_default_registry()
        result = registry.dispatch(
            ToolCall(
                name="DelegateAgent",
                input={
                    "agent_card_url": f"{self.base}/card",
                    "task": "Do the thing",
                    "timeout": 10,
                },
            ),
            self.context,
        )
        # Full dispatch pipeline (input repair, schema validation,
        # permissions) runs the tool and returns the reassembled result.
        self.assertEqual(result.name, "DelegateAgent")
        self.assertFalse(result.is_error)
        self.assertEqual(result.output["state"], "completed")
        self.assertEqual(result.output["result_text"], "done!")

    def test_bad_timeout_input_raises(self) -> None:
        from src.tool_system.errors import ToolInputError

        with self.assertRaises(ToolInputError):
            self.tool.run(
                {"agent_card_url": f"{self.base}/card", "task": "x", "timeout": -1},
                self.context,
            )
        with self.assertRaises(ToolInputError):
            self.tool.run(
                {"agent_card_url": "", "task": "x"},
                self.context,
            )


if __name__ == "__main__":
    unittest.main()
