from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from urllib.parse import urlencode

from src.agent import Session
from src.tool_system.agent_loop import AgentLoopCancelled, ToolEvent
from src.tool_system.context import ToolContext
from src.web_app import (
    _AppState,
    _HTML,
    _Job,
    _apply_tool_timing,
    _diff_stat,
    _drive_label,
    _job_emit,
    _make_handler,
    _run_job,
    _session_title,
    _source_signature,
    _tool_event_payload,
    _truncate_conversation_for_retry,
)


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name).resolve()
        self.state = _AppState(
            workspace=self.workspace,
            provider_name="nvidia",
            provider=SimpleNamespace(model="test-model"),
            session=Session.create("nvidia", "test-model"),
            registry=object(),
            tool_context=ToolContext(workspace_root=self.workspace),
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self.state))
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2)
        self.temp.cleanup()

    def test_api_is_accessible_without_a_token(self) -> None:
        request = urllib.request.Request(f"{self.url}/api/state")
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.load(response)
        self.assertEqual(payload["workspace"], str(self.workspace))

    def test_state_includes_workspace_and_model_without_authentication(self) -> None:
        request = urllib.request.Request(
            f"{self.url}/api/state",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.load(response)
        self.assertEqual(payload["workspace"], str(self.workspace))
        self.assertEqual(payload["provider"], "nvidia")
        self.assertEqual(payload["model"], "test-model")

    def test_directory_browser_lists_subdirectories(self) -> None:
        child = self.workspace / "project"
        child.mkdir()
        request = urllib.request.Request(
            f"{self.url}/api/directories?{urlencode({'path': str(self.workspace)})}",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.load(response)
        self.assertEqual(payload["path"], str(self.workspace))
        self.assertEqual(payload["directories"], [{"name": "project", "path": str(child)}])

    def test_directory_search_stays_within_selected_folder_and_ranks_exact_names(self) -> None:
        search_root = self.workspace / "selected-folder"
        exact_match = search_root / "nested" / "PS4"
        partial_match = search_root / "PS4-tools"
        outside_match = self.workspace / "outside" / "PS4"
        exact_match.mkdir(parents=True)
        partial_match.mkdir(parents=True)
        outside_match.mkdir(parents=True)
        request = urllib.request.Request(
            f"{self.url}/api/directories/search?{urlencode({'path': str(search_root), 'query': 'ps4'})}",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.load(response)
        self.assertEqual(
            [directory["path"] for directory in payload["directories"]],
            [str(exact_match), str(partial_match)],
        )
        self.assertEqual(payload["directories"][0]["name"], "PS4")
        self.assertEqual(payload["directories"][0]["parent"], str(exact_match.parent))
        self.assertEqual(
            payload["directories"][0]["drive"],
            _drive_label(exact_match),
        )
        self.assertFalse(payload["truncated"])
        self.assertEqual(payload["skipped"], 0)

    def test_folder_picker_exposes_location_and_optional_scoped_search(self) -> None:
        self.assertIn('id="folderPath"', _HTML)
        self.assertIn('id="goFolderPath"', _HTML)
        self.assertIn('id="folderSearch"', _HTML)
        self.assertIn("/api/directories/search?", _HTML)
        self.assertIn("Search within this folder", _HTML)
        self.assertIn("path:currentDirectory,query:queryText", _HTML)
        self.assertIn('addEventListener("input",()=>{selectedFolderPath=null;', _HTML)

    def test_request_switches_workspace_context_without_authentication(self) -> None:
        selected = self.workspace / "selected"
        selected.mkdir()
        old_session_id = self.state.session.session_id
        request = urllib.request.Request(
            f"{self.url}/api/workspace",
            data=json.dumps({"path": str(selected)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.load(response)
        self.assertEqual(payload["workspace"], str(selected))
        self.assertEqual(self.state.workspace, selected)
        self.assertEqual(self.state.tool_context.workspace_root, selected)
        self.assertEqual(self.state.session.conversation.messages, [])
        self.assertNotEqual(self.state.session.session_id, old_session_id)
        self.assertEqual(self.state.session.workspace, str(selected))

    def test_workspace_switch_rejects_non_directories(self) -> None:
        request = urllib.request.Request(
            f"{self.url}/api/workspace",
            data=json.dumps({"path": str(self.workspace / "missing")}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=2)
        self.assertEqual(raised.exception.code, 400)
        self.assertEqual(self.state.workspace, self.workspace)

    def test_workspace_switch_does_not_change_a_running_jobs_context(self) -> None:
        selected = self.workspace / "selected"
        selected.mkdir()
        job_context = ToolContext(workspace_root=self.workspace)
        job = _Job(job_id="busy", session_id=self.state.session.session_id,
                   session=self.state.session, tool_context=job_context)
        self.state.jobs[job.job_id] = job
        request = urllib.request.Request(
            f"{self.url}/api/workspace",
            data=json.dumps({"path": str(selected)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.load(response)
        self.assertEqual(payload["workspace"], str(selected))
        self.assertIs(job.tool_context, job_context)
        self.assertEqual(job.tool_context.workspace_root, self.workspace)
        self.assertEqual(self.state.workspace, selected)

    def test_new_session_clears_conversation_and_rebuilds_workspace_context(self) -> None:
        self.state.session.conversation.add_user_message("Previous conversation")
        old_session_id = self.state.session.session_id
        request = urllib.request.Request(
            f"{self.url}/api/session/new",
            data=json.dumps({"start": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.load(response)
        self.assertTrue(payload["ok"])
        self.assertEqual(self.state.session.conversation.messages, [])
        self.assertNotEqual(self.state.session.session_id, old_session_id)
        self.assertEqual(self.state.session.workspace, str(self.workspace))
        self.assertEqual(self.state.tool_context.workspace_root, self.workspace)

    def test_new_session_can_start_while_another_chat_is_running(self) -> None:
        old_session_id = self.state.session.session_id
        job = _Job(job_id="busy", session_id=old_session_id, session=self.state.session,
                   tool_context=ToolContext(workspace_root=self.workspace))
        self.state.jobs[job.job_id] = job
        request = urllib.request.Request(
            f"{self.url}/api/session/new",
            data=json.dumps({"start": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.load(response)
        self.assertTrue(payload["ok"])
        self.assertNotEqual(payload["session_id"], old_session_id)
        self.assertEqual(self.state.session.session_id, payload["session_id"])
        self.assertEqual(self.state.session.workspace, str(self.workspace))
        self.assertIs(self.state.jobs[job.job_id], job)

    def test_saved_chats_are_listed_and_can_be_loaded(self) -> None:
        saved = Session.create("nvidia", "test-model")
        saved.session_id = "saved_chat_20261011"
        saved.conversation.add_user_message("Explain this project")
        saved.conversation.add_assistant_message("It is a Python project.")
        saved_workspace = self.workspace / "saved-project"
        saved_workspace.mkdir()
        saved.workspace = str(saved_workspace)
        with patch("pathlib.Path.home", return_value=self.workspace):
            saved.save()
            request = urllib.request.Request(
                f"{self.url}/api/sessions",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                listing = json.load(response)
            self.assertEqual(listing["sessions"][0]["session_id"], saved.session_id)
            self.assertEqual(listing["sessions"][0]["title"], "Explain this project")

            request = urllib.request.Request(
                f"{self.url}/api/session/load",
                data=json.dumps({"session_id": saved.session_id}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                loaded = json.load(response)

        self.assertEqual(loaded["session_id"], saved.session_id)
        self.assertEqual(loaded["workspace"], str(saved_workspace))
        self.assertEqual(
            [(m["role"], m["text"]) for m in loaded["messages"]],
            [
                ("user", "Explain this project"),
                ("assistant", "It is a Python project."),
            ],
        )
        # Timestamps ride along for the chat UI's title tooltips.
        self.assertTrue(all("ts" in m and m["ts"] for m in loaded["messages"]))
        self.assertEqual(self.state.session.session_id, saved.session_id)
        self.assertEqual(self.state.workspace, saved_workspace)
        self.assertEqual(self.state.tool_context.workspace_root, saved_workspace)

    def test_loading_legacy_session_uses_web_app_launch_folder(self) -> None:
        saved = Session.create("nvidia", "test-model")
        saved.session_id = "legacy_session_20261011"
        saved.conversation.add_user_message("Legacy conversation")
        saved.workspace = None
        with patch("pathlib.Path.home", return_value=self.workspace):
            saved.save()
            request = urllib.request.Request(
                f"{self.url}/api/session/load",
                data=json.dumps({"session_id": saved.session_id}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.load(response)
        self.assertEqual(payload["workspace"], str(self.workspace))
        self.assertEqual(self.state.session.workspace, str(self.workspace))
        self.assertEqual(self.state.tool_context.workspace_root, self.workspace)

    def test_session_load_rejects_invalid_session_id(self) -> None:
        request = urllib.request.Request(
            f"{self.url}/api/session/load",
            data=json.dumps({"session_id": "../outside"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=2)
        self.assertEqual(raised.exception.code, 400)

    def test_running_chats_are_isolated_and_can_run_concurrently(self) -> None:
        second = Session.create("nvidia", "test-model")
        second.session_id = "other_session_20261011"
        other_workspace = self.workspace / "other-project"
        other_workspace.mkdir()
        second.workspace = str(other_workspace)
        self.state.sessions[second.session_id] = second
        started = threading.Event()
        release = threading.Event()
        starts_lock = threading.Lock()
        started_count = 0
        contexts = []

        def run_loop(**kwargs):
            nonlocal started_count
            contexts.append(kwargs["tool_context"])
            with starts_lock:
                started_count += 1
                if started_count == 2:
                    started.set()
            self.assertTrue(release.wait(timeout=3))
            kwargs["on_text_chunk"]("Isolated reply")
            return SimpleNamespace(response_text="Isolated reply")

        def start_chat(session_id: str) -> dict:
            request = urllib.request.Request(
                f"{self.url}/api/chat",
                data=json.dumps({"session_id": session_id, "message": f"Message for {session_id}"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                return json.load(response)

        with patch("pathlib.Path.home", return_value=self.workspace), patch(
            "src.web_app.run_agent_loop", side_effect=run_loop
        ):
            first_job = start_chat(self.state.session.session_id)
            second_job = start_chat(second.session_id)
            self.assertTrue(started.wait(timeout=2))
            self.assertEqual(len(self.state.jobs), 2)
            self.assertEqual(
                {job.session_id for job in self.state.jobs.values()},
                {self.state.session.session_id, second.session_id},
            )
            duplicate = urllib.request.Request(
                f"{self.url}/api/chat",
                data=json.dumps({
                    "session_id": self.state.session.session_id,
                    "message": "A concurrent turn in the same chat",
                }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(duplicate, timeout=2)
            self.assertEqual(raised.exception.code, 409)
            request = urllib.request.Request(
                f"{self.url}/api/sessions",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                listed = {entry["session_id"]: entry for entry in json.load(response)["sessions"]}
            self.assertTrue(listed[self.state.session.session_id]["running"])
            self.assertTrue(listed[second.session_id]["running"])
            request = urllib.request.Request(
                f"{self.url}/api/session/load",
                data=json.dumps({"session_id": self.state.session.session_id}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                self.assertEqual(json.load(response)["session_id"], self.state.session.session_id)
            release.set()
            for _ in range(100):
                if all(job.status == "completed" for job in self.state.jobs.values()):
                    break
                threading.Event().wait(0.02)

        self.assertEqual(self.state.jobs[first_job["job_id"]].status, "completed")
        self.assertEqual(self.state.jobs[second_job["job_id"]].status, "completed")
        first_session = self.state.jobs[first_job["job_id"]].session
        self.assertEqual(first_session.conversation.messages[0].content,
                         f"Message for {first_session.session_id}")
        self.assertEqual(second.conversation.messages[0].content, f"Message for {second.session_id}")
        self.assertEqual(len(contexts), 2)
        self.assertIsNot(contexts[0], contexts[1])
        self.assertEqual({context.workspace_root for context in contexts},
                         {self.workspace, other_workspace})

    def test_saved_chat_can_be_deleted_but_running_chat_cannot(self) -> None:
        saved = Session.create("nvidia", "test-model")
        saved.session_id = "delete_me_20261011"
        saved.conversation.add_user_message("Delete this chat")
        busy = Session.create("nvidia", "test-model")
        busy.session_id = "busy_chat_20261011"
        busy.conversation.add_user_message("Still running")
        with patch("pathlib.Path.home", return_value=self.workspace):
            saved.save()
            busy.save()
            busy_job = _Job(
                job_id="busy-delete",
                session_id=busy.session_id,
                session=busy,
                tool_context=ToolContext(workspace_root=self.workspace),
            )
            self.state.jobs[busy_job.job_id] = busy_job
            request = urllib.request.Request(
                f"{self.url}/api/session/delete",
                data=json.dumps({"session_id": busy.session_id}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request, timeout=2)
            self.assertEqual(raised.exception.code, 409)
            self.assertTrue((self.workspace / ".clawd" / "sessions" / f"{busy.session_id}.json").exists())

            self.state.session = saved
            self.state.sessions[saved.session_id] = saved
            request = urllib.request.Request(
                f"{self.url}/api/session/delete",
                data=json.dumps({"session_id": saved.session_id}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                result = json.load(response)
            self.assertTrue(result["ok"])
            self.assertNotEqual(result["session_id"], saved.session_id)
            self.assertEqual(self.state.session.session_id, result["session_id"])
            self.assertEqual(self.state.session.conversation.messages, [])
            self.assertFalse((self.workspace / ".clawd" / "sessions" / f"{saved.session_id}.json").exists())

    def test_saved_chat_listing_skips_a_corrupt_session_file(self) -> None:
        saved = Session.create("nvidia", "test-model")
        saved.session_id = "healthy_chat_20261011"
        saved.conversation.add_user_message("Still here")
        with patch("pathlib.Path.home", return_value=self.workspace):
            saved.save()
            sessions_dir = self.workspace / ".clawd" / "sessions"
            (sessions_dir / "truncated_chat_20261011.json").write_text("", encoding="utf-8")
            (sessions_dir / "garbage_chat_20261011.json").write_text("{not json", encoding="utf-8")

            request = urllib.request.Request(f"{self.url}/api/sessions")
            with urllib.request.urlopen(request, timeout=2) as response:
                listing = json.load(response)

        self.assertEqual(
            [entry["session_id"] for entry in listing["sessions"]],
            [saved.session_id],
        )

    def test_session_save_is_atomic(self) -> None:
        session = Session.create("nvidia", "test-model")
        session.conversation.add_user_message("Persist me")
        with patch("pathlib.Path.home", return_value=self.workspace):
            session.save()
            sessions_dir = self.workspace / ".clawd" / "sessions"
            self.assertTrue((sessions_dir / f"{session.session_id}.json").exists())
            self.assertEqual(list(sessions_dir.glob("*.tmp")), [])
            reloaded = Session.load(session.session_id)
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.conversation.messages[0].content, "Persist me")

    def test_root_serves_browser_ui_without_exposing_state(self) -> None:
        with urllib.request.urlopen(f"{self.url}/", timeout=2) as response:
            body = response.read().decode("utf-8")
        self.assertNotIn("Connect to Clawd", body)
        self.assertIn("id=\"sidebar\"", body)
        self.assertIn("id=\"welcome\"", body)
        self.assertIn("id=\"newSession\"", body)
        self.assertIn("id=\"sessionsList\"", body)
        self.assertIn("className=\"markdown-body\"", body)
        self.assertIn("function renderMarkdown", body)
        self.assertIn("function renderTimelineItem", body)
        self.assertIn("Used", body)
        self.assertIn("function isTableSeparator", body)
        self.assertIn("activity-pulse", body)
        self.assertIn('activity==="typing"?"Typing":"Thinking"', body)
        self.assertNotIn("access token", body.lower())

    def test_chat_uses_existing_agent_loop_and_context(self) -> None:
        job = _Job(
            job_id="job",
            session_id=self.state.session.session_id,
            session=self.state.session,
            tool_context=ToolContext(workspace_root=self.workspace),
        )

        def run_loop(**kwargs):
            self.assertIs(kwargs["provider"], self.state.provider)
            self.assertIs(kwargs["tool_registry"], self.state.registry)
            self.assertIs(kwargs["tool_context"], job.tool_context)
            self.assertTrue(kwargs["stream"])
            kwargs["on_activity"]("thinking")
            self.assertEqual(job.activity, "thinking")
            kwargs["on_text_chunk"]("Hello from ")
            self.assertEqual(job.activity, "typing")
            kwargs["on_event"](
                ToolEvent(
                    kind="tool_use",
                    tool_name="Read",
                    tool_input={"file_path": "src/main.py"},
                    tool_use_id="read-1",
                )
            )
            self.assertEqual(job.activity, "tool")
            kwargs["on_event"](
                ToolEvent(
                    kind="tool_result",
                    tool_name="Read",
                    tool_output={"type": "text", "file": {"filePath": "src/main.py", "content": "print('hi')"}},
                    tool_use_id="read-1",
                )
            )
            self.assertEqual(job.activity, "tool")
            kwargs["on_activity"]("thinking")
            self.assertEqual(job.activity, "thinking")
            kwargs["on_text_chunk"]("the shared agent loop")
            self.assertEqual(job.activity, "typing")
            return SimpleNamespace(response_text="Hello from the shared agent loop")

        with patch("src.web_app.run_agent_loop", side_effect=run_loop):
            _run_job(self.state, job, "Hi")

        self.assertEqual(job.status, "completed")
        self.assertEqual(job.text, "Hello from the shared agent loop")
        self.assertEqual(job.activity, "")
        self.assertEqual(job.events[0]["kind"], "tool_use")
        self.assertEqual(job.events[0]["tool_name"], "Read")
        self.assertEqual(job.events[0]["tool_use_id"], "read-1")
        self.assertIn("src/main.py", job.events[0]["input"])
        self.assertEqual(job.events[1]["kind"], "tool_result")
        self.assertIn("print('hi')", job.events[1]["output"])
        self.assertEqual(
            [item["kind"] for item in job.timeline],
            ["text", "tool", "tool", "text"],
        )
        self.assertEqual(job.timeline[1]["event"]["tool_name"], "Read")
        self.assertEqual(job.timeline[2]["event"]["output"], job.events[1]["output"])
        self.assertEqual(self.state.session.conversation.messages[0].content, "Hi")

    def test_job_endpoint_includes_current_activity(self) -> None:
        timeline = [{"kind": "text", "text": "Working through the request."}]
        job = _Job(
            job_id="active",
            session_id=self.state.session.session_id,
            activity="typing",
            timeline=timeline,
        )
        self.state.jobs[job.job_id] = job
        request = urllib.request.Request(
            f"{self.url}/api/jobs/{job.job_id}",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.load(response)
        self.assertEqual(payload["activity"], "typing")
        self.assertEqual(payload["timeline"], timeline)

    def _post_json(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            return json.load(response)

    def _wait_until(self, condition, timeout: float = 3.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if condition():
                return True
            threading.Event().wait(0.01)
        return condition()

    def test_stop_endpoint_cancels_a_running_turn(self) -> None:
        started = threading.Event()

        def run_loop(**kwargs):
            kwargs["on_text_chunk"]("Partial ")
            started.set()
            deadline = time.time() + 3
            while not kwargs["should_stop"]() and time.time() < deadline:
                threading.Event().wait(0.01)
            raise AgentLoopCancelled("Partial ")

        with patch("pathlib.Path.home", return_value=self.workspace), patch(
            "src.web_app.run_agent_loop", side_effect=run_loop
        ):
            started_job = self._post_json(
                "/api/chat",
                {"session_id": self.state.session.session_id, "message": "Long running task"},
            )
            self.assertTrue(started.wait(timeout=2))
            job_id = started_job["job_id"]

            cancelled = self._post_json(f"/api/jobs/{job_id}/cancel", {})
            self.assertTrue(cancelled["ok"])

            self.assertTrue(self._wait_until(lambda: self.state.jobs[job_id].status == "stopped"))
            job = self.state.jobs[job_id]
            self.assertEqual(job.activity, "")
            self.assertEqual(job.text, "Partial ")
            self.assertTrue(job.cancel_event.is_set())
            self.assertTrue(
                (self.workspace / ".clawd" / "sessions" / f"{job.session_id}.json").exists()
            )

    def test_stop_endpoint_rejects_an_unknown_job(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self._post_json("/api/jobs/missing-job/cancel", {})
        self.assertEqual(raised.exception.code, 404)

    def test_stop_during_a_browser_interaction_unblocks_the_worker(self) -> None:
        started = threading.Event()

        def run_loop(**kwargs):
            started.set()
            with self.assertRaises(AgentLoopCancelled):
                kwargs["tool_context"].permission_handler("Bash", "Run a long command?", None)
            raise AgentLoopCancelled("Stopped during the permission prompt")

        with patch("pathlib.Path.home", return_value=self.workspace), patch(
            "src.web_app.run_agent_loop", side_effect=run_loop
        ):
            started_job = self._post_json(
                "/api/chat",
                {"session_id": self.state.session.session_id, "message": "Run something"},
            )
            self.assertTrue(started.wait(timeout=2))
            job_id = started_job["job_id"]
            self.assertTrue(
                self._wait_until(lambda: self.state.jobs[job_id].interaction is not None)
            )

            cancelled = self._post_json(f"/api/jobs/{job_id}/cancel", {})
            self.assertTrue(cancelled["ok"])

            self.assertTrue(self._wait_until(lambda: self.state.jobs[job_id].status == "stopped"))
            self.assertIsNone(self.state.jobs[job_id].interaction)

    def test_stopped_turn_frees_the_chat_for_the_next_message(self) -> None:
        session = self.state.session
        job = _Job(
            job_id="stopped-job",
            session_id=session.session_id,
            session=session,
            tool_context=ToolContext(workspace_root=self.workspace),
        )
        job.cancel_event.set()
        job.status = "stopped"
        self.state.jobs[job.job_id] = job

        def run_loop(**kwargs):
            kwargs["on_text_chunk"]("Follow-up reply")
            return SimpleNamespace(response_text="Follow-up reply")

        with patch("pathlib.Path.home", return_value=self.workspace), patch(
            "src.web_app.run_agent_loop", side_effect=run_loop
        ):
            started_job = self._post_json(
                "/api/chat",
                {"session_id": session.session_id, "message": "Follow-up message"},
            )
            self.assertIn("job_id", started_job)
            job_id = started_job["job_id"]
            self.assertTrue(self._wait_until(lambda: self.state.jobs[job_id].status == "completed"))

    def test_browser_ui_offers_a_stop_control_for_running_turns(self) -> None:
        with urllib.request.urlopen(f"{self.url}/", timeout=2) as response:
            body = response.read().decode("utf-8")
        self.assertIn('id="stop"', body)
        self.assertIn("/cancel", body)
        self.assertIn('j.status==="stopped"', body)
        self.assertIn('notice.textContent="Stopped."', body)

    def test_browser_ui_model_label_opens_settings_and_drops_redundant_chrome(self) -> None:
        with urllib.request.urlopen(f"{self.url}/", timeout=2) as response:
            body = response.read().decode("utf-8")
        # The composer model label is a button that opens the compact model picker.
        self.assertIn('class="composer-model" id="composerModel" type="button"', body)
        self.assertIn('document.querySelector("#composerModel").onclick=()=>openModelPicker()', body)
        self.assertIn('id="modelPicker"', body)
        self.assertIn("function openModelPicker", body)
        self.assertIn("function chooseModel", body)
        # The full settings dialog stays available behind the gear button only.
        self.assertIn('document.querySelector("#openSettings").onclick=openSettings', body)
        # Removed: topbar model text and assistant avatar squares.
        self.assertNotIn("top-model", body)
        self.assertNotIn('id="model"', body)
        self.assertNotIn("message-avatar", body)
        # The prompt box grows with content instead of showing a resize grabber.
        self.assertIn("#prompt{display:block;width:100%;min-height:44px;max-height:180px;resize:none", body)

    def test_dev_version_endpoint_reports_source_version_and_active_jobs(self) -> None:
        self.state.dev_reload_version = "source-version"
        job = _Job(job_id="active", session_id=self.state.session.session_id)
        self.state.jobs[job.job_id] = job
        with urllib.request.urlopen(f"{self.url}/api/dev-version", timeout=2) as response:
            payload = json.load(response)
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["version"], "source-version")
        self.assertTrue(payload["busy"])

        job.status = "completed"
        with urllib.request.urlopen(f"{self.url}/api/dev-version", timeout=2) as response:
            payload = json.load(response)
        self.assertFalse(payload["busy"])

    def test_source_signature_changes_when_source_content_changes(self) -> None:
        source_root = self.workspace / "src"
        source_root.mkdir()
        source_file = source_root / "web_app.py"
        source_file.write_text("VERSION = 1\n", encoding="utf-8")
        original = _source_signature(source_root)
        source_file.write_text("VERSION = 2\n", encoding="utf-8")
        self.assertNotEqual(_source_signature(source_root), original)

    def test_requests_do_not_write_to_stdout(self) -> None:
        # The browser UI polls constantly, so request logs would flood the
        # terminal. The handler must stay silent for every request.
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            with urllib.request.urlopen(f"{self.url}/api/state", timeout=2) as response:
                payload = json.load(response)
        self.assertEqual(payload["workspace"], str(self.workspace))
        self.assertEqual(captured.getvalue(), "")


    def test_drive_label_handles_posix_and_windows_anchors(self) -> None:
        # POSIX "/" must not be stripped to an empty string; Windows "C:\\" shows as "C:".
        self.assertEqual(_drive_label(PurePosixPath("/tmp/project")), "/")
        self.assertEqual(_drive_label(PureWindowsPath("C:\\project")), "C:")
        self.assertEqual(_drive_label(PureWindowsPath("D:\\")), "D:")

    def test_session_title_prefers_saved_title(self) -> None:
        session = Session.create("nvidia", "test-model")
        session.conversation.add_user_message("First question text")
        session.title = "Custom name"
        self.assertEqual(_session_title(session), "Custom name")
        session.title = None
        self.assertEqual(_session_title(session), "First question text")

    def test_tool_event_payload_includes_edit_diff(self) -> None:
        event = ToolEvent(
            kind="tool_use",
            tool_use_id="u1",
            tool_name="edit",
            tool_input={"file_path": "a.py", "old_string": "x = 1\n", "new_string": "x = 2\n"},
        )
        payload = _tool_event_payload(event)
        self.assertIn("-x = 1", payload["diff"])
        self.assertIn("+x = 2", payload["diff"])
        self.assertIn("@@", payload["diff"])

    def test_tool_event_payload_write_diff_shows_additions(self) -> None:
        event = ToolEvent(
            kind="tool_use",
            tool_use_id="u2",
            tool_name="Write",
            tool_input={"file_path": "b.py", "content": "print(1)\n"},
        )
        payload = _tool_event_payload(event)
        self.assertIn("+print(1)", payload["diff"])

    def test_tool_event_payload_skips_diff_without_strings(self) -> None:
        event = ToolEvent(
            kind="tool_use",
            tool_use_id="u3",
            tool_name="bash",
            tool_input={"command": "ls"},
        )
        payload = _tool_event_payload(event)
        self.assertNotIn("diff", payload)

    def test_job_payload_includes_seq_and_usage(self) -> None:
        job = _Job(job_id="seq-job", session_id=self.state.session.session_id)
        job.usage = {"input_tokens": 1200, "output_tokens": 300}
        _job_emit(job, "text", {"text": "hi"})
        self.state.jobs[job.job_id] = job
        request = urllib.request.Request(f"{self.url}/api/jobs/{job.job_id}")
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.load(response)
        self.assertEqual(payload["seq"], 1)
        self.assertEqual(payload["usage"], {"input_tokens": 1200, "output_tokens": 300})

    def test_sse_stream_replays_events_in_order(self) -> None:
        job = _Job(job_id="sse-job", session_id=self.state.session.session_id)
        self.state.jobs[job.job_id] = job
        _job_emit(job, "activity", {"activity": "thinking"})
        _job_emit(job, "text", {"text": "hello"})
        _job_emit(job, "status", {"status": "completed", "error": None, "usage": None})

        def read_events(last_seq: int) -> list[dict]:
            request = urllib.request.Request(
                f"{self.url}/api/jobs/{job.job_id}/stream?last_seq={last_seq}"
            )
            events: list[dict] = []
            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertIn("text/event-stream", response.headers.get("Content-Type", ""))
                while len(events) < 3:
                    line = response.readline()
                    if not line:
                        break
                    if line.startswith(b"data: "):
                        events.append(json.loads(line[len(b"data: "):]))
            return events

        events = read_events(0)
        self.assertEqual([event["seq"] for event in events], [1, 2, 3])
        self.assertEqual(
            [event["kind"] for event in events], ["activity", "text", "status"]
        )
        self.assertEqual(events[1]["text"], "hello")
        # last_seq filters already-seen events (reconnect replay).
        events = read_events(2)
        self.assertEqual([event["seq"] for event in events], [3])

    def test_sse_stream_rejects_unknown_job(self) -> None:
        request = urllib.request.Request(f"{self.url}/api/jobs/nope/stream")
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=2)
        self.assertEqual(raised.exception.code, 404)

    def test_rename_session_persists_title(self) -> None:
        saved = Session.create("nvidia", "test-model")
        saved.session_id = "rename_me_20261011"
        saved.conversation.add_user_message("Some long question about the codebase here")
        with patch("pathlib.Path.home", return_value=self.workspace):
            saved.save()
            renamed = self._post_json(
                "/api/session/rename",
                {"session_id": saved.session_id, "title": "My custom name"},
            )
            reloaded = Session.load(saved.session_id)
        self.assertEqual(renamed["title"], "My custom name")
        self.assertIsNotNone(reloaded)
        assert reloaded is not None
        self.assertEqual(reloaded.title, "My custom name")
        self.assertEqual(_session_title(reloaded), "My custom name")
        with patch("pathlib.Path.home", return_value=self.workspace):
            request = urllib.request.Request(f"{self.url}/api/sessions")
            with urllib.request.urlopen(request, timeout=2) as response:
                listing = json.load(response)
        titles = {item["session_id"]: item["title"] for item in listing["sessions"]}
        self.assertEqual(titles[saved.session_id], "My custom name")

    def test_rename_rejects_bad_input(self) -> None:
        cases = [
            ({"session_id": "missing_session_zzz", "title": ""}, 400),
            ({"session_id": "../outside", "title": "ok"}, 400),
            ({"session_id": "missing_session_zzz", "title": "x" * 121}, 400),
            ({"session_id": "missing_session_zzz", "title": "ok"}, 404),
        ]
        for payload, code in cases:
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self._post_json("/api/session/rename", payload)
            self.assertEqual(raised.exception.code, code, payload)

    def test_browser_ui_mentions_streaming_and_rename(self) -> None:
        with urllib.request.urlopen(f"{self.url}/", timeout=2) as response:
            body = response.read().decode("utf-8")
        self.assertIn("EventSource", body)
        self.assertIn("/api/jobs/", body)
        self.assertIn("stream?last_seq=", body)
        self.assertIn("/api/session/rename", body)
        self.assertIn("diff-block", body)
        self.assertIn("copy-button", body)

    def test_tool_event_payload_includes_diff_stat(self) -> None:
        event = ToolEvent(
            kind="tool_use",
            tool_name="Edit",
            tool_input={
                "file_path": "a.py",
                "old_string": "x = 1\ny = 2\n",
                "new_string": "x = 1\ny = 3\nz = 4\n",
            },
            tool_use_id="u1",
        )
        payload = _tool_event_payload(event)
        self.assertIn("diff_stat", payload)
        self.assertEqual(payload["diff_stat"], {"added": 2, "removed": 1})

    def test_diff_stat_counts_content_lines_only(self) -> None:
        diff = "@@ -1,2 +1,2 @@\n x = 1\n-y = 2\n+y = 3\n+z = 4\n--- a/a.py\n+++ b/a.py"
        self.assertEqual(_diff_stat(diff), {"added": 2, "removed": 1})
        self.assertEqual(_diff_stat(""), {"added": 0, "removed": 0})

    def test_apply_tool_timing_attaches_elapsed_ms(self) -> None:
        timings: dict = {}
        _apply_tool_timing(timings, {}, ToolEvent(kind="tool_use", tool_name="Read", tool_use_id="r1"))
        self.assertIn("r1", timings)
        time.sleep(0.01)
        payload: dict = {}
        _apply_tool_timing(timings, payload, ToolEvent(kind="tool_result", tool_name="Read", tool_use_id="r1"))
        self.assertIn("elapsed_ms", payload)
        self.assertGreaterEqual(payload["elapsed_ms"], 0)
        self.assertNotIn("r1", timings)

    def test_apply_tool_timing_ignores_unknown_ids(self) -> None:
        payload: dict = {}
        _apply_tool_timing({}, payload, ToolEvent(kind="tool_result", tool_name="Read", tool_use_id="nope"))
        self.assertNotIn("elapsed_ms", payload)

    def test_truncate_for_regenerate_drops_trailing_assistant(self) -> None:
        session = self.state.session
        session.conversation.add_user_message("Hello")
        session.conversation.add_assistant_message("Hi there")
        session.conversation.add_user_message("Again")
        session.conversation.add_assistant_message("Once more")
        prompt = _truncate_conversation_for_retry(session, "regenerate")
        self.assertEqual(prompt, "Again")
        texts = [m.content for m in session.conversation.messages if isinstance(m.content, str)]
        self.assertEqual(texts, ["Hello", "Hi there", "Again"])

    def test_truncate_for_regenerate_needs_a_user_message(self) -> None:
        with self.assertRaises(ValueError):
            _truncate_conversation_for_retry(self.state.session, "regenerate")

    def test_truncate_for_edit_drops_message_and_after(self) -> None:
        session = self.state.session
        session.conversation.add_user_message("First")
        session.conversation.add_assistant_message("Reply one")
        session.conversation.add_user_message("Second")
        session.conversation.add_assistant_message("Reply two")
        prompt = _truncate_conversation_for_retry(session, "edit", "Second")
        self.assertEqual(prompt, "Second")
        texts = [m.content for m in session.conversation.messages if isinstance(m.content, str)]
        self.assertEqual(texts, ["First", "Reply one"])

    def test_truncate_for_edit_rejects_unknown_message(self) -> None:
        self.state.session.conversation.add_user_message("Only")
        with self.assertRaises(ValueError):
            _truncate_conversation_for_retry(self.state.session, "edit", "Missing")

    def _post_retry(self, body: dict):
        request = urllib.request.Request(
            f"{self.url}/api/chat/retry",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)

    def test_chat_retry_endpoint_regenerates(self) -> None:
        session = self.state.session
        session.conversation.add_user_message("Hello")
        session.conversation.add_assistant_message("Hi")
        status, result = self._post_retry({"session_id": session.session_id, "mode": "regenerate"})
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        self.assertEqual(result["prompt"], "Hello")
        self.assertEqual(
            [m.content for m in session.conversation.messages if isinstance(m.content, str)],
            ["Hello"],
        )

    def test_chat_retry_endpoint_edits(self) -> None:
        session = self.state.session
        session.conversation.add_user_message("Old text")
        session.conversation.add_assistant_message("Old reply")
        status, result = self._post_retry(
            {"session_id": session.session_id, "mode": "edit", "message": "Old text"}
        )
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        self.assertEqual(
            [m.content for m in session.conversation.messages if isinstance(m.content, str)],
            [],
        )

    def test_chat_retry_rejects_running_job(self) -> None:
        job = _Job(job_id="busy-1", session_id=self.state.session.session_id, status="running")
        with self.state.lock:
            self.state.jobs["busy-1"] = job
        try:
            status, _ = self._post_retry(
                {"session_id": self.state.session.session_id, "mode": "regenerate"}
            )
        finally:
            with self.state.lock:
                self.state.jobs.pop("busy-1", None)
        self.assertEqual(status, 409)

    def test_chat_retry_rejects_bad_input(self) -> None:
        status, _ = self._post_retry({"session_id": "bad id!", "mode": "regenerate"})
        self.assertEqual(status, 400)
        status, _ = self._post_retry({"session_id": self.state.session.session_id, "mode": "bogus"})
        self.assertEqual(status, 400)
        status, _ = self._post_retry({"session_id": "no_such_session", "mode": "regenerate"})
        self.assertEqual(status, 404)

    def test_run_job_records_elapsed_ms(self) -> None:
        job = _Job(
            job_id="job",
            session_id=self.state.session.session_id,
            session=self.state.session,
            tool_context=ToolContext(workspace_root=self.workspace),
        )

        def run_loop(**kwargs):
            return SimpleNamespace(
                response_text="done",
                usage={"input_tokens": 1, "output_tokens": 1},
            )

        with patch("src.web_app.run_agent_loop", side_effect=run_loop):
            _run_job(self.state, job, "Hi")
        self.assertEqual(job.status, "completed")
        self.assertIsNotNone(job.elapsed_ms)
        self.assertGreaterEqual(job.elapsed_ms, 0)

    def test_browser_ui_has_chat_view_upgrades(self) -> None:
        for marker in [
            'id="sessionSearch"',
            'id="exportChat"',
            'id="scrollBottom"',
            "code-header",
            "code-lang",
            "assistant-thought",
            "tool-slim",
            "tool-diffstat",
            "tool-elapsed",
            "tool-meta",
            "Regenerate response",
            "Edit message",
            "setThoughtFor",
            "highlightCode",
            "submitChatText",
            "regenerateResponse",
            "editUserMessage",
            "/api/chat/retry",
            "scroll-bottom",
            "session-search",
            "message-edit",
            "tok-kw",
            "exportChat",
        ]:
            self.assertIn(marker, _HTML)


if __name__ == "__main__":
    unittest.main()
