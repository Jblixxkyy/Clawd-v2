"""Tests for `clawd web --kill` pidfile tracking and process cleanup."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src import web_app
from src.web_app import (
    _is_clawd_web_process,
    _process_cmdline,
    _remove_web_pidfile,
    _web_pidfile,
    _write_web_pidfile,
    kill_web_apps,
)


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    """Isolate ~/.clawd for pidfile tests."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def _spawn_sleeper() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_pidfile_roundtrip(fake_home):
    _write_web_pidfile(8765, "/tmp/ws")
    pf = _web_pidfile(8765)
    assert pf.exists()
    info = json.loads(pf.read_text())
    assert info["pid"] == os.getpid()
    assert info["port"] == 8765
    assert info["workspace"] == "/tmp/ws"
    _remove_web_pidfile(8765)
    assert not pf.exists()


def test_kill_live_process(fake_home, monkeypatch):
    proc = _spawn_sleeper()
    try:
        _write_web_pidfile(8765, "/tmp/ws")
        # point the pidfile at the sleeper instead of ourselves
        info = json.loads(_web_pidfile(8765).read_text())
        info["pid"] = proc.pid
        _web_pidfile(8765).write_text(json.dumps(info))
        # the sleeper's cmdline won't match; force the safety check on
        monkeypatch.setattr(web_app, "_is_clawd_web_process", lambda pid: True)
        summary = kill_web_apps()
        assert proc.poll() is not None, "process should be dead"
        assert not _web_pidfile(8765).exists()
        assert summary["killed"] == [f"port 8765 (pid {proc.pid})"]
    finally:
        if proc.poll() is None:
            proc.kill()


def test_safety_check_rejects_foreign_process(fake_home):
    proc = _spawn_sleeper()
    try:
        _write_web_pidfile(8765, "/tmp/ws")
        info = json.loads(_web_pidfile(8765).read_text())
        info["pid"] = proc.pid
        _web_pidfile(8765).write_text(json.dumps(info))
        # no monkeypatching: the real check must refuse (cmdline has no clawd/web)
        assert _is_clawd_web_process(proc.pid) is False
        summary = kill_web_apps()
        assert proc.poll() is None, "foreign process must survive"
        assert summary["killed"] == []
    finally:
        proc.kill()


def test_never_kills_self(fake_home):
    _write_web_pidfile(8765, "/tmp/ws")  # pidfile points at our own pid
    assert _is_clawd_web_process(os.getpid()) is False
    summary = kill_web_apps()
    assert summary["killed"] == []
    assert _web_pidfile(8765).exists() is False  # cleaned as stale, not killed


def test_stale_pidfile_cleaned(fake_home):
    dead_pid = 999999
    assert _process_cmdline(dead_pid) is None
    _web_pidfile(8765).parent.mkdir(parents=True, exist_ok=True)
    _web_pidfile(8765).write_text(json.dumps({"pid": dead_pid, "port": 8765}))
    summary = kill_web_apps()
    assert not _web_pidfile(8765).exists()
    assert len(summary["not_running"]) == 1
    assert summary["killed"] == []


def test_corrupt_pidfile_removed(fake_home):
    _web_pidfile(8765).parent.mkdir(parents=True, exist_ok=True)
    _web_pidfile(8765).write_text("not-json{{{")
    summary = kill_web_apps()
    assert not _web_pidfile(8765).exists()
    assert summary["killed"] == []


def test_port_filter(fake_home, monkeypatch):
    procs = [_spawn_sleeper(), _spawn_sleeper()]
    try:
        for port, proc in zip((8765, 9000), procs):
            _write_web_pidfile(port, "/tmp/ws")
            info = json.loads(_web_pidfile(port).read_text())
            info["pid"] = proc.pid
            _web_pidfile(port).write_text(json.dumps(info))
        monkeypatch.setattr(web_app, "_is_clawd_web_process", lambda pid: True)
        summary = kill_web_apps(port=9000)
        assert procs[0].poll() is None, "other port must survive"
        assert procs[1].poll() is not None, "target port must die"
        assert _web_pidfile(8765).exists()
        assert not _web_pidfile(9000).exists()
        assert summary["killed"] == [f"port 9000 (pid {procs[1].pid})"]
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()


def test_kill_with_no_pidfiles(fake_home):
    summary = kill_web_apps()
    assert summary == {"killed": [], "not_running": [], "skipped": []}


def test_cli_accepts_kill_flag():
    out = subprocess.run(
        [sys.executable, "-m", "src.cli", "web", "--help"],
        capture_output=True, text=True, cwd=str(Path(".").resolve()),
    )
    assert out.returncode == 0
    assert "--kill" in out.stdout


def test_process_cmdline_self():
    cmd = _process_cmdline(os.getpid())
    assert cmd, "should read our own cmdline"
    assert "pytest" in cmd or "python" in cmd.lower()
