"""Minimal A2A (Agent-to-Agent) JSON-RPC client (brief 010, client only).

Covers the delegation half of the A2A protocol:

1. ``fetch_agent_card`` — GET a URL, parse the JSON, validate the card.
2. ``A2AClient.delegate`` — POST JSON-RPC ``message/send`` (blocking) or
   ``message/stream`` (SSE events) to the card's ``url``, then reassemble
   status updates and artifact chunks into a single ``DelegationResult``.

Stdlib-only: no ``a2a-sdk`` dependency.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class A2AError(Exception):
    """Base class for all A2A client errors."""


class AgentCardError(A2AError):
    """The AgentCard could not be fetched or is malformed."""


class A2AProtocolError(A2AError):
    """The remote agent answered, but not in a usable way."""


class A2ATimeoutError(A2AError):
    """Delegation took longer than the requested timeout."""


# ---------------------------------------------------------------------------
# AgentCard
# ---------------------------------------------------------------------------

_REQUIRED_CARD_FIELDS = ("name", "url")


@dataclass(frozen=True)
class AgentCard:
    """A validated A2A AgentCard (the fields delegation needs)."""

    name: str
    url: str
    description: str = ""
    version: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> "AgentCard":
        if not isinstance(data, dict):
            raise AgentCardError(
                f"AgentCard must be a JSON object, got {type(data).__name__}"
            )
        missing = [
            f
            for f in _REQUIRED_CARD_FIELDS
            if not (isinstance(data.get(f), str) and data[f].strip())
        ]
        if missing:
            raise AgentCardError(
                "AgentCard validation failed: missing required "
                f"field(s): {', '.join(missing)}"
            )
        capabilities = data.get("capabilities", {}) or {}
        if not isinstance(capabilities, dict):
            raise AgentCardError(
                "AgentCard validation failed: 'capabilities' must be an object"
            )
        return cls(
            name=data["name"].strip(),
            url=data["url"].strip(),
            description=data.get("description", "") or "",
            version=data.get("version", "") or "",
            capabilities=capabilities,
        )

    @property
    def supports_streaming(self) -> bool:
        return bool(self.capabilities.get("streaming"))


def fetch_agent_card(card_url: str, *, timeout: float = 30.0) -> AgentCard:
    """Fetch and validate an AgentCard from ``card_url``.

    Raises:
        AgentCardError: network failure, non-JSON body, or a malformed card.
    """
    if not isinstance(card_url, str) or not card_url.strip():
        raise AgentCardError("agent_card_url must be a non-empty string")
    scheme = urllib.parse.urlsplit(card_url.strip()).scheme.lower()
    if scheme not in ("http", "https"):
        raise AgentCardError(
            f"agent_card_url must be an http(s) URL, got {card_url!r}"
        )
    request = urllib.request.Request(
        card_url.strip(), headers={"Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise AgentCardError(
                    f"failed to fetch AgentCard from {card_url}: "
                    f"HTTP {response.status}"
                )
            body = response.read().decode("utf-8", "replace")
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise AgentCardError(
            f"failed to fetch AgentCard from {card_url}: {reason}"
        ) from exc
    except TimeoutError as exc:
        raise AgentCardError(
            f"fetching AgentCard from {card_url} timed out after {timeout}s"
        ) from exc
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise AgentCardError(
            f"AgentCard at {card_url} is not valid JSON: {exc}"
        ) from exc
    return AgentCard.from_dict(data)


# ---------------------------------------------------------------------------
# Result reassembly
# ---------------------------------------------------------------------------


@dataclass
class DelegationResult:
    """The reassembled outcome of one delegation."""

    agent_name: str
    agent_url: str
    state: str  # completed | failed | working | ...
    result_text: str  # concatenated artifact text + final status message
    artifacts: list[dict[str, Any]]  # [{artifactId, name, text}, ...]
    status_updates: list[str]  # states seen, in order


_TERMINAL_STATES = {"completed", "failed", "rejected", "canceled", "cancelled"}


def _parts_text(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get("kind") == "text":
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "".join(texts)


def _assemble_task(task: dict[str, Any], agent_name: str, agent_url: str) -> DelegationResult:
    """Reassemble a final A2A Task object into a DelegationResult."""
    status = task.get("status") or {}
    state = status.get("state", "unknown")
    status_text = _parts_text((status.get("message") or {}).get("parts"))
    artifacts: list[dict[str, Any]] = []
    for artifact in task.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        artifacts.append(
            {
                "artifactId": artifact.get("artifactId", ""),
                "name": artifact.get("name", ""),
                "text": _parts_text(artifact.get("parts")),
            }
        )
    result_text = "".join(a["text"] for a in artifacts)
    if not result_text:
        result_text = status_text
    return DelegationResult(
        agent_name=agent_name,
        agent_url=agent_url,
        state=state,
        result_text=result_text,
        artifacts=artifacts,
        status_updates=[state],
    )


def _assemble_events(
    events: list[tuple[str | None, Any]],
    agent_name: str,
    agent_url: str,
) -> DelegationResult:
    """Reassemble SSE status/artifact-update events into a DelegationResult."""
    status_updates: list[str] = []
    status_texts: list[str] = []
    artifact_texts: dict[str, str] = {}
    artifact_names: dict[str, str] = {}

    for _event_type, payload in events:
        if not isinstance(payload, dict):
            continue
        kind = payload.get("kind")
        if kind == "status-update":
            status = payload.get("status") or {}
            state = status.get("state", "unknown")
            status_updates.append(state)
            text = _parts_text((status.get("message") or {}).get("parts"))
            if text:
                status_texts.append(text)
        elif kind == "artifact-update":
            artifact = payload.get("artifact") or {}
            artifact_id = str(artifact.get("artifactId", ""))
            chunk = _parts_text(artifact.get("parts"))
            if artifact.get("append", True) or artifact_id not in artifact_texts:
                artifact_texts[artifact_id] = artifact_texts.get(artifact_id, "") + chunk
            else:
                artifact_texts[artifact_id] = chunk
            artifact_names.setdefault(artifact_id, artifact.get("name", ""))

    artifacts = [
        {
            "artifactId": artifact_id,
            "name": artifact_names.get(artifact_id, ""),
            "text": text,
        }
        for artifact_id, text in artifact_texts.items()
    ]
    state = status_updates[-1] if status_updates else "unknown"
    result_text = "".join(a["text"] for a in artifacts)
    if not result_text and status_texts:
        result_text = status_texts[-1]
    return DelegationResult(
        agent_name=agent_name,
        agent_url=agent_url,
        state=state,
        result_text=result_text,
        artifacts=artifacts,
        status_updates=status_updates,
    )


def _parse_sse(raw: bytes) -> list[tuple[str | None, Any]]:
    """Parse Server-Sent Events bytes into (event_type, payload) pairs.

    Payloads are decoded as JSON when possible; non-JSON data is kept as
    ``{"text": ...}`` so one bad event cannot poison the whole stream.
    """
    events: list[tuple[str | None, Any]] = []
    event_type: str | None = None
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_type
        if not data_lines:
            return
        raw_data = "\n".join(data_lines)
        try:
            payload: Any = json.loads(raw_data)
        except json.JSONDecodeError:
            payload = {"text": raw_data}
        events.append((event_type, payload))
        event_type = None
        data_lines.clear()

    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.strip():
            flush()
            continue
        if line.startswith(":"):  # SSE comment / heartbeat
            continue
        name, _, value = line.partition(":")
        if name == "event":
            event_type = value.strip() or None
        elif name == "data":
            data_lines.append(value.lstrip(" "))
    flush()
    return events


# ---------------------------------------------------------------------------
# JSON-RPC transport
# ---------------------------------------------------------------------------


def _jsonrpc_request(
    endpoint: str,
    method: str,
    params: dict[str, Any],
    *,
    timeout: float,
) -> tuple[Any, str]:
    """POST one JSON-RPC request; returns (result_or_None, content_type)."""
    payload = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": method,
        "params": params,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise A2AProtocolError(
            f"JSON-RPC {method} to {endpoint} failed: HTTP {exc.code}"
        ) from exc
    except urllib.error.URLError as exc:
        raise A2AProtocolError(
            f"JSON-RPC {method} to {endpoint} failed: "
            f"{getattr(exc, 'reason', exc)}"
        ) from exc
    except TimeoutError as exc:
        raise A2ATimeoutError(
            f"JSON-RPC {method} to {endpoint} timed out after {timeout}s"
        ) from exc

    if content_type.startswith("text/event-stream"):
        return raw, "sse"
    try:
        envelope = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise A2AProtocolError(
            f"JSON-RPC {method} returned non-JSON body: {exc}"
        ) from exc
    if isinstance(envelope, dict) and "error" in envelope:
        error = envelope["error"] or {}
        raise A2AProtocolError(
            f"remote agent error {error.get('code', '?')}: "
            f"{error.get('message', 'unknown error')}"
        )
    return envelope.get("result") if isinstance(envelope, dict) else envelope, "json"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 600.0  # long tasks (e.g. deep research) need room
MAX_TIMEOUT = 1800.0


class A2AClient:
    """Delegates tasks to one A2A agent identified by its AgentCard."""

    def __init__(self, card: AgentCard) -> None:
        self.card = card

    def delegate(self, task_text: str, *, timeout: float = DEFAULT_TIMEOUT) -> DelegationResult:
        """Send ``task_text`` and block until the remote agent finishes.

        Uses ``message/stream`` (SSE reassembly) when the card advertises
        streaming, else blocking ``message/send``.
        """
        if not isinstance(task_text, str) or not task_text.strip():
            raise A2AError("task must be a non-empty string")
        message = {
            "messageId": uuid.uuid4().hex,
            "role": "user",
            "parts": [{"kind": "text", "text": task_text}],
        }
        params = {"message": message}
        if self.card.supports_streaming:
            raw, _ = _jsonrpc_request(
                self.card.url, "message/stream", params, timeout=timeout
            )
            return _assemble_events(
                _parse_sse(raw if isinstance(raw, bytes) else b""),
                self.card.name,
                self.card.url,
            )
        result, _ = _jsonrpc_request(
            self.card.url, "message/send", params, timeout=timeout
        )
        if not isinstance(result, dict):
            raise A2AProtocolError(
                f"message/send returned an unexpected result: {type(result).__name__}"
            )
        return _assemble_task(result, self.card.name, self.card.url)


def _run_with_timeout(
    fn: Any, timeout: float, *, label: str
) -> Any:
    """Run ``fn`` in a daemon thread; raise A2ATimeoutError if it overruns.

    Guarantees the delegation path can never hang the harness: socket
    timeouts bound each request, and this join bounds the whole exchange.
    """
    outcome: dict[str, Any] = {}

    def _target() -> None:
        try:
            outcome["result"] = fn()
        except Exception as exc:  # captured and re-raised on this thread
            outcome["error"] = exc

    thread = threading.Thread(target=_target, name=f"a2a-{label}", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise A2ATimeoutError(
            f"delegation to the remote agent timed out after {timeout:g}s "
            f"(no response within the requested timeout)"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")


def delegate_task(
    card: AgentCard, task_text: str, *, timeout: float = DEFAULT_TIMEOUT
) -> DelegationResult:
    """Convenience: delegate with a hard overall timeout (never hangs)."""
    client = A2AClient(card)
    return _run_with_timeout(
        lambda: client.delegate(task_text, timeout=timeout),
        timeout,
        label="delegate",
    )
