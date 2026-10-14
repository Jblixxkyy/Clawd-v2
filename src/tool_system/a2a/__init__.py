"""A2A (Agent-to-Agent) client (brief 010).

Client-only: fetch + parse an AgentCard, send a task via JSON-RPC
(``message/send`` or the streaming ``message/stream`` variant), and
reassemble incremental status/artifact updates into a final result.

Stdlib-only (urllib) — the ``a2a-sdk`` package did not install cleanly on
this box, so this is the minimal ~150-line JSON-RPC client the brief calls
for. Server mode (accepting delegation) is out of scope for this brief.
"""

from __future__ import annotations

from .client import (
    A2AClient,
    A2AError,
    A2AProtocolError,
    A2ATimeoutError,
    AgentCard,
    AgentCardError,
    DelegationResult,
    delegate_task,
    fetch_agent_card,
)
from .tool import DelegateAgentTool

__all__ = [
    "A2AClient",
    "A2AError",
    "A2AProtocolError",
    "A2ATimeoutError",
    "AgentCard",
    "AgentCardError",
    "DelegationResult",
    "DelegateAgentTool",
    "delegate_task",
    "fetch_agent_card",
]
