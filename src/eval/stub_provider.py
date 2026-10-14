"""Stub provider for --provider stub smoke mode.

Exists to validate the harness plumbing (runner -> trace -> grader -> report)
without credentials or network. It returns canned no-op responses, so it is
NOT a measure of agent quality — every stub run should record zero tool calls.
"""

from __future__ import annotations

from typing import Any, Generator, Optional

from ..providers.base import BaseProvider, ChatResponse


class StubProvider(BaseProvider):
    """A provider that never calls a model: canned text, no tool calls."""

    def __init__(self, model: str = "stub/smoke", **kwargs: Any) -> None:
        super().__init__(api_key="stub", base_url=None, model=model)

    def chat(
        self,
        messages: list[Any],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        return ChatResponse(
            content="stub: harness smoke run, no model consulted.",
            model=self.model or "stub/smoke",
            usage={"input_tokens": 4, "output_tokens": 8},
            finish_reason="stop",
            tool_uses=None,
        )

    def chat_stream(
        self,
        messages: list[Any],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        yield "stub: harness smoke run, no model consulted."

    def get_available_models(self) -> list[str]:
        return ["stub/smoke"]
