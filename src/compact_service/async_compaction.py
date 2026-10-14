"""Async/background compaction engine (brief 022 -- module side only).

Runs a compaction in a background thread at turn-lifecycle checkpoints and
adopts the ready result at a later checkpoint, preserving everything that
arrived after the snapshot. Deliberately loop-agnostic: the later
loop-wiring worker owns the checkpoints and supplies the LLM-backed
``work_fn``; this module only implements the concurrency discipline:

- launch a background task capturing a world-state snapshot + client
  routing state (non-blocking; an already-running task is reused),
- adopt the result while preserving newer messages / tool outputs / state
  updates, rejecting stale or cancelled results,
- protect client-marked evidence (tool outputs) from summarization --
  they survive verbatim,
- run exactly ONE foreground compaction when the adopted result still
  exceeds the limit,
- never launch new background work after a cancelled turn.

Intended call order (for the later loop-wiring worker)::

    manager = AsyncCompactionManager(
        work_fn=my_llm_summarizer,          # (snapshot) -> BackgroundCompactionResult
        config=async_compaction_config(),   # or an explicit dict
    )
    manager.add_status_listener(ui_callback)  # "compacting"/"adopted"/...

    # per turn:
    manager.clear_turn_cancelled()            # turn start
    ...
    # at an eligible checkpoint:
    if manager.is_enabled() and manager.is_eligible(token_count):
        snapshot = manager.capture_snapshot(messages, state=world_state,
                                            routing=routing_state)
        task = manager.start_background_compaction(snapshot)
        # non-blocking; returns the in-flight task if one is already running
    ...
    # at the next checkpoint:
    outcome = manager.adopt_result(task, LiveState(messages, state, routing))
    # outcome.status: "adopted" | "not_ready" | "stale" | "cancelled" |
    #                 "failed" | "no_task"
    # "adopted" -> swap in outcome.adopted (messages/state/routing)
    # "not_ready" -> carry on; adopt at a later checkpoint
    # "stale"/"cancelled"/"failed" -> drop it (optionally launch fresh)

    # over-limit safety (exactly one foreground pass, then continue):
    outcome = manager.adopt_with_foreground_fallback(
        task, live_state, limit=context_limit,
        foreground_compact=my_foreground_compact,  # (AdoptedState) -> AdoptedState
    )

    # on turn cancellation (keeps in-flight work for later reuse):
    manager.mark_turn_cancelled()
    # hard stop:
    manager.cancel_all()

This module is fully unit-testable standalone: inject ``work_fn``,
``estimate_tokens`` and ``foreground_compact`` fakes; no LLM, no loop.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .messages import (
    create_compact_boundary_message,
    create_compact_summary_message,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EVIDENCE_MARK_KEY",
    "STATUS_IDLE",
    "STATUS_COMPACTING",
    "STATUS_ADOPTED",
    "STATUS_FOREGROUND_FALLBACK",
    "STATUS_STALE_REJECTED",
    "STATUS_CANCELLED",
    "STATUS_LAUNCH_REFUSED",
    "STATUS_FAILED",
    "CompactionSnapshot",
    "BackgroundCompactionResult",
    "BackgroundCompactionTask",
    "LiveState",
    "AdoptedState",
    "AdoptOutcome",
    "mark_evidence",
    "is_evidence",
    "default_estimate_tokens",
    "AsyncCompactionManager",
]

# ---------------------------------------------------------------------------
# Status hook values surfaced to the UI (watchable, not mysterious).
# ---------------------------------------------------------------------------
STATUS_IDLE = "idle"
STATUS_COMPACTING = "compacting"
STATUS_ADOPTED = "adopted"
STATUS_FOREGROUND_FALLBACK = "foreground_fallback"
STATUS_STALE_REJECTED = "stale_rejected"
STATUS_CANCELLED = "cancelled"
STATUS_LAUNCH_REFUSED = "launch_refused"
STATUS_FAILED = "failed"

# ---------------------------------------------------------------------------
# Evidence protection.
# ---------------------------------------------------------------------------

#: Metadata key marking a message/tool output as evidence that must survive
#: compaction verbatim instead of being summarized away.
EVIDENCE_MARK_KEY = "protect_from_compaction"


def mark_evidence(message: Any) -> Any:
    """Mark a message (e.g. a tool result) as protected evidence.

    Works on plain dicts and on message objects (attribute or ``metadata``
    dict). Returns the message unchanged otherwise.
    """
    try:
        if isinstance(message, dict):
            meta = message.setdefault("metadata", {})
            if isinstance(meta, dict):
                meta[EVIDENCE_MARK_KEY] = True
            else:
                message["_preserve_verbatim"] = True
        else:
            meta = getattr(message, "metadata", None)
            if isinstance(meta, dict):
                meta[EVIDENCE_MARK_KEY] = True
            else:
                setattr(message, "_preserve_verbatim", True)
    except Exception:
        pass
    return message


def is_evidence(message: Any) -> bool:
    """True when a message was marked via :func:`mark_evidence`."""
    try:
        if isinstance(message, dict):
            meta = message.get("metadata") or {}
            if isinstance(meta, dict) and meta.get(EVIDENCE_MARK_KEY):
                return True
            return bool(message.get("_preserve_verbatim"))
        meta = getattr(message, "metadata", None)
        if isinstance(meta, dict) and meta.get(EVIDENCE_MARK_KEY):
            return True
        return bool(getattr(message, "_preserve_verbatim", False))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Data structures.
# ---------------------------------------------------------------------------


@dataclass
class CompactionSnapshot:
    """World-state snapshot captured when a background compaction launches.

    ``messages`` is a shallow copy -- message identity is preserved so
    :meth:`AsyncCompactionManager.adopt_result` can tell which live messages
    arrived *after* the snapshot.
    """

    messages: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    routing: dict = field(default_factory=dict)
    generation: int = 0
    launched_at: float = field(default_factory=time.time)
    token_count: int = 0


@dataclass
class BackgroundCompactionResult:
    """Result produced by the background ``work_fn``."""

    summary_text: str = ""
    generation: int = 0
    tokens_saved: int = 0
    state_updates: dict = field(default_factory=dict)
    trigger: str = "background"


@dataclass
class BackgroundCompactionTask:
    """Handle for one in-flight (or finished, un-adopted) background task."""

    snapshot: CompactionSnapshot
    future: "concurrent.futures.Future[BackgroundCompactionResult]"
    cancelled: bool = False
    launched_at: float = field(default_factory=time.time)

    def is_done(self) -> bool:
        try:
            return bool(self.future.done())
        except Exception:
            return False


@dataclass
class LiveState:
    """Live conversation state presented at an adoption checkpoint."""

    messages: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    routing: dict = field(default_factory=dict)


@dataclass
class AdoptedState:
    """State produced by a successful adoption."""

    messages: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    routing: dict = field(default_factory=dict)
    generation: int = 0


@dataclass
class AdoptOutcome:
    """Outcome of :meth:`AsyncCompactionManager.adopt_result`."""

    status: str  # adopted|not_ready|stale|cancelled|failed|no_task
    adopted: Optional[AdoptedState] = None
    result: Optional[BackgroundCompactionResult] = None
    detail: str = ""
    foreground_ran: bool = False


# ---------------------------------------------------------------------------
# Token estimation (injectable; heuristic default).
# ---------------------------------------------------------------------------


def _message_text_len(message: Any) -> int:
    try:
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    total += len(text) if isinstance(text, str) else len(str(block))
                else:
                    text = getattr(block, "text", None)
                    total += len(text) if isinstance(text, str) else len(str(block))
            return total
        return len(str(content))
    except Exception:
        return 0


def default_estimate_tokens(messages: list) -> int:
    """Rough token estimate: ~4 chars per token, minimum 1 per message."""
    total = 0
    for message in messages or []:
        total += max(1, _message_text_len(message) // 4)
    return total


# ---------------------------------------------------------------------------
# Manager.
# ---------------------------------------------------------------------------


class AsyncCompactionManager:
    """Background compaction lifecycle manager. See module docstring for the
    intended call order.

    All methods are thread-safe for the manager's own bookkeeping; the
    background ``work_fn`` runs on a single worker thread and must not
    mutate the live conversation (it only reads its snapshot).
    """

    def __init__(
        self,
        *,
        config: Optional[dict] = None,
        work_fn: Optional[Callable[[CompactionSnapshot], BackgroundCompactionResult]] = None,
        estimate_tokens: Optional[Callable[[list], int]] = None,
        executor: Optional[concurrent.futures.Executor] = None,
    ) -> None:
        if config is None:
            from ..config import async_compaction_config

            config = async_compaction_config()
        self._config: dict = dict(config)
        self._work_fn = work_fn
        self._estimate_tokens = estimate_tokens or default_estimate_tokens
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="async-compact"
        )
        self._owns_executor = executor is None
        self._lock = threading.Lock()
        self._task: Optional[BackgroundCompactionTask] = None
        self._turn_cancelled = False
        self._generation = 0
        self._status = STATUS_IDLE
        self._status_listeners: list[Callable[[str], None]] = []

    # -- config ---------------------------------------------------------

    def is_enabled(self) -> bool:
        """Master switch (default OFF)."""
        try:
            return bool(self._config.get("enabled", False))
        except Exception:
            return False

    def is_eligible(self, token_count: int) -> bool:
        """Eligible for a background launch: enabled and at/over min_tokens."""
        try:
            if not self.is_enabled():
                return False
            min_tokens = int(self._config.get("min_tokens", 0) or 0)
            return int(token_count or 0) >= min_tokens
        except Exception:
            return False

    @property
    def reserve_tokens(self) -> int:
        try:
            return int(self._config.get("reserve_tokens", 16_000) or 0)
        except Exception:
            return 16_000

    @property
    def shrink_steps(self) -> list:
        try:
            steps = self._config.get("shrink_steps") or [0.7, 0.5, 0.35]
            return [float(s) for s in steps]
        except Exception:
            return [0.7, 0.5, 0.35]

    def foreground_target_limit(self, limit: int) -> int:
        """Target limit for the single foreground fallback pass (steps[0])."""
        try:
            return int(limit * self.shrink_steps[0])
        except Exception:
            return int(limit * 0.7)

    # -- status hook (UI) -----------------------------------------------

    @property
    def status(self) -> str:
        return self._status

    def add_status_listener(self, listener: Callable[[str], None]) -> None:
        """UI hook: called with each status transition (compacting… /
        adopted / foreground_fallback / ...). Never raises."""
        with self._lock:
            self._status_listeners.append(listener)

    def remove_status_listener(self, listener: Callable[[str], None]) -> None:
        with self._lock:
            if listener in self._status_listeners:
                self._status_listeners.remove(listener)

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status
            listeners = list(self._status_listeners)
        for listener in listeners:
            try:
                listener(status)
            except Exception:
                logger.debug("async_compaction status listener failed", exc_info=True)

    # -- cancellation discipline ----------------------------------------

    def mark_turn_cancelled(self) -> None:
        """Record that the current turn was cancelled.

        Blocks *new* background launches; an already-running compaction is
        kept for later reuse (it is NOT cancelled here).
        """
        with self._lock:
            self._turn_cancelled = True

    def clear_turn_cancelled(self) -> None:
        """Clear the cancelled-turn flag (call at turn start)."""
        with self._lock:
            self._turn_cancelled = False

    @property
    def turn_cancelled(self) -> bool:
        with self._lock:
            return self._turn_cancelled

    def cancel_all(self) -> None:
        """Hard stop: cancel the in-flight task and drop it.

        The cancelled task's result will be rejected if anyone still tries
        to adopt it. Does not clear the cancelled-turn flag.
        """
        with self._lock:
            task = self._task
            self._task = None
        if task is not None:
            task.cancelled = True
            try:
                task.future.cancel()
            except Exception:
                pass
        self._set_status(STATUS_CANCELLED)

    # -- lifecycle ------------------------------------------------------

    def capture_snapshot(
        self,
        messages: list,
        *,
        state: Optional[dict] = None,
        routing: Optional[dict] = None,
        token_count: Optional[int] = None,
    ) -> CompactionSnapshot:
        """Capture a world-state snapshot for a background launch.

        Copies the message *list* (message identity is preserved for
        new-message detection at adoption); ``state``/``routing`` dicts are
        shallow-copied.
        """
        with self._lock:
            generation = self._generation
        msgs = list(messages or [])
        return CompactionSnapshot(
            messages=msgs,
            state=dict(state or {}),
            routing=dict(routing or {}),
            generation=generation,
            launched_at=time.time(),
            token_count=(
                token_count
                if token_count is not None
                else self._estimate_tokens(msgs)
            ),
        )

    def is_compaction_running(self) -> bool:
        """True when a background task is in flight (not done, not cancelled)."""
        with self._lock:
            task = self._task
        return task is not None and not task.cancelled and not task.is_done()

    def start_background_compaction(
        self,
        snapshot: CompactionSnapshot,
        work_fn: Optional[Callable[[CompactionSnapshot], BackgroundCompactionResult]] = None,
    ) -> Optional[BackgroundCompactionTask]:
        """Launch a background compaction (non-blocking).

        Returns the already-running (or finished-but-unadopted) task when
        one exists -- reuse, never a second concurrent compaction. Returns
        None when the launch is refused: disabled, below ``min_tokens``,
        or the turn was cancelled (``mark_turn_cancelled``).

        Args:
            snapshot: from :meth:`capture_snapshot`.
            work_fn: overrides the constructor's ``work_fn`` for this launch.

        Raises:
            ValueError: when no work function is available.
        """
        with self._lock:
            existing = self._task
            turn_cancelled = self._turn_cancelled
        # Reuse in-flight (or un-adopted finished) work -- never double-launch.
        if existing is not None and not existing.cancelled:
            return existing
        if not self.is_enabled():
            self._set_status(STATUS_LAUNCH_REFUSED)
            logger.debug("async_compaction launch refused: disabled")
            return None
        if not self.is_eligible(snapshot.token_count):
            self._set_status(STATUS_LAUNCH_REFUSED)
            logger.debug(
                "async_compaction launch refused: below min_tokens (%d)",
                snapshot.token_count,
            )
            return None
        # Cancellation discipline: a cancelled turn never launches NEW work.
        if turn_cancelled:
            self._set_status(STATUS_LAUNCH_REFUSED)
            logger.debug("async_compaction launch refused: turn cancelled")
            return None
        fn = work_fn or self._work_fn
        if fn is None:
            raise ValueError(
                "start_background_compaction requires a work_fn "
                "(constructor or per-call); none was provided"
            )

        def _runner() -> BackgroundCompactionResult:
            return fn(snapshot)

        future = self._executor.submit(_runner)
        task = BackgroundCompactionTask(snapshot=snapshot, future=future)
        with self._lock:
            # Recheck under lock: adopt/cancel may have raced us.
            if self._task is not None and not self._task.cancelled:
                try:
                    future.cancel()
                except Exception:
                    pass
                return self._task
            self._task = task
        self._set_status(STATUS_COMPACTING)
        return task

    def adopt_result(
        self, task: Optional[BackgroundCompactionTask], live_state: LiveState
    ) -> AdoptOutcome:
        """Adopt a ready background result at a checkpoint (non-blocking).

        - ``"not_ready"``: still running -- carry on, adopt at a later
          checkpoint. Nothing is blocked or mutated.
        - ``"adopted"``: ``outcome.adopted`` holds the new messages/state/
          routing. Messages and state updates that arrived *after* the
          snapshot are preserved verbatim; client-marked evidence from the
          summarized region survives verbatim after the summary.
        - ``"stale"``: result belongs to an older generation or is not the
          current task -- discarded.
        - ``"cancelled"`` / ``"failed"`` / ``"no_task"``: nothing to adopt.
        """
        if task is None:
            return AdoptOutcome(status="no_task", detail="no task to adopt")
        if task.cancelled:
            self._drop_task_if_current(task)
            self._set_status(STATUS_CANCELLED)
            return AdoptOutcome(status="cancelled", detail="task was cancelled")
        with self._lock:
            current = self._task
            generation = self._generation
        if current is not task:
            self._set_status(STATUS_STALE_REJECTED)
            return AdoptOutcome(
                status="stale", detail="task is not the current compaction task"
            )
        if not task.is_done():
            return AdoptOutcome(status="not_ready", detail="compaction still running")
        try:
            result = task.future.result()
        except Exception as exc:  # background work blew up
            self._drop_task_if_current(task)
            self._set_status(STATUS_FAILED)
            logger.warning("async_compaction background work failed: %s", exc)
            return AdoptOutcome(status="failed", detail=f"background work failed: {exc}")
        if not isinstance(result, BackgroundCompactionResult):
            self._drop_task_if_current(task)
            self._set_status(STATUS_FAILED)
            return AdoptOutcome(
                status="failed",
                detail=f"work_fn returned {type(result).__name__}, not BackgroundCompactionResult",
            )
        if result.generation != generation:
            # A newer adoption already moved the generation forward.
            self._drop_task_if_current(task)
            self._set_status(STATUS_STALE_REJECTED)
            return AdoptOutcome(
                status="stale",
                detail=(
                    f"stale result: generation {result.generation} "
                    f"!= current {generation}"
                ),
            )

        adopted = self._build_adopted(task, result, live_state)
        with self._lock:
            self._generation = generation + 1
            adopted.generation = self._generation
            if self._task is task:
                self._task = None
        self._set_status(STATUS_ADOPTED)
        return AdoptOutcome(status="adopted", adopted=adopted, result=result)

    def adopt_with_foreground_fallback(
        self,
        task: Optional[BackgroundCompactionTask],
        live_state: LiveState,
        *,
        limit: int,
        foreground_compact: Callable[[AdoptedState], AdoptedState],
        estimate_tokens: Optional[Callable[[list], int]] = None,
    ) -> AdoptOutcome:
        """Adopt, then run exactly ONE foreground compaction if still over limit.

        The adopted state must leave ``reserve_tokens`` headroom under
        ``limit``; when it does not, ``foreground_compact`` is called exactly
        once and its result replaces the adopted messages. When the
        foreground pass still exceeds the limit we continue anyway -- never
        a second pass.
        """
        outcome = self.adopt_result(task, live_state)
        if outcome.status != "adopted" or outcome.adopted is None:
            return outcome
        estimate = estimate_tokens or self._estimate_tokens
        try:
            tokens = int(estimate(outcome.adopted.messages))
        except Exception:
            tokens = 0
        effective_limit = max(0, int(limit) - self.reserve_tokens)
        if tokens <= effective_limit:
            return outcome
        logger.info(
            "async_compaction adopted result still over limit "
            "(%d > %d); running single foreground fallback",
            tokens,
            effective_limit,
        )
        try:
            compacted = foreground_compact(outcome.adopted)
        except Exception as exc:
            logger.warning("async_compaction foreground fallback failed: %s", exc)
            return outcome
        if isinstance(compacted, AdoptedState):
            outcome.adopted = compacted
        outcome.foreground_ran = True
        self._set_status(STATUS_FOREGROUND_FALLBACK)
        return outcome

    def shutdown(self, wait: bool = False) -> None:
        """Shut down the worker executor (only if this manager owns it)."""
        if self._owns_executor:
            try:
                self._executor.shutdown(wait=wait, cancel_futures=True)
            except Exception:
                pass

    # -- internals ------------------------------------------------------

    def _drop_task_if_current(self, task: BackgroundCompactionTask) -> None:
        with self._lock:
            if self._task is task:
                self._task = None

    def _build_adopted(
        self,
        task: BackgroundCompactionTask,
        result: BackgroundCompactionResult,
        live_state: LiveState,
    ) -> AdoptedState:
        snapshot = task.snapshot
        snapshot_ids = {id(m) for m in snapshot.messages}

        # Messages/tool outputs/state that arrived AFTER the snapshot are
        # preserved verbatim -- never summarized away.
        live_messages = list(live_state.messages or [])
        new_messages = [m for m in live_messages if id(m) not in snapshot_ids]
        new_ids = {id(m) for m in new_messages}

        # Client-marked evidence inside the summarized region survives
        # verbatim (exempt from summarization).
        evidence = [
            m
            for m in snapshot.messages
            if is_evidence(m) and id(m) not in new_ids
        ]

        boundary = create_compact_boundary_message(
            trigger="background",
            pre_compact_token_count=snapshot.token_count,
        )
        summary = create_compact_summary_message(result.summary_text or "")

        adopted_messages = [boundary, summary] + evidence + new_messages

        # World-state updates that arrived after the snapshot win; the
        # background result may additionally patch state.
        adopted_state = dict(live_state.state or {})
        try:
            adopted_state.update(result.state_updates or {})
        except Exception:
            pass
        adopted_routing = dict(live_state.routing or {})

        return AdoptedState(
            messages=adopted_messages,
            state=adopted_state,
            routing=adopted_routing,
        )
