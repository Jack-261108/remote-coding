"""External Ghostty session input service.

The single orchestration point for everything described in
``docs/specs/2026-08-03-external-ghostty-input-design.md`` §4-9: pairing a
bound external Claude session to a Ghostty terminal surface, activating an
input target from ``/list``, and sending ordinary Telegram text (or an
unregistered slash command) into the verified Claude TUI — or enqueuing it
while Claude is busy and draining later.

Why a service: handlers must NOT touch ``GhosttyTerminalAdapter``,
``LocalProcessProbe``, ``ExternalBindingStore`` internals, the pairing
registry, or ``osascript`` directly. They route through this service, which
owns the per-session input lock, the in-flight marker, the drain task and the
security-ordered checks (owner → binding generation → process/TTY → terminal).

Concurrency invariants (design §6 / §9):
  * Per-session sending is serialised by ``_input_locks`` (a
    ``RefCountedLockRegistry``), independent of the external reply-delivery
    and session-event locks.
  * An in-flight marker (``_in_flight``) — NOT ``SessionState.phase`` — guards
    against a rapid double-send landing between our injection and the next
    Hook ``Stop``/``TurnStarted`` event updating phase.
  * Draining reuses the same per-session lock, so drain and immediate send
    never race within one session.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.adapters.process.ghostty_terminal_adapter import (
    GhosttyTerminal,
    GhosttyTerminalAdapter,
    InjectionOutcome,
)
from app.domain.external_session_models import ExternalBinding, GhosttyInputTarget
from app.domain.models import utc_now
from app.domain.session_models import SessionPhase
from app.domain.user_question_models import (
    ExternalGhosttyQuestionTarget,
    ExternalQuestionActionResult,
    ExternalQuestionActionStatus,
    ExternalUserQuestionContext,
)
from app.infra.lock_registry import RefCountedLockRegistry
from app.infra.source_text_normalization import normalize_line_endings
from app.services.external_binding_store import ExternalBindingStore
from app.services.external_input_mode_state import ExternalInputTargetStore
from app.services.external_input_queue import (
    ExternalInputQueue,
    QueuedInput,
    QueueEnqueueOverflow,
)
from app.services.external_user_question_state import (
    ExternalUserQuestionState,
    PendingExternalUserQuestionSnapshot,
)
from app.services.local_process_probe import LocalProcessProbe, ProcessTargetValidation
from app.services.pairing_callback_registry import (
    PairConsumeOk,
    PairConsumeResult,
    PairConsumeUnauthorized,
    PairingCallbackRegistry,
)
from app.services.session_store import SessionStore
from app.services.user_question_callback_registry import UserQuestionCallbackRegistry

logger = logging.getLogger(__name__)


class SendOutcome(StrEnum):
    """User-visible result of a send_text attempt."""

    SENT = "sent"
    QUEUED = "queued"
    NOT_PAIRED = "not_paired"
    NO_TARGET = "no_target"  # user has not selected an input session
    NOT_OWNER = "not_owner"
    BINDING_STALE = "binding_stale"  # selected binding_id no longer matches live
    SESSION_ENDED = "session_ended"
    PROCESS_INVALID = "process_invalid"
    TERMINAL_INVALID = "terminal_invalid"
    ADAPTER_UNAVAILABLE = "adapter_unavailable"
    QUEUE_FULL = "queue_full"
    INJECTION_FAILED = "injection_failed"
    INJECTION_INDETERMINATE = "injection_indeterminate"


class PairOutcome(StrEnum):
    """User-visible result of pairing/activating."""

    NEEDS_PAIRING = "needs_pairing"  # no target yet; handler should show candidates
    PAIRED = "paired"
    ACTIVATED = "activated"  # had a valid target; set as current input session
    NOT_OWNER = "not_owner"
    BINDING_STALE = "binding_stale"
    SESSION_ENDED = "session_ended"
    PROCESS_INVALID = "process_invalid"
    TERMINAL_INVALID = "terminal_invalid"
    ADAPTER_UNAVAILABLE = "adapter_unavailable"
    NO_TERMINALS = "no_terminals"
    TOKEN_INVALID = "token_invalid"
    TOKEN_UNAUTHORIZED = "token_unauthorized"
    PAIRING_NOT_ENABLED = "pairing_not_enabled"


class _DrainStep(StrEnum):
    INJECTED = "injected"
    WAIT = "wait"
    WAIT_SETTLED = "wait_settled"
    EMPTY = "empty"
    ABORT = "abort"


# Phases in which direct injection is allowed (design §7). Any other phase
# queues instead. AskUserQuestion and pending permission gate via the state
# fields, not the phase enum.
_SENDABLE_PHASES = frozenset({SessionPhase.IDLE, SessionPhase.WAITING_FOR_INPUT})

# Injection failures that the adapter reports BEFORE anything was pasted into
# the terminal. Single source of truth shared by both injection paths:
# ``send_text`` maps them to ``ADAPTER_UNAVAILABLE``; the drain treats the
# dequeued entry as safe to restore and retry once Ghostty becomes reachable.
_RETRYABLE_PREFLIGHT_OUTCOMES = frozenset(
    {
        InjectionOutcome.GHOSTTY_NOT_RUNNING,
        InjectionOutcome.APPLESCRIPT_DISABLED,
        InjectionOutcome.NON_DARWIN,
        InjectionOutcome.OSASCRIPT_MISSING,
        InjectionOutcome.TCC_DENIED,
    }
)

# Best-effort user notification for queue drops (design §9: "丢弃并通知").
# Wired to ``ExternalSessionPushNotifier.notify_info`` at the composition root.
_Notice = tuple[int, str]


class _NotifyUserFn(Protocol):
    async def __call__(self, *, user_id: int, text: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class PairingCandidates:
    """Read-only candidate list for the pairing UI."""

    binding_id: str
    paired_tty: str  # the trust anchor resolved from the Claude PID
    terminals: list[GhosttyTerminal]


@dataclass(frozen=True, slots=True)
class _DrainSlot:
    """Per-session long-lived drain task handle."""

    session_id: str
    task: asyncio.Task[None]
    wake: asyncio.Event


class ExternalSessionInputService:
    """Pairing + activation + send/drain for external Ghostty input.

    Construct with a feature flag ``enabled`` (mirrors
    ``Settings.GHOSTTY_INPUT_ENABLED``); when False, every method short-circuits
    to ``PairingNotEnabled`` / ``SendOutcome.ADAPTER_UNAVAILABLE`` so the rest of
    the binding/reply system keeps working.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        binding_store: ExternalBindingStore,
        session_store: SessionStore,
        ghostty_adapter: GhosttyTerminalAdapter,
        process_probe: LocalProcessProbe,
        pairing_registry: PairingCallbackRegistry,
        input_mode_store: ExternalInputTargetStore,
        input_queue: ExternalInputQueue,
        input_locks: RefCountedLockRegistry,
        external_user_question_state: ExternalUserQuestionState | None = None,
        user_question_callback_registry: UserQuestionCallbackRegistry | None = None,
        drain_publish_wait_timeout_sec: float = 30.0,
        in_flight_ttl_sec: float = 2.0,
        notify_user: _NotifyUserFn | None = None,
    ) -> None:
        self._enabled = enabled
        self._binding_store = binding_store
        self._session_store = session_store
        self._adapter = ghostty_adapter
        self._probe = process_probe
        self._notify_user = notify_user
        self._pairing = pairing_registry
        self._mode_store = input_mode_store
        self._queue = input_queue
        self._input_locks = input_locks
        self._external_uq_state = external_user_question_state
        self._uq_callback_registry = user_question_callback_registry
        self._drain_wait_timeout = drain_publish_wait_timeout_sec
        self._in_flight_ttl_sec = in_flight_ttl_sec
        self._injecting: set[str] = set()
        self._in_flight: set[str] = set()
        self._in_flight_generation: dict[str, int] = {}
        self._in_flight_timers: dict[str, asyncio.Task[None]] = {}
        self._drain_slots: dict[str, _DrainSlot] = {}
        self._lifecycle_lock = asyncio.Lock()
        self._shutting_down: bool = False
        self._pending_notices: set[asyncio.Task[None]] = set()

    # ─── public: pairing ────────────────────────────────────────────

    async def pair_candidates(
        self,
        *,
        user_id: int,
        session_id: str,
    ) -> tuple[PairOutcome, PairingCandidates | None]:
        """Resolve the pairing trust anchor and enumerate Ghostty terminals.

        Returns ``(PairingCandidates)`` for display, or a refusal outcome.
        Caller (handler) then registers a token per chosen terminal and sends
        inline buttons.
        """
        if not self._enabled:
            return PairOutcome.PAIRING_NOT_ENABLED, None
        binding = self._binding_store.get_binding(session_id)
        if binding is None or binding.user_id != user_id:
            return PairOutcome.NOT_OWNER, None
        if binding.ended_at is not None:
            return PairOutcome.SESSION_ENDED, None
        if not self._adapter.is_available():
            return PairOutcome.ADAPTER_UNAVAILABLE, None

        paired_tty = self._resolve_paired_tty(binding)
        if paired_tty is None:
            return PairOutcome.PROCESS_INVALID, None
        process = await self._validate_foreground(
            pid=binding.pid or 0,
            paired_tty=paired_tty,
        )
        if not process.ok:
            return PairOutcome.PROCESS_INVALID, None

        terminals, _err = await self._adapter.list_terminals()
        if terminals is None:
            # Ghostty down / TCC / disabled / not on darwin: pairing is
            # currently impossible, but the external binding remains usable.
            return PairOutcome.ADAPTER_UNAVAILABLE, None
        if not terminals:
            return PairOutcome.NO_TERMINALS, None

        # Ghostty 1.3 does not expose terminal PID/TTY, so cwd/title are display
        # hints only and MUST NOT auto-select a target. Put exact-cwd candidates
        # first while preserving adapter order within each group; the handler
        # shows enough identity for the owner to choose explicitly.
        terminals = sorted(terminals, key=lambda terminal: terminal.cwd != binding.cwd)
        return (
            PairOutcome.NEEDS_PAIRING,
            PairingCandidates(
                binding_id=binding.binding_id,
                paired_tty=paired_tty,
                terminals=terminals,
            ),
        )

    async def register_pair_token(
        self,
        *,
        user_id: int,
        session_id: str,
        expected_binding_id: str,
        terminal_id: str,
    ) -> str | None:
        """Issue a short-lived token for one displayed terminal candidate.

        ``expected_binding_id`` comes from ``PairingCandidates``. If the
        binding was removed/recreated between listing and token registration,
        registration fails instead of anchoring an old UI to the new binding.
        """
        if not self._enabled:
            return None
        binding = self._binding_store.get_binding(session_id)
        if binding is None or binding.user_id != user_id or binding.ended_at is not None or binding.binding_id != expected_binding_id:
            return None
        return await self._pairing.register_token(
            user_id=user_id,
            session_id=session_id,
            binding_id=binding.binding_id,
            terminal_id=terminal_id,
        )

    async def consume_pair_token(
        self,
        *,
        token: str,
        user_id: int,
    ) -> PairOutcome:
        """Consume a callback token and finalise pairing.

        Order: token resolve → owner match (registry-enforced) → binding still
        owner & not ended & same generation → terminal UUID still unique →
        process foreground check → persist target → set input mode. Every
        failure is fail-closed.
        """
        if not self._enabled:
            return PairOutcome.PAIRING_NOT_ENABLED
        if not self._adapter.is_available():
            return PairOutcome.ADAPTER_UNAVAILABLE
        result: PairConsumeResult = await self._pairing.consume(token, user_id)
        if not isinstance(result, PairConsumeOk):
            if isinstance(result, PairConsumeUnauthorized):
                return PairOutcome.TOKEN_UNAUTHORIZED
            return PairOutcome.TOKEN_INVALID
        snap = result.snapshot
        async with self._input_locks.lock(snap.session_id):
            binding = self._binding_store.get_binding(snap.session_id)
            if binding is None or binding.user_id != user_id:
                return PairOutcome.NOT_OWNER
            if binding.ended_at is not None:
                return PairOutcome.SESSION_ENDED
            if binding.binding_id != snap.binding_id:
                # ABA: token was issued under a generation that no longer matches.
                return PairOutcome.BINDING_STALE

            ok, terminal, terminal_error = await self._adapter.validate_terminal(snap.terminal_id)
            if not ok or terminal is None:
                return (
                    PairOutcome.TERMINAL_INVALID
                    if terminal_error in {InjectionOutcome.NOT_FOUND, InjectionOutcome.NOT_UNIQUE}
                    else PairOutcome.ADAPTER_UNAVAILABLE
                )

            # validate_terminal awaited AppleScript; repeat binding/process checks
            # while holding the same lock used by question actions.
            binding = self._binding_store.get_binding(snap.session_id)
            if binding is None or binding.user_id != user_id or binding.binding_id != snap.binding_id:
                return PairOutcome.BINDING_STALE
            if binding.ended_at is not None:
                return PairOutcome.SESSION_ENDED
            paired_tty = self._resolve_paired_tty(binding)
            if paired_tty is None:
                return PairOutcome.PROCESS_INVALID
            process = await self._validate_foreground(
                pid=binding.pid or 0,
                paired_tty=paired_tty,
            )
            if not process.ok:
                return PairOutcome.PROCESS_INVALID

            old_target = binding.ghostty_target
            if old_target is not None and self._external_uq_state is not None:
                self._external_uq_state.invalidate_ghostty_target(
                    session_id=snap.session_id,
                    binding_id=old_target.binding_id,
                    terminal_id=old_target.terminal_id,
                    paired_tty=old_target.paired_tty,
                    paired_at=old_target.paired_at,
                )
            if old_target is not None and self._uq_callback_registry is not None:
                await self._uq_callback_registry.invalidate_session(snap.session_id)
            saved = self._binding_store.set_ghostty_target(
                snap.session_id,
                binding.binding_id,
                terminal_id=snap.terminal_id,
                paired_tty=paired_tty,
                paired_at=utc_now(),
                name=terminal.name,
                cwd=terminal.cwd,
            )
            if not saved:
                return PairOutcome.BINDING_STALE
            binding_id = binding.binding_id

        activated = await self._activate_target(
            user_id=user_id,
            session_id=snap.session_id,
            binding_id=binding_id,
        )
        return PairOutcome.PAIRED if activated else PairOutcome.BINDING_STALE

    # ─── public: activation ──────────────────────────────────────────

    async def activate_select(
        self,
        *,
        user_id: int,
        session_id: str,
    ) -> PairOutcome:
        """Activated from ``/list`` for an already-bound session.

        If a valid persisted Ghostty target exists, sets it as the user's
        current input session and returns ``ACTIVATED``. If no target or the
        target/process no longer validating, returns ``NEEDS_PAIRING`` so the
        handler kicks off pairing. Returns ``NOT_OWNER`` for foreign bindings
        and ``SESSION_ENDED`` for ended ones.
        """
        if not self._enabled:
            return PairOutcome.PAIRING_NOT_ENABLED
        binding = self._binding_store.get_binding(session_id)
        if binding is None or binding.user_id != user_id:
            return PairOutcome.NOT_OWNER
        if binding.ended_at is not None:
            return PairOutcome.SESSION_ENDED
        target = binding.ghostty_target
        if target is None:
            return PairOutcome.NEEDS_PAIRING
        if target.binding_id != binding.binding_id:
            self._binding_store.clear_ghostty_target(session_id, binding.binding_id)
            return PairOutcome.BINDING_STALE
        if not self._adapter.is_available():
            return PairOutcome.ADAPTER_UNAVAILABLE

        ok, _terminal, terminal_error = await self._adapter.validate_terminal(target.terminal_id)
        if not ok:
            if terminal_error in {InjectionOutcome.NOT_FOUND, InjectionOutcome.NOT_UNIQUE}:
                self._binding_store.clear_ghostty_target(session_id, binding.binding_id)
                return PairOutcome.NEEDS_PAIRING
            return PairOutcome.ADAPTER_UNAVAILABLE

        process = await self._validate_foreground(
            pid=binding.pid or 0,
            paired_tty=target.paired_tty,
        )
        if not process.ok:
            # A process/TTY mismatch can be transient (e.g. a foreground tool),
            # so retain the persisted pairing but do not enter input mode.
            return PairOutcome.PROCESS_INVALID
        activated = await self._activate_target(
            user_id=user_id,
            session_id=session_id,
            binding_id=binding.binding_id,
        )
        return PairOutcome.ACTIVATED if activated else PairOutcome.BINDING_STALE

    async def has_target(self, user_id: int) -> bool:
        """Whether this user currently has an active external input target (no side effects)."""
        return await self._mode_store.get_target(user_id) is not None

    async def leave(self, *, user_id: int) -> bool:
        """Exit external input mode and discard queued input for that target."""
        target = await self._mode_store.get_target(user_id)
        if target is None:
            return False
        async with self._input_locks.lock(target.session_id):
            current = await self._mode_store.get_target(user_id)
            if current is None or current.session_id != target.session_id:
                return False
            await self._mode_store.clear_target(user_id)
            dropped = await self._queue.clear(target.session_id)
            self._injecting.discard(target.session_id)
            self._clear_in_flight(target.session_id)
            await self._stop_drain(target.session_id)
        if dropped:
            await self._deliver_notices(
                target.session_id,
                [(user_id, f"已退出外部输入模式，丢弃 {len(dropped)} 条未发送的排队消息。")],
            )
        return True

    # ─── public: AskUserQuestion transport ───────────────────────────

    async def select_option(
        self,
        *,
        context: ExternalUserQuestionContext,
        question_index: int,
        option_count: int,
        option_index: int,
        submit_after: bool,
    ) -> ExternalQuestionActionResult:
        return await self._apply_user_question_action(
            context=context,
            question_index=question_index,
            option_count=option_count,
            action="select",
            option_index=option_index,
            text="",
            final_action=submit_after,
        )

    async def answer_with_text(
        self,
        *,
        context: ExternalUserQuestionContext,
        question_index: int,
        option_count: int,
        text: str,
        submit_after: bool,
    ) -> ExternalQuestionActionResult:
        return await self._apply_user_question_action(
            context=context,
            question_index=question_index,
            option_count=option_count,
            action="answer_text",
            option_index=-1,
            text=text,
            final_action=submit_after,
        )

    async def advance_after_multi_select(
        self,
        *,
        context: ExternalUserQuestionContext,
        question_index: int,
        option_count: int,
        final_question: bool,
    ) -> ExternalQuestionActionResult:
        return await self._apply_user_question_action(
            context=context,
            question_index=question_index,
            option_count=option_count,
            action="advance_multi",
            option_index=-1,
            text="",
            final_action=final_question,
        )

    async def question_completed(self, *, context: ExternalUserQuestionContext) -> None:
        state = self._external_uq_state
        if state is None:
            return
        async with self._input_locks.lock(context.session_id):
            pending = state.get(context.tool_use_id)
            if not self._pending_matches_context(pending, context):
                return
            if not state.mark_completed(
                tool_use_id=context.tool_use_id,
                expected_target=context.target,
            ):
                return
        await self._schedule_drain(context.session_id)

    async def question_indeterminate(self, *, context: ExternalUserQuestionContext, reason: str) -> None:
        state = self._external_uq_state
        if state is None:
            return
        async with self._input_locks.lock(context.session_id):
            state.mark_indeterminate(
                tool_use_id=context.tool_use_id,
                expected_target=context.target,
                reason=reason,
            )

    # ─── public: send / drain ────────────────────────────────────────

    async def send_text(
        self,
        *,
        user_id: int,
        text: str,
    ) -> SendOutcome:
        """Send ordinary Telegram text to the user's active external session.

        CRLF/CR normalised to LF (design §8). On a non-sendable phase, enqueues
        (§9); on a target/process failure, refuses and does not enqueue.
        ``INDETERMINATE`` is never retried and is not enqueued.
        """
        if not self._enabled:
            return SendOutcome.ADAPTER_UNAVAILABLE
        payload = _normalise_text(text)
        if payload == "":
            # Empty after normalisation: nothing meaningful to inject.
            return SendOutcome.SENT

        target = await self._mode_store.get_target(user_id)
        if target is None:
            return SendOutcome.NO_TARGET
        session_id = target.session_id

        async with self._input_locks.lock(session_id):
            binding = self._binding_store.get_binding(session_id)
            if binding is None or binding.user_id != user_id:
                await self._mode_store.clear_target(user_id)
                return SendOutcome.NOT_OWNER
            state = self._session_store.get(session_id)
            if binding.ended_at is not None or (state is not None and state.phase is SessionPhase.ENDED):
                await self._mode_store.clear_target_for_session(session_id)
                await self._notify_queue_cleared(session_id, user_id=user_id, reason="会话已结束")
                return SendOutcome.SESSION_ENDED
            if binding.binding_id != target.binding_id:
                # Stale selection from before a rebind.
                await self._mode_store.clear_target(user_id)
                return SendOutcome.BINDING_STALE
            ghostty_target = binding.ghostty_target
            if ghostty_target is None:
                await self._mode_store.clear_target(user_id)
                return SendOutcome.NOT_PAIRED
            if ghostty_target.binding_id != binding.binding_id:
                self._binding_store.clear_ghostty_target(session_id, binding.binding_id)
                await self._mode_store.clear_target(user_id)
                return SendOutcome.BINDING_STALE
            if not self._adapter.is_available():
                return SendOutcome.ADAPTER_UNAVAILABLE

            process = await self._validate_foreground(
                pid=binding.pid or 0,
                paired_tty=ghostty_target.paired_tty,
            )
            if not process.ok:
                return SendOutcome.PROCESS_INVALID

            ok, _terminal, terminal_error = await self._adapter.validate_terminal(ghostty_target.terminal_id)
            if not ok:
                if terminal_error in {InjectionOutcome.NOT_FOUND, InjectionOutcome.NOT_UNIQUE}:
                    self._binding_store.clear_ghostty_target(session_id, binding.binding_id)
                    await self._mode_store.clear_target_for_session(session_id)
                    await self._notify_queue_cleared(session_id, user_id=user_id, reason="配对的终端已失效")
                    return SendOutcome.TERMINAL_INVALID
                return SendOutcome.ADAPTER_UNAVAILABLE

            current = self._current_binding_target(
                session_id=session_id,
                user_id=user_id,
                expected_binding_id=binding.binding_id,
                expected_target=ghostty_target,
            )
            if current is None:
                return SendOutcome.BINDING_STALE
            binding, ghostty_target = current

            if session_id in self._in_flight:
                # A previous send is mid-flight (between our inject and the
                # next Hook phase update); do not pile on — enqueue instead.
                return await self._enqueue(session_id, binding.binding_id, payload)

            sendable = self._is_sendable(session_id)
            if not sendable:
                return await self._enqueue(session_id, binding.binding_id, payload)
            queue_pending = await self._queue.has_pending(
                session_id,
                binding_id=binding.binding_id,
            )
            current = self._current_binding_target(
                session_id=session_id,
                user_id=user_id,
                expected_binding_id=binding.binding_id,
                expected_target=ghostty_target,
            )
            if current is None:
                return SendOutcome.BINDING_STALE
            binding, ghostty_target = current
            if queue_pending:
                # Preserve FIFO even if a new message wins the input lock just
                # as the session becomes ready before the existing drain does.
                return await self._enqueue(session_id, binding.binding_id, payload)

            # Terminal validation above can block on AppleScript for seconds.
            # Re-check the process trust anchor immediately before injection so
            # a shell that took over during that await never receives text+Enter.
            process = await self._validate_foreground(
                pid=binding.pid or 0,
                paired_tty=ghostty_target.paired_tty,
            )
            if not process.ok:
                return SendOutcome.PROCESS_INVALID

            self._injecting.add(session_id)
            try:
                outcome = await self._adapter.inject_text(ghostty_target.terminal_id, payload)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("ghostty inject raised", extra={"session_id": session_id})
                return SendOutcome.INJECTION_FAILED
            else:
                if outcome == InjectionOutcome.OK:
                    # Stay in _injecting until the guard timer is armed. Arming
                    # waits on the lifecycle lock; a Stop scheduled during
                    # AppleScript would otherwise clear the new guard and the
                    # mismatched generation would then skip the timer.
                    await self._mark_in_flight(session_id)
            finally:
                self._injecting.discard(session_id)
            if outcome == InjectionOutcome.OK:
                return SendOutcome.SENT
            if outcome == InjectionOutcome.INDETERMINATE:
                return SendOutcome.INJECTION_INDETERMINATE
            if outcome in _RETRYABLE_PREFLIGHT_OUTCOMES:
                return SendOutcome.ADAPTER_UNAVAILABLE
            return SendOutcome.INJECTION_FAILED

    async def notify_hook_event(
        self,
        *,
        session_id: str,
        event_kind: str,
    ) -> None:
        """Called by the Hook pipeline on phase-affecting events.

        ``event_kind`` is the Claude hook event name (``Stop``, ``TurnStarted``,
        ``SessionEnd``, etc.) — we do not depend on exact strings beyond the
        ones that should clear in-flight / schedule a drain.
        """
        if not self._enabled:
            return
        kind = event_kind.strip().lower().replace("-", "_")
        if kind in {
            "stop",
            "turn_completed",
            "permission_approved",
            "permission_denied",
            "permission_resolved",
            "permissionresolved",
            "user_question_resolved",
            "userquestionresolved",
        }:
            if session_id in self._injecting:
                # A completion from an older turn can race with the AppleScript
                # call for a new turn. It must not release the new turn's guard.
                return
            self._clear_in_flight(session_id)
            await self._schedule_drain(session_id)
        elif kind in {"session_end", "sessionend", "sessionended", "clear"}:
            await self.invalidate_binding(session_id, reason="session_end")

    async def invalidate_binding(self, session_id: str, *, reason: str) -> None:
        """Tear down all input state for a session (unbind / SessionEnd / reaper / target failure)."""
        if not self._enabled:
            return
        async with self._input_locks.lock(session_id):
            # Resolve the §9 notice recipient under the input lock so the owner
            # matches the binding generation whose queue is cleared below.
            binding = self._binding_store.get_binding(session_id)
            cleared_targets = await self._mode_store.clear_target_for_session(session_id)
            dropped = await self._queue.clear(session_id)
            await self._pairing.invalidate_session(session_id)
            self._injecting.discard(session_id)
            self._clear_in_flight(session_id)
            await self._stop_drain(session_id)
            if self._external_uq_state is not None:
                self._external_uq_state.invalidate_session(session_id)
            if self._uq_callback_registry is not None:
                await self._uq_callback_registry.invalidate_session(session_id)
        if cleared_targets or dropped:
            logger.info(
                "external input invalidated",
                extra={
                    "session_id": session_id,
                    "reason": reason,
                    "targets_cleared": len(cleared_targets),
                    "queued_dropped": len(dropped),
                },
            )
        # Design §9: report how many queued entries were discarded to every
        # affected owner — the binding owner plus each user whose input target
        # pointed here (``clear_target_for_session`` supports several).
        owners: set[int] = set()
        if binding is not None:
            owners.add(binding.user_id)
        owners.update(cleared.user_id for cleared in cleared_targets)
        if dropped and owners:
            text = f"⚠️ 外部输入会话已失效（{reason}），丢弃 {len(dropped)} 条未发送的排队消息。"
            # One delivery per owner; ``_deliver_notices`` requires a single
            # owner per batch.
            for owner in owners:
                self._spawn_notice_delivery(session_id, [(owner, text)])

    async def rebind_aba(self, session_id: str, new_binding_id: str) -> None:
        """Called after an unbind+rebind produced a new generation."""
        if not self._enabled:
            return
        async with self._input_locks.lock(session_id):
            await self._mode_store.invalidate_for_binding_aba(session_id, new_binding_id)
            await self._pairing.invalidate_binding(session_id, new_binding_id)
            await self._queue.clear(session_id)
            self._injecting.discard(session_id)
            self._clear_in_flight(session_id)
            await self._stop_drain(session_id)
            if self._external_uq_state is not None:
                self._external_uq_state.invalidate_stale_bindings(session_id, new_binding_id)
            if self._uq_callback_registry is not None:
                await self._uq_callback_registry.invalidate_session(session_id)

    async def shutdown(self) -> None:
        """Cancel all drain tasks and pending notice deliveries. Call on container shutdown."""
        async with self._lifecycle_lock:
            self._shutting_down = True
            slots = list(self._drain_slots.values())
            self._drain_slots.clear()
            pending = list(self._pending_notices)
            self._pending_notices.clear()
        for slot in slots:
            slot.task.cancel()
        for task in pending:
            task.cancel()
        await asyncio.gather(*(slot.task for slot in slots), *pending, return_exceptions=True)

    # ─── internals ──────────────────────────────────────────────────

    async def _apply_user_question_action(
        self,
        *,
        context: ExternalUserQuestionContext,
        question_index: int,
        option_count: int,
        action: str,
        option_index: int,
        text: str,
        final_action: bool,
    ) -> ExternalQuestionActionResult:
        state = self._external_uq_state
        if not self._enabled or state is None:
            return ExternalQuestionActionResult(
                ExternalQuestionActionStatus.REJECTED,
                "外部 Ghostty 问答功能未启用",
            )
        target = context.target
        if not isinstance(target, ExternalGhosttyQuestionTarget):
            return ExternalQuestionActionResult(
                ExternalQuestionActionStatus.REJECTED,
                "该问题目标不支持 Ghostty 注入",
            )

        async with self._input_locks.lock(context.session_id):
            pending = state.get_active(context.tool_use_id)
            if not self._pending_matches_context(pending, context):
                return ExternalQuestionActionResult(
                    ExternalQuestionActionStatus.REJECTED,
                    "问题已过期或目标已变化",
                )
            assert pending is not None
            prompt = next(
                (item for item in pending.prompts if item.question_index == question_index),
                None,
            )
            if prompt is None or len(prompt.options) != option_count:
                return ExternalQuestionActionResult(
                    ExternalQuestionActionStatus.REJECTED,
                    "问题选项已变化",
                )
            if action == "select" and (option_index < 0 or option_index >= option_count):
                return ExternalQuestionActionResult(ExternalQuestionActionStatus.REJECTED, "无效的选项")
            if action == "advance_multi" and not prompt.multi_select:
                return ExternalQuestionActionResult(ExternalQuestionActionStatus.REJECTED, "这个问题不是多选题")
            if action == "answer_text" and not text:
                return ExternalQuestionActionResult(ExternalQuestionActionStatus.REJECTED, "回复内容不能为空")

            if not self._structured_question_is_active(context):
                return ExternalQuestionActionResult(
                    ExternalQuestionActionStatus.REJECTED,
                    "Claude 已不再等待这个问题",
                )
            binding = self._binding_for_question_context(context)
            if binding is None:
                return ExternalQuestionActionResult(
                    ExternalQuestionActionStatus.REJECTED,
                    "外部会话绑定或 Ghostty 配对已变化",
                )
            if not self._adapter.is_available():
                return ExternalQuestionActionResult(
                    ExternalQuestionActionStatus.REJECTED,
                    "Ghostty 适配器不可用",
                )
            process = await self._validate_foreground(
                pid=binding.pid or 0,
                paired_tty=target.paired_tty,
            )
            if not process.ok:
                return ExternalQuestionActionResult(
                    ExternalQuestionActionStatus.REJECTED,
                    "无法验证前台 Claude 进程",
                )

            terminal_ok, _terminal, terminal_error = await self._adapter.validate_terminal(target.terminal_id)
            if not terminal_ok:
                return ExternalQuestionActionResult(
                    ExternalQuestionActionStatus.REJECTED,
                    f"Ghostty 终端不可用: {terminal_error or 'unknown'}",
                )

            # validate_terminal is an AppleScript await. Re-read every mutable
            # security anchor and foreground status before sending any key.
            pending = state.get_active(context.tool_use_id)
            binding = self._binding_for_question_context(context)
            if not self._pending_matches_context(pending, context) or binding is None or not self._structured_question_is_active(context):
                return ExternalQuestionActionResult(
                    ExternalQuestionActionStatus.REJECTED,
                    "问题或目标在校验期间发生变化",
                )
            process = await self._validate_foreground(
                pid=binding.pid or 0,
                paired_tty=target.paired_tty,
            )
            if not process.ok:
                return ExternalQuestionActionResult(
                    ExternalQuestionActionStatus.REJECTED,
                    "前台 Claude 进程在发送前发生变化",
                )

            self._injecting.add(context.session_id)
            try:
                if action == "select":
                    outcome = await self._adapter.select_user_question_option(
                        target.terminal_id,
                        option_count=option_count,
                        option_index=option_index,
                        submit_after=final_action,
                    )
                elif action == "answer_text":
                    outcome = await self._adapter.answer_user_question_with_text(
                        target.terminal_id,
                        option_count=option_count,
                        text=text,
                        submit_after=final_action,
                    )
                elif action == "advance_multi":
                    outcome = await self._adapter.advance_user_question_after_multi_select(
                        target.terminal_id,
                        option_count=option_count,
                        final_question=final_action,
                    )
                else:  # pragma: no cover - only typed public methods call here
                    raise ValueError(f"unsupported external question action: {action}")
            except asyncio.CancelledError:
                state.mark_indeterminate(
                    tool_use_id=context.tool_use_id,
                    expected_target=context.target,
                    reason="Ghostty 问答操作被取消，终端状态不确定",
                )
                raise
            except Exception:
                logger.exception(
                    "ghostty user-question action raised",
                    extra={
                        "session_id": context.session_id,
                        "tool_use_id": context.tool_use_id,
                        "action": action,
                    },
                )
                state.mark_indeterminate(
                    tool_use_id=context.tool_use_id,
                    expected_target=context.target,
                    reason="Ghostty 问答操作异常，终端状态不确定",
                )
                return ExternalQuestionActionResult(
                    ExternalQuestionActionStatus.INDETERMINATE,
                    "Ghostty 操作结果不确定，请检查终端",
                )
            finally:
                self._injecting.discard(context.session_id)

            if outcome == InjectionOutcome.OK:
                if final_action and not state.mark_terminal_action_applied(
                    tool_use_id=context.tool_use_id,
                    expected_target=context.target,
                ):
                    state.mark_indeterminate(
                        tool_use_id=context.tool_use_id,
                        expected_target=context.target,
                        reason="终端操作已完成，但问题状态无法确认",
                    )
                    return ExternalQuestionActionResult(
                        ExternalQuestionActionStatus.INDETERMINATE,
                        "终端操作已完成，但问题状态无法确认",
                    )
                return ExternalQuestionActionResult(ExternalQuestionActionStatus.APPLIED)

            if outcome in {InjectionOutcome.INDETERMINATE, InjectionOutcome.TIMEOUT}:
                state.mark_indeterminate(
                    tool_use_id=context.tool_use_id,
                    expected_target=context.target,
                    reason=f"Ghostty 问答操作结果不确定: {outcome}",
                )
                return ExternalQuestionActionResult(
                    ExternalQuestionActionStatus.INDETERMINATE,
                    "Ghostty 操作结果不确定，请检查终端",
                )
            return ExternalQuestionActionResult(
                ExternalQuestionActionStatus.REJECTED,
                f"Ghostty 问答操作失败: {outcome}",
            )

    @staticmethod
    def _pending_matches_context(
        pending: PendingExternalUserQuestionSnapshot | None,
        context: ExternalUserQuestionContext,
    ) -> bool:
        return (
            pending is not None
            and pending.tool_use_id == context.tool_use_id
            and pending.session_id == context.session_id
            and pending.user_id == context.user_id
            and pending.target == context.target
        )

    def _structured_question_is_active(self, context: ExternalUserQuestionContext) -> bool:
        state = self._session_store.find_by_active_user_question_tool_use_id(context.tool_use_id)
        return state is not None and state.session_id == context.session_id

    def _binding_for_question_context(
        self,
        context: ExternalUserQuestionContext,
    ) -> ExternalBinding | None:
        target = context.target
        if not isinstance(target, ExternalGhosttyQuestionTarget):
            # Only Ghostty-bound sessions carry an injectable question target.
            return None
        binding = self._binding_store.get_binding(context.session_id)
        if binding is None or binding.user_id != context.user_id or binding.ended_at is not None or binding.binding_id != target.binding_id:
            return None
        binding_target = binding.ghostty_target
        if (
            binding_target is None
            or binding_target.binding_id != target.binding_id
            or binding_target.terminal_id != target.terminal_id
            or binding_target.paired_tty != target.paired_tty
            or binding_target.paired_at != target.paired_at
        ):
            return None
        return binding

    def _current_binding_target(
        self,
        *,
        session_id: str,
        user_id: int,
        expected_binding_id: str,
        expected_target: GhosttyInputTarget,
    ) -> tuple[ExternalBinding, GhosttyInputTarget] | None:
        """Re-read binding/target after an await and enforce the ABA snapshot."""
        binding = self._binding_store.get_binding(session_id)
        if binding is None or binding.user_id != user_id or binding.ended_at is not None or binding.binding_id != expected_binding_id:
            return None
        target = binding.ghostty_target
        if target is None or target.binding_id != binding.binding_id or target != expected_target:
            return None
        return binding, target

    def _is_sendable(self, session_id: str) -> bool:
        state = self._session_store.get(session_id)
        if state is None:
            # The design only permits known idle/waiting_for_input states.
            # Unknown state is no positive proof, so fail closed and queue.
            return False
        if state.phase not in _SENDABLE_PHASES:
            return False
        if state.pending_permission is not None:
            return False
        if state.structured_user_question_key:
            return False
        return True

    async def _validate_foreground(self, *, pid: int, paired_tty: str) -> ProcessTargetValidation:
        """Offload ``validate_claude_foreground`` to a worker thread (code-review #12).

        The probe spawns up to four synchronous ``ps`` subprocesses per call; if
        run on the event-loop thread it blocks every Hook/Telegram/drain coroutine
        for the fork+exec+wait duration. Running it via ``asyncio.to_thread``
        keeps the loop responsive while the system subprocess executes. The
        probe interface itself stays synchronous and loop-agnostic.
        """
        return await asyncio.to_thread(
            self._probe.validate_claude_foreground,
            pid=pid,
            paired_tty=paired_tty,
        )

    async def _enqueue(self, session_id: str, binding_id: str, payload: str) -> SendOutcome:
        result = await self._queue.enqueue(session_id, text=payload, binding_id=binding_id)
        if isinstance(result, QueueEnqueueOverflow):
            return SendOutcome.QUEUE_FULL
        await self._ensure_drain(session_id)
        return SendOutcome.QUEUED

    async def _activate_target(
        self,
        *,
        user_id: int,
        session_id: str,
        binding_id: str,
    ) -> bool:
        """Switch input intent if the binding generation is still current."""
        binding = self._binding_store.get_binding(session_id)
        if binding is None or binding.user_id != user_id or binding.ended_at is not None or binding.binding_id != binding_id:
            return False

        previous = await self._mode_store.get_target(user_id)
        if previous is not None and previous.session_id != session_id:
            async with self._input_locks.lock(previous.session_id):
                current = await self._mode_store.get_target(user_id)
                if current is not None and current.session_id == previous.session_id:
                    await self._notify_queue_cleared(previous.session_id, user_id=user_id, reason="已切换到其他输入会话")
                    self._injecting.discard(previous.session_id)
                    self._clear_in_flight(previous.session_id)
                    await self._stop_drain(previous.session_id)
        async with self._input_locks.lock(session_id):
            binding = self._binding_store.get_binding(session_id)
            if binding is None or binding.user_id != user_id or binding.ended_at is not None or binding.binding_id != binding_id:
                return False
            await self._mode_store.set_target(
                user_id=user_id,
                session_id=session_id,
                binding_id=binding_id,
            )
        return True

    def _resolve_paired_tty(self, binding: ExternalBinding) -> str | None:
        """Resolve the trust-anchor TTY for a binding.

        Prefer the binding's already-recorded tty (carried by hooks); fall
        back to querying the controlling tty of the live pid. Never guesses.
        """
        if binding.tty:
            return binding.tty
        if not binding.pid or binding.pid <= 0:
            return None
        return self._probe.pid_controlling_tty(binding.pid)

    # --- drain task ----------------------------------------------------------

    async def _mark_in_flight(self, session_id: str) -> None:
        """Arm the post-inject guard and a single generation-scoped TTL timer.

        A newer inject or an explicit clear bumps the generation so a timer
        started for the previous guard cannot drop the current one. Expiry is
        the only path that removes the id without a hook, and it always wakes
        the drain.
        """
        generation = self._in_flight_generation.get(session_id, 0) + 1
        self._in_flight_generation[session_id] = generation
        self._in_flight.add(session_id)
        try:
            await self._schedule_in_flight_timeout(session_id, generation)
        except asyncio.CancelledError:
            # Text is already in the terminal. If arming was cancelled while
            # waiting for the lifecycle lock, finish it; otherwise the guard
            # has no timer and Stop was ignored for the whole inject.
            if session_id not in self._in_flight_timers and self._in_flight_generation.get(session_id) == generation:
                await self._schedule_in_flight_timeout(session_id, generation)
            raise

    def _clear_in_flight(self, session_id: str) -> None:
        self._in_flight.discard(session_id)
        self._in_flight_generation[session_id] = self._in_flight_generation.get(session_id, 0) + 1
        self._cancel_in_flight_timer(session_id)

    def _cancel_in_flight_timer(self, session_id: str) -> None:
        task = self._in_flight_timers.pop(session_id, None)
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def _schedule_in_flight_timeout(self, session_id: str, generation: int) -> None:
        """Schedule an automatic release after the in-flight guard TTL expires."""
        ttl = self._in_flight_ttl_sec

        async def _release_after_ttl() -> None:
            try:
                await asyncio.sleep(ttl)
            except asyncio.CancelledError:
                raise
            await self._release_in_flight_if_current(session_id, generation)

        async with self._lifecycle_lock:
            if self._shutting_down or self._in_flight_generation.get(session_id) != generation:
                return
            self._cancel_in_flight_timer(session_id)
            task = asyncio.create_task(_release_after_ttl())
            self._in_flight_timers[session_id] = task
            self._pending_notices.add(task)
            task.add_done_callback(self._pending_notices.discard)

    async def _release_in_flight_if_current(self, session_id: str, generation: int) -> None:
        if self._shutting_down or self._in_flight_generation.get(session_id) != generation:
            return
        self._in_flight.discard(session_id)
        current = self._in_flight_timers.get(session_id)
        if current is asyncio.current_task():
            self._in_flight_timers.pop(session_id, None)
        await self._schedule_drain(session_id)

    async def _schedule_drain(self, session_id: str) -> None:
        """Wake an existing drain loop so it re-checks phase + dequeues."""
        slot = self._drain_slots.get(session_id)
        if slot is not None and not slot.task.done():
            slot.wake.set()
            return
        # No slot yet but a queue may have entries from before the slot spun up.
        if await self._queue.peek_size(session_id) > 0:
            await self._ensure_drain(session_id)

    async def _ensure_drain(self, session_id: str) -> None:
        async with self._lifecycle_lock:
            if self._shutting_down:
                # Container is shutting down: do not spawn a fresh drain loop that
                # ``shutdown``'s snapshot+gather would miss (race: shutdown clears
                # slots and releases the lock, an in-flight caller then enters here
                # and creates an orphan task surviving past teardown).
                return
            existing = self._drain_slots.get(session_id)
            if existing is not None and not existing.task.done():
                existing.wake.set()
                return
            wake = asyncio.Event()
            task = asyncio.create_task(self._drain_loop(session_id, wake))
            self._drain_slots[session_id] = _DrainSlot(session_id=session_id, task=task, wake=wake)

    async def _stop_drain(self, session_id: str) -> None:
        async with self._lifecycle_lock:
            slot = self._drain_slots.pop(session_id, None)
        if slot is not None and not slot.task.done():
            slot.wake.set()
            slot.task.cancel()
            try:
                await slot.task
            except (asyncio.CancelledError, Exception):
                pass

    async def _drain_loop(self, session_id: str, wake: asyncio.Event) -> None:
        """Drain one entry per ready turn, waiting on wake or state publish.

        A queued entry remains queued while phase/process/adapter is temporarily
        not ready. It is dequeued only after every precondition succeeds, so a
        phase flip never silently drops user input.
        """
        restart_if_queued = False
        aborted = False
        try:
            while True:
                step = await self._try_drain_one(session_id)
                if step is _DrainStep.INJECTED:
                    # _in_flight now blocks the next item until Stop/resolution.
                    continue
                if step is _DrainStep.ABORT:
                    # _try_drain_one cleared the old queue while still holding
                    # the input lock; messages arriving afterwards are new work.
                    aborted = True
                    return
                if step is _DrainStep.EMPTY:
                    restart_if_queued = True
                    return
                if step is _DrainStep.WAIT_SETTLED:
                    # The terminal input line may still hold unsubmitted pasted
                    # text; only a real session publish (Claude produced output
                    # ⇒ the line was submitted or cleared) or an explicit user
                    # re-send proves it is safe to inject the next head. The
                    # timeout-driven re-check below must NOT apply here.
                    await self._wait_for_drain_activity(session_id, wake, settle=True)
                    continue
                await self._wait_for_drain_activity(session_id, wake)
        finally:
            async with self._lifecycle_lock:
                current = self._drain_slots.get(session_id)
                if current is not None and current.task is asyncio.current_task():
                    self._drain_slots.pop(session_id, None)
            if (aborted or restart_if_queued) and await self._queue.peek_size(session_id) > 0:
                # Enqueue may have raced after the abort/empty clear while this
                # slot was still visible. The old slot is now removed, so start
                # a replacement instead of silently swallowing/stranding it.
                await self._ensure_drain(session_id)

    async def _wait_for_drain_activity(
        self,
        session_id: str,
        wake: asyncio.Event,
        *,
        settle: bool = False,
    ) -> None:
        """Wait for an explicit Hook wake or a SessionStore publish.

        ``settle`` disables the publish timeout: after an indeterminate drop
        the terminal input line may hold unsubmitted pasted text, and a
        timeout-driven wake-up could concatenate the next queued entry onto
        it. Only a real publish or wake may release that wait.
        """
        if wake.is_set():
            wake.clear()
            if not settle:
                return
            # settle: the pending wake predates the indeterminate drop (the
            # enqueue that raised it is already inside the paused queue), so
            # it is not proof the input line is clear — keep waiting for a
            # NEW wake (user re-send) or publish.
        since_cursor = self._session_store.get_publish_cursor(session_id)
        wake_task = asyncio.create_task(wake.wait())
        publish_task = asyncio.create_task(
            self._session_store.wait_for_publish(
                session_id,
                since_cursor=since_cursor,
                timeout_sec=None if settle else self._drain_wait_timeout,
            )
        )
        try:
            await asyncio.wait({wake_task, publish_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (wake_task, publish_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(wake_task, publish_task, return_exceptions=True)
            if wake.is_set():
                wake.clear()

    async def _abort_drain_locked(
        self,
        session_id: str,
        *,
        user_id: int | None,
        reason: str,
        extra_dropped: int = 0,
    ) -> tuple[_DrainStep, list[_Notice]]:
        """Clear old queued work while the caller still owns the input lock.

        Design §9 requires notifying the user how many unsent entries were
        discarded; the notice itself is delivered after the lock is released.
        ``extra_dropped`` counts entries the caller already removed from the
        queue (e.g. a just-dequeued head) so the reported total is exact.
        """
        dropped = await self._queue.clear(session_id)
        total = len(dropped) + extra_dropped
        notices: list[_Notice] = []
        if total and user_id is not None:
            notices.append((user_id, f"⚠️ 已丢弃 {total} 条未发送的排队消息（{reason}）。"))
        return _DrainStep.ABORT, notices

    async def _drop_current_and_continue(
        self,
        session_id: str,
        *,
        user_id: int | None,
        notices: list[_Notice],
        detail: str,
        terminal_state_unknown: bool = False,
    ) -> tuple[_DrainStep, list[_Notice]]:
        """Finish a drain iteration whose just-dequeued head cannot be retried.

        Only that entry is dropped — the remaining entries were never at risk.
        The drain then pauses until the session publishes a real event (or the
        user re-sends) instead of re-kicking itself: with
        ``terminal_state_unknown`` the adapter could not tell whether the text
        was pasted without Enter (INDETERMINATE/TIMEOUT/…), so injecting the
        next head immediately could concatenate it onto the unsubmitted line;
        otherwise the adapter/queue is in a state where an immediate retry of
        the next head would only repeat the failure — one notice and one
        subprocess attempt per entry, never a backoff-free drop loop.
        """
        # Count only live entries: reporting a soon-to-expire remainder as
        # "paused" would contradict the very next drop notice for it (§9).
        pruned = await self._queue.prune_expired(session_id)
        if pruned and user_id is not None:
            notices.append((user_id, f"⏳ 已丢弃 {pruned} 条等待超时的排队消息。"))
        remaining = await self._queue.peek_size(session_id)
        if user_id is not None:
            if terminal_state_unknown:
                tail = f"。其余 {remaining} 条已暂缓，待会话出现新事件或你再次发消息后继续" if remaining else ""
                notices.append(
                    (
                        user_id,
                        f"⚠️ 有 1 条排队消息发送结果不确定已丢弃（{detail}）{tail}。请到终端确认输入行没有残留文本。",
                    )
                )
            else:
                tail = f"。其余 {remaining} 条已暂缓，待会话出现新事件或你再次发消息后继续" if remaining else ""
                notices.append((user_id, f"⚠️ 有 1 条排队消息发送失败已丢弃（{detail}）{tail}。"))
        if remaining == 0:
            return _DrainStep.EMPTY, notices
        return _DrainStep.WAIT_SETTLED, notices

    async def _try_drain_one(self, session_id: str) -> _DrainStep:
        """Inject one ready queue head under the per-session input lock.

        Drop notices are collected while holding the input lock and delivered
        in a background task: Telegram I/O never extends the lock hold, and a
        ``_stop_drain`` cancellation racing this delivery cannot lose already
        counted drops (design §9).
        """
        async with self._input_locks.lock(session_id):
            step, notices = await self._try_drain_one_locked(session_id)
        if notices:
            self._spawn_notice_delivery(session_id, notices)
        return step

    async def _notify_queue_cleared(self, session_id: str, *, user_id: int, reason: str) -> None:
        """Clear the queue and report the discarded count (design §9).

        Used by the ``send_text`` dead-binding paths and ``_activate_target``'s
        old-session cleanup, which previously cleared silently. Called under
        the per-session input lock; the notice itself is delivered in the
        background (see ``_spawn_notice_delivery``).
        """
        dropped = await self._queue.clear(session_id)
        if dropped:
            self._spawn_notice_delivery(session_id, [(user_id, f"⚠️ {reason}，丢弃 {len(dropped)} 条未发送的排队消息。")])

    def _spawn_notice_delivery(self, session_id: str, notices: list[_Notice]) -> None:
        """Schedule best-effort notice delivery without blocking the caller.

        ``invalidate_binding`` callers may hold the external reply-delivery
        lock, and the drain loop itself may be cancelled mid-flight
        (``_stop_drain``) with notices already counted — the notice must not
        extend that critical section nor die with the drain task. Keeps a
        strong reference so the task cannot be garbage-collected mid-flight;
        ``_deliver_notices`` already swallows every delivery error.
        """
        task = asyncio.create_task(self._deliver_notices(session_id, notices))
        self._pending_notices.add(task)
        task.add_done_callback(self._pending_notices.discard)

    async def _deliver_notices(self, session_id: str, notices: list[_Notice]) -> None:
        """Best-effort delivery of queue-drop notices (design §9). Never raises.

        Every notice in a batch carries the SAME owner — resolved from the
        session's binding at entry (``_try_drain_one_locked``) or from the
        cleared target (``invalidate_binding``). Merging the texts and sending
        to ``notices[0]``'s owner is safe only by that invariant; do not add
        notices for other users to a batch.
        """
        if not notices or self._notify_user is None:
            return
        user_id = notices[0][0]
        text = "\n".join(t for _, t in notices)
        try:
            delivered = await self._notify_user(user_id=user_id, text=text)
        except Exception:
            logger.exception(
                "queue-drop notification raised",
                extra={"session_id": session_id, "user_id": user_id},
            )
            return
        if not delivered:
            logger.warning(
                "queue-drop notification not delivered",
                extra={"session_id": session_id, "user_id": user_id},
            )

    async def _try_drain_one_locked(self, session_id: str) -> tuple[_DrainStep, list[_Notice]]:
        notices: list[_Notice] = []
        binding = self._binding_store.get_binding(session_id)
        user_id = binding.user_id if binding is not None else None

        # Age out expired heads up front so timeouts surface as a counted,
        # notifyable drop instead of being silently swallowed by dequeue.
        pruned = await self._queue.prune_expired(session_id)
        if pruned and user_id is not None:
            notices.append((user_id, f"⏳ 已丢弃 {pruned} 条等待超时的排队消息。"))

        if await self._queue.peek_size(session_id) == 0:
            return _DrainStep.EMPTY, notices
        if session_id in self._in_flight or not self._is_sendable(session_id):
            return _DrainStep.WAIT, notices

        if binding is None or binding.ended_at is not None:
            return await self._abort_drain_locked(session_id, user_id=user_id, reason="会话已结束")
        target = binding.ghostty_target
        if target is None or target.binding_id != binding.binding_id:
            return await self._abort_drain_locked(session_id, user_id=user_id, reason="绑定已变更，请重新配对")

        process = await self._validate_foreground(
            pid=binding.pid or 0,
            paired_tty=target.paired_tty,
        )
        if not process.ok:
            return _DrainStep.WAIT, notices
        if not self._adapter.is_available():
            return _DrainStep.WAIT, notices
        terminal_ok, _terminal, terminal_error = await self._adapter.validate_terminal(target.terminal_id)
        if not terminal_ok:
            if terminal_error in {InjectionOutcome.NOT_FOUND, InjectionOutcome.NOT_UNIQUE}:
                self._binding_store.clear_ghostty_target(session_id, binding.binding_id)
                await self._mode_store.clear_target_for_session(session_id)
                return await self._abort_drain_locked(session_id, user_id=user_id, reason="配对的终端已失效")
            return _DrainStep.WAIT, notices

        current = self._current_binding_target(
            session_id=session_id,
            user_id=binding.user_id,
            expected_binding_id=binding.binding_id,
            expected_target=target,
        )
        if current is None:
            return await self._abort_drain_locked(session_id, user_id=user_id, reason="绑定已变更，请重新配对")
        binding, target = current

        before_dequeue = await self._queue.peek_size(session_id)
        entry: QueuedInput | None = await self._queue.dequeue(
            session_id,
            binding_id=binding.binding_id,
        )
        if entry is None:
            # dequeue silently discards heads that expired or went stale during
            # the multi-second validation awaits above (its internal
            # _drop_unusable_head_locked). Surface that drop as a notice
            # instead of letting the entries vanish uncounted.
            lost = before_dequeue - await self._queue.peek_size(session_id)
            if lost and user_id is not None:
                notices.append((user_id, f"⚠️ 已丢弃 {lost} 条过期或失效的排队消息。"))
            return _DrainStep.EMPTY, notices
        # A successful dequeue may still have skipped expired/stale entries in
        # front of it (same internal drop as above): before - after - 1 (the
        # entry now in hand) is what vanished silently.
        skipped = before_dequeue - await self._queue.peek_size(session_id) - 1
        if skipped and user_id is not None:
            notices.append((user_id, f"⚠️ 已丢弃 {skipped} 条过期或失效的排队消息。"))

        try:
            current = self._current_binding_target(
                session_id=session_id,
                user_id=binding.user_id,
                expected_binding_id=binding.binding_id,
                expected_target=target,
            )
            if current is None:
                # The entry belongs to the old generation/target; abort clears
                # the remaining old queue rather than injecting after rebind.
                # The just-dequeued entry is dropped too — count it (design §9).
                return await self._abort_drain_locked(
                    session_id,
                    user_id=user_id,
                    reason="绑定代际已变更",
                    extra_dropped=1,
                )
            binding, target = current

            # As in the immediate-send path, close the AppleScript validation
            # TOCTOU window by checking the process again at the last possible
            # point. Restore the FIFO head if the process changed meanwhile.
            process = await self._validate_foreground(
                pid=binding.pid or 0,
                paired_tty=target.paired_tty,
            )
            if not process.ok:
                restored = await self._queue.prepend(session_id, entry)
                if not restored:
                    logger.warning(
                        "failed to restore queued input after process revalidation",
                        extra={"session_id": session_id},
                    )
                    if user_id is not None:
                        notices.append((user_id, "⚠️ 有 1 条排队消息恢复失败已被丢弃。"))
                return _DrainStep.WAIT, notices

            self._injecting.add(session_id)
            try:
                outcome = await self._adapter.inject_text(target.terminal_id, entry.text)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("ghostty drain inject raised", extra={"session_id": session_id})
                # Drop the flag before the awaited cleanup. `return await` runs
                # before `finally`, and a Stop during that await must not be ignored.
                self._injecting.discard(session_id)
                return await self._drop_current_and_continue(
                    session_id,
                    user_id=user_id,
                    notices=notices,
                    detail="内部错误",
                    terminal_state_unknown=True,
                )
            else:
                if outcome == InjectionOutcome.OK:
                    # Same window as send_text: hold _injecting across arming.
                    await self._mark_in_flight(session_id)
            finally:
                self._injecting.discard(session_id)
            if outcome == InjectionOutcome.OK:
                return _DrainStep.INJECTED, notices
            if outcome in _RETRYABLE_PREFLIGHT_OUTCOMES:
                # Preflight failure before anything was pasted (same set send_text
                # maps to ADAPTER_UNAVAILABLE): restore the head and wait for
                # Ghostty to become reachable again.
                restored = await self._queue.prepend(session_id, entry)
                if not restored:
                    logger.warning(
                        "failed to restore queued input after preflight failure",
                        extra={"session_id": session_id, "outcome": outcome},
                    )
                    return await self._drop_current_and_continue(
                        session_id,
                        user_id=user_id,
                        notices=notices,
                        detail=f"{outcome}，恢复失败",
                    )
                logger.info(
                    "ghostty drain deferred until adapter recovers",
                    extra={"session_id": session_id, "outcome": outcome},
                )
                return _DrainStep.WAIT, notices
            logger.warning(
                "ghostty drain inject failed",
                extra={"session_id": session_id, "outcome": outcome},
            )
            # The dequeued entry may or may not have been pasted (INDETERMINATE/
            # TIMEOUT/OS_ERROR): it cannot be retried safely. The rest of the
            # queue was never touched and must survive — and because the input
            # line may still hold the unsubmitted head, it must wait for a real
            # session event rather than being injected right away.
            return await self._drop_current_and_continue(
                session_id,
                user_id=user_id,
                notices=notices,
                detail=outcome,
                terminal_state_unknown=True,
            )
        except asyncio.CancelledError:
            # _stop_drain cancelled this drain while the dequeued entry was in
            # hand (leave / invalidate / target switch): it will never be
            # injected, and the caller's queue clear() can no longer see it.
            # Report the drop (design §9) in the background, then let the
            # cancellation propagate. The entry's text may already be pasted
            # without Enter, hence the line-check hint.
            self._injecting.discard(session_id)
            if user_id is not None:
                self._spawn_notice_delivery(
                    session_id,
                    [(user_id, "⚠️ 有 1 条排队消息在退出或切换输入模式时被丢弃，请到终端确认输入行没有残留文本。")],
                )
            raise


def _normalise_text(text: str) -> str:
    """Normalise CRLF/CR to LF (design §8). No shell escaping."""
    return normalize_line_endings(text)
