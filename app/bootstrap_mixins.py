from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from inspect import iscoroutine
from pathlib import Path
from typing import Any, cast

from app.bootstrap_base import AppContainerBase
from app.config.settings import is_workdir_allowed
from app.domain.external_session_models import ExternalBinding, OwnershipResult
from app.domain.external_session_models import SessionOrigin as ExternalSessionOrigin
from app.domain.hook_models import HookEvent
from app.domain.models import SessionContext, TaskStatus, utc_now
from app.domain.session_models import (
    IDLE_PHASES,
    INTERRUPTIBLE_PHASES,
    ConversationTurn,
    FileSyncedPayload,
    HookReceivedPayload,
    PermissionDecisionPayload,
    PermissionResponseFailedPayload,
    SessionEvent,
    SessionEventType,
    SessionPhase,
    SessionState,
)
from app.domain.user_question_models import (
    ExternalGhosttyQuestionTarget,
    ExternalTmuxQuestionTarget,
    UserQuestionPrompt,
    extract_user_question_prompts,
)
from app.infra.text_formatting import ensure_aware_utc
from app.services.external_reply_delivery_pump import ExternalReplyDrainResult
from app.services.permission_callback_registry import AutoApproveOutcome, SessionOrigin
from app.services.user_question_callback_registry import UserQuestionCallbackOrigin

logger = logging.getLogger(__name__)


# Hook events that signal the same "turn ended / back to sendable phase" as ``Stop``:
# stop failed mid-flush, or compaction finished. ``notify_hook_event`` normalises only
# with ``.lower().replace("-","_")`` (no CamelCase→snake_case), so raw
# ``StopFailure`` / ``PostCompact`` would miss its ``stop`` branch — translate them here.
_HOOK_STOP_EQUIVALENTS = frozenset({"Stop", "StopFailure"})


def _map_hook_event_kind(event: HookEvent) -> str | None:
    """Map a real HookEvent to the snake_case kind ``notify_hook_event`` recognises.

    Returns None when the event is not phase-affecting for external input and should be ignored.
    SessionEnd / status="ended" → "session_end"; Stop-family → "stop"; PostCompact →
    "turn_completed" (compaction finished, phase returns to sendable). Other Hook event names are
    passed through verbatim — the service lowercases them, and only Stop/SessionEnd currently
    matter among the raw PascalCase Hook names.
    """
    if event.event == "SessionEnd" or event.status == "ended":
        return "session_end"
    if event.event in _HOOK_STOP_EQUIVALENTS:
        return "stop"
    if event.event == "PostCompact":
        return "turn_completed"
    return event.event


def _is_session_end_event(event: HookEvent) -> bool:
    return event.event == "SessionEnd" or event.status == "ended"


def _is_completed_assistant_reply(turn: ConversationTurn) -> bool:
    return turn.role == "assistant" and turn.is_complete and bool(turn.text.strip())


class _StageShortCircuitError(Exception):
    """Raised by a pipeline stage to terminate the rest of the stage list.

    The orchestration loop catches this, logs at INFO level, closes unawaited
    coroutines, and stops further stage execution. Not treated as an error.
    """

    def __init__(self, *, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class JsonlSyncMixin(AppContainerBase):
    """JSONL sync: debounced incremental parsing and event dispatch."""

    async def _invalidate_external_input(self, session_id: str, *, reason: str) -> None:
        """Tear down external input state (target/queue/pair-tokens/drain) for a session.

        No-op when the input service is not assembled (feature disabled or tests). The service
        acquires its own per-session input lock internally; callers here may already hold the
        reply-delivery lock, which is independent and order-safe. Never raises — cleanup failures
        must not prevent binding removal.
        """
        input_service = getattr(self, "external_session_input_service", None)
        if input_service is None:
            return
        try:
            await input_service.invalidate_binding(session_id, reason=reason)
        except Exception:
            logger.exception(
                "external input invalidate_binding failed",
                extra={"session_id": session_id, "reason": reason},
            )

    async def _rebind_external_input_aba(self, session_id: str, binding_id: str) -> None:
        """Clear stale-generation input state after a rebind produces a new binding_id."""
        input_service = getattr(self, "external_session_input_service", None)
        if input_service is None:
            return
        try:
            await input_service.rebind_aba(session_id, binding_id)
        except Exception:
            logger.exception(
                "external input rebind_aba failed",
                extra={"session_id": session_id, "binding_id": binding_id},
            )

    async def _save_external_binding(self, binding: ExternalBinding) -> bool:
        async with self._external_reply_delivery_locks.lock(binding.session_id):
            reaper = getattr(self, "external_binding_reaper", None)
            if reaper is not None and reaper.is_cleanup_in_progress(binding.session_id):
                return False
            if self.external_binding_store.get_binding(binding.session_id) is not None:
                return False
            self.external_binding_store.save_binding(binding)
            # Defensive ABA sweep: a same-session rebind (unbind then re-bind) produces a new
            # binding_id. Clear any input target/queue/drain left from the old generation so it
            # cannot drive the new binding. Already-current generation sees a no-op clear.
            await self._rebind_external_input_aba(binding.session_id, binding.binding_id)
            return True

    async def _mark_external_binding_ended(
        self,
        session_id: str,
        *,
        expected_binding_id: str | None = None,
        cleanup_callback: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        cwd: str | None = None
        async with self._external_reply_delivery_locks.lock(session_id):
            binding = self.external_binding_store.get_binding(session_id)
            if binding is None or (expected_binding_id is not None and binding.binding_id != expected_binding_id):
                return False
            self.external_binding_store.mark_ended(session_id, utc_now())
            if cleanup_callback is not None:
                await cleanup_callback()
            # SessionEnd does not go through the reaper: tear down input state here so queued
            # text and drain tasks do not outlive the session. Invalidates inside its own input
            # lock (independent of the reply-delivery lock held here).
            await self._invalidate_external_input(session_id, reason="session_end")
            cwd = binding.cwd
        if cwd is not None and hasattr(self, "external_reply_delivery_pump"):
            self.external_reply_delivery_pump.request_finalize(session_id=session_id, cwd=cwd)
        return True

    async def _finalize_bound_external_session(self, session_id: str) -> bool:
        async with self._jsonl_sync_locks.lock(session_id):
            async with self._external_reply_delivery_locks.lock(session_id):
                async with self._session_event_locks.lock(session_id):
                    binding = self.external_binding_store.get_binding(session_id)
                    if binding is None:
                        return True
                    if binding.ended_at is None:
                        return False
                    if self.settings.external_push_reply_enabled:
                        if not binding.reply_cursor_initialized:
                            return False
                        state = self.structured_session_store.get(session_id)
                        turns = tuple(state.turns) if state is not None else ()
                        cursor_id = binding.last_pushed_reply_turn_id
                        cursor_index = next(
                            (index for index, turn in enumerate(turns) if turn.turn_id == cursor_id),
                            None,
                        )
                        if cursor_index is not None:
                            has_pending_reply = any(_is_completed_assistant_reply(turn) for turn in turns[cursor_index + 1 :])
                        else:
                            bound_at = ensure_aware_utc(binding.bound_at)
                            has_pending_reply = any(
                                _is_completed_assistant_reply(turn) and ensure_aware_utc(turn.ended_at or turn.started_at) > bound_at
                                for turn in turns
                            )
                        if has_pending_reply:
                            return False
                    if hasattr(self, "push_notifier"):
                        delivered = await self.push_notifier.notify_session_end(
                            user_id=binding.user_id,
                            session_id=session_id,
                            cwd=binding.cwd,
                        )
                        if not delivered:
                            return False
                    self.external_binding_store.remove_binding(session_id)
                if hasattr(self, "push_notifier"):
                    self.push_notifier.discard_assistant_reply_progress(session_id)
                if hasattr(self, "session_supervisor"):
                    try:
                        await self.session_supervisor.forget(session_id)
                    except Exception:
                        logger.exception("external session watcher cleanup failed", extra={"session_id": session_id})
                return True

    async def _remove_external_binding(
        self,
        session_id: str,
        expected_binding_id: str | None = None,
        expected_last_activity_at: datetime | None = None,
        expected_pid: int | None = None,
    ) -> ExternalBinding | None:
        async with self._external_reply_delivery_locks.lock(session_id):
            binding = self.external_binding_store.get_binding(session_id)
            if binding is None or (expected_binding_id is not None and binding.binding_id != expected_binding_id):
                return None
            if expected_last_activity_at is not None and (
                binding.last_activity_at != expected_last_activity_at or binding.pid != expected_pid
            ):
                return None
            # Drop input state before the binding leaves the store, so a drain task cannot inject
            # into a session whose binding is gone. Covers manual unbind, dead-PID and idle-TTL
            # reaping. SessionEnd takes the other path via _mark_external_binding_ended.
            await self._invalidate_external_input(session_id, reason="reaper_remove")
            self.external_binding_store.remove_binding(session_id)
            if hasattr(self, "push_notifier"):
                self.push_notifier.discard_assistant_reply_progress(session_id)
            if hasattr(self, "external_reply_delivery_pump"):
                try:
                    await self.external_reply_delivery_pump.stop(session_id)
                except Exception:
                    logger.exception("external reply pump stop failed", extra={"session_id": session_id})
            if hasattr(self, "session_supervisor"):
                try:
                    await self.session_supervisor.forget(session_id)
                except Exception:
                    logger.exception("external session watcher cleanup failed", extra={"session_id": session_id})
            return binding

    async def _unbind_external_binding(
        self,
        session_id: str,
        expected_binding_id: str | None = None,
    ) -> ExternalBinding | None:
        binding = self.external_binding_store.get_binding(session_id)
        if binding is None or (expected_binding_id is not None and binding.binding_id != expected_binding_id):
            return None
        reaper = cast(Any, self).external_binding_reaper
        removed = await reaper.remove_with_cleanup(
            session_id,
            reason="manual_unbind",
            expected_binding_id=binding.binding_id,
            expected_last_activity_at=binding.last_activity_at,
            expected_pid=binding.pid,
        )
        return binding if removed else None

    async def sync_claude_session(self, session_id: str, cwd: str) -> None:
        async with self._jsonl_sync_locks.lock(session_id):
            snapshot = self.claude_jsonl_parser.parse_incremental(session_id=session_id, cwd=cwd)
            logger.info(
                "claude session synced",
                extra={
                    "session_id": session_id,
                    "cwd": cwd,
                    "turn_count": len(snapshot.turns),
                    "tool_call_count": len(snapshot.tool_calls),
                    "last_reply": snapshot.last_reply,
                    "last_reply_role": snapshot.last_reply_role,
                    "last_offset": snapshot.last_offset,
                    "clear_detected": snapshot.clear_detected,
                },
            )
            await self._dispatch_session_event(  # type: ignore[attr-defined]
                SessionEvent(
                    session_id=session_id,
                    type=SessionEventType.FILE_SYNCED,
                    payload=FileSyncedPayload.from_mapping(snapshot.to_payload()),
                )
            )

    async def _sync_and_baseline_external_reply(self, session_id: str, cwd: str) -> None:
        try:
            await self.sync_claude_session(session_id, cwd)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "initial external reply sync failed",
                extra={"session_id": session_id, "cwd": cwd},
            )
            if not getattr(self, "_stopping", False):
                self._schedule_jsonl_sync(session_id, cwd)
                if hasattr(self, "external_reply_delivery_pump"):
                    self.external_reply_delivery_pump.request_settle(session_id=session_id, cwd=cwd)
            return

        async with self._external_reply_delivery_locks.lock(session_id):
            async with self._session_event_locks.lock(session_id):
                state = self.structured_session_store.get(session_id)
                turns = tuple(state.turns) if state is not None else ()
            binding = self.external_binding_store.get_binding(session_id)
            if binding is None:
                return
            if binding.last_pushed_reply_turn_id is None:
                bound_at = ensure_aware_utc(binding.bound_at)
                latest_reply = next(
                    (
                        turn
                        for turn in reversed(turns)
                        if _is_completed_assistant_reply(turn) and ensure_aware_utc(turn.ended_at or turn.started_at) <= bound_at
                    ),
                    None,
                )
                self.external_binding_store.set_reply_cursor(
                    session_id,
                    latest_reply.turn_id if latest_reply is not None else None,
                )
        if getattr(self, "_stopping", False):
            return
        if hasattr(self, "session_supervisor"):
            self.session_supervisor.watch(session_id=session_id, workdir=cwd)
        if hasattr(self, "external_reply_delivery_pump"):
            self.external_reply_delivery_pump.ensure(session_id=session_id, cwd=cwd)

    def _schedule_jsonl_sync(self, session_id: str, cwd: str) -> None:
        self.session_supervisor.watch(session_id=session_id, workdir=cwd)
        self.session_supervisor.schedule_jsonl_sync(session_id, cwd)


class HookHandlingMixin(AppContainerBase):
    """Hook event handling: validate, bind session, dispatch events."""

    def _is_current_bound_ownership(
        self,
        event: HookEvent,
        ownership: OwnershipResult,
    ) -> bool:
        if ownership.origin != ExternalSessionOrigin.EXTERNAL or ownership.ownership_state != "bound":
            return True
        binding_store = getattr(self, "external_binding_store", None)
        if binding_store is None or ownership.binding_id is None:
            return False
        binding = binding_store.get_binding(event.session_id)
        return (
            binding is not None
            and binding.binding_id == ownership.binding_id
            and binding.user_id == ownership.owner_user_id
            and (binding.ended_at is None or _is_session_end_event(event))
        )

    async def _cleanup_session_end_permission_state(self, session_id: str) -> None:
        async def run_async_cleanup(
            label: str,
            cleanup: Callable[[], Awaitable[object]],
        ) -> None:
            try:
                await cleanup()
            except Exception:
                logger.exception(
                    "session end permission cleanup failed",
                    extra={"session_id": session_id, "step": label},
                )

        def run_sync_cleanup(label: str, cleanup: Callable[[], object]) -> None:
            try:
                cleanup()
            except Exception:
                logger.exception(
                    "session end permission cleanup failed",
                    extra={"session_id": session_id, "step": label},
                )

        if hasattr(self, "auto_approve_service"):
            await run_async_cleanup(
                "auto approve deactivation",
                lambda: self.auto_approve_service.deactivate_all_for_session(session_id),
            )
            await run_async_cleanup(
                "auto approve slot release",
                lambda: self.auto_approve_service.release_all_slots_for_session(session_id),
            )
        if hasattr(self, "permission_callback_registry"):
            await run_async_cleanup(
                "permission callback registry",
                lambda: self.permission_callback_registry.invalidate_session(session_id),
            )
        if hasattr(self, "unbound_permission_handler"):
            await run_async_cleanup(
                "unbound permission handler",
                lambda: self.unbound_permission_handler.invalidate_session(session_id),
            )
        if hasattr(self, "external_uq_state"):
            invalidator = getattr(self.external_uq_state, "invalidate_session", None)
            if callable(invalidator):
                run_sync_cleanup(
                    "external user question state",
                    lambda: invalidator(session_id),
                )
        if hasattr(self, "user_question_callback_registry"):
            await run_async_cleanup(
                "user question callback registry",
                lambda: self.user_question_callback_registry.invalidate_session(session_id),
            )
        if hasattr(self, "hook_socket_server"):
            await run_async_cleanup(
                "hook pending permissions",
                lambda: self.hook_socket_server.cancel_pending_permissions(session_id=session_id),
            )

    async def _notify_input_service_hook_event(self, event: HookEvent) -> None:
        """Forward a phase-affecting hook event to the external input service.

        No-op when the service is unavailable (not assembled, or feature disabled). Maps the raw
        HookEvent to the snake_case ``event_kind`` the service recognises (see
        ``_map_hook_event_kind``). Never raises — input notifications must not halt the hook
        pipeline; a failure here only delays draining until the next Hook wake or publish.
        """
        input_service = getattr(self, "external_session_input_service", None)
        if input_service is None:
            return
        kind = _map_hook_event_kind(event)
        if kind is None:
            return
        try:
            await input_service.notify_hook_event(session_id=event.session_id, event_kind=kind)
        except Exception:
            logger.exception(
                "external input notify_hook_event failed",
                extra={"session_id": event.session_id, "event": event.event, "kind": kind},
            )

    async def _handle_hook_event(self, event: HookEvent) -> None:
        if getattr(self, "_stopping", False):
            return
        logger.debug(
            "hook event received",
            extra={
                "session_id": event.session_id,
                "event": event.event,
                "status": event.status,
                "tool": event.tool,
            },
        )

        # Stage 1: Ownership resolution (gate — failure halts pipeline)
        ownership = await self._resolve_ownership_stage(event)
        if ownership is None:
            return

        # Let the external input service clear its in-flight marker / schedule a drain in response
        # to phase-affecting hook events (Stop-family, PostCompact, SessionEnd). Runs before the
        # bound stage list so a Stop arriving during injection releases the guard before the
        # session-event lock is taken for phase dispatch. Fail-closed: never blocks the pipeline.
        await self._notify_input_service_hook_event(event)

        # Stages 2+: each wrapped independently in error boundaries.
        # A stage may raise _StageShortCircuitError to terminate the pipeline early.
        stages = self._build_stage_list(event, ownership)
        executed_up_to = -1
        try:
            for i, (stage_name, stage_coro) in enumerate(stages):
                if getattr(self, "_stopping", False):
                    break
                if not self._is_current_bound_ownership(event, ownership):
                    logger.info(
                        "bound hook pipeline stopped after binding changed",
                        extra={
                            "stage_name": stage_name,
                            "session_id": event.session_id,
                            "event_type": event.event,
                            "binding_id": ownership.binding_id,
                            "owner_user_id": ownership.owner_user_id,
                        },
                    )
                    break
                try:
                    await stage_coro
                    executed_up_to = i
                except _StageShortCircuitError as sc:
                    logger.info(
                        "hook pipeline short-circuited",
                        extra={
                            "stage_name": stage_name,
                            "reason": sc.reason,
                            "session_id": event.session_id,
                            "event_type": event.event,
                        },
                    )
                    executed_up_to = i
                    break
                except Exception:
                    logger.exception(
                        "hook pipeline stage failed",
                        extra={
                            "stage_name": stage_name,
                            "session_id": getattr(event, "session_id", None),
                            "event_type": getattr(event, "event", None),
                            "hook_cwd": getattr(event, "cwd", None),
                        },
                    )
        finally:
            # Closing skipped coroutines also covers cancellation during a stage.
            for j in range(executed_up_to + 1, len(stages)):
                coro = stages[j][1]
                if iscoroutine(coro):
                    coro.close()

    async def _resolve_ownership_stage(self, event: HookEvent) -> OwnershipResult | None:
        """Gate stage: workdir check, SessionEnd cleanup, and ownership resolution.

        Returns the OwnershipResult on success, or None if the event should be
        skipped (rejected by workdir allowlist or handled via legacy fallback).
        Exceptions are logged as ERROR with stage_name="ownership_resolution" and
        the method returns None so the pipeline halts gracefully.
        """
        try:
            # Workdir allowlist check
            if not is_workdir_allowed(event.cwd, self.settings.allowed_workdirs):
                logger.warning(
                    "hook event rejected by workdir allowlist",
                    extra={"session_id": event.session_id, "cwd": event.cwd, "event": event.event},
                )
                return None

            is_session_end = _is_session_end_event(event)

            # If ownership_resolver is not wired (e.g. in tests), fall back to old behavior.
            if not hasattr(self, "ownership_resolver"):
                if is_session_end:
                    await self._cleanup_session_end_permission_state(event.session_id)
                    if hasattr(self, "external_binding_store"):
                        self.external_binding_store.remove_binding(event.session_id)
                await self._bind_hook_session(event)
                await self._dispatch_session_event(  # type: ignore[attr-defined]
                    SessionEvent(
                        session_id=event.session_id,
                        type=SessionEventType.HOOK_RECEIVED,
                        payload=HookReceivedPayload.from_hook_event(event),
                    )
                )
                self._schedule_jsonl_sync(event.session_id, event.cwd)  # type: ignore[attr-defined]
                return None

            # Resolve before SessionEnd cleanup so an in-flight old generation
            # cannot clear permission state belonging to a replacement binding.
            ownership = await self.ownership_resolver.resolve(event.session_id)

            if is_session_end:
                if ownership.origin == ExternalSessionOrigin.EXTERNAL and ownership.ownership_state == "bound":
                    marked = await self._mark_external_binding_ended(  # type: ignore[attr-defined]
                        event.session_id,
                        expected_binding_id=ownership.binding_id,
                        cleanup_callback=lambda: self._cleanup_session_end_permission_state(event.session_id),
                    )
                    if not marked:
                        return None
                else:
                    await self._cleanup_session_end_permission_state(event.session_id)

            if not is_session_end and ownership.origin == ExternalSessionOrigin.EXTERNAL and hasattr(self, "external_discovery"):
                is_ended = getattr(self.external_discovery, "is_session_ended", None)
                if callable(is_ended) and is_ended(event.session_id):
                    if event.expects_response and hasattr(self, "hook_socket_server"):
                        await self.hook_socket_server.cancel_pending_permissions(session_id=event.session_id)
                    return None

            if (
                not is_session_end
                and ownership.origin == ExternalSessionOrigin.EXTERNAL
                and ownership.ownership_state == "unbound"
                and hasattr(self, "session_service")
            ):
                match_session_context = cast(
                    Callable[[HookEvent], Awaitable[SessionContext | None]] | None,
                    getattr(cast(Any, self), "_match_session_context", None),
                )
                if match_session_context is not None:
                    matched = await match_session_context(event)
                    if matched is not None and matched.terminal_id is not None:
                        expected_tmux_session = None
                        actual_tmux_session = None
                        tmux_runner = getattr(self, "tmux_runner", None)
                        if event.pid is not None and event.pid > 0 and tmux_runner is not None:
                            from app.adapters.process.pty_injector import find_tmux_session_for_pid

                            expected_tmux_session = tmux_runner.build_session_name(matched.terminal_id)
                            settings = getattr(self, "settings", None)
                            tmux_bin = getattr(settings, "tmux_bin", "tmux")
                            try:
                                actual_tmux_session = await find_tmux_session_for_pid(event.pid, tmux_bin)
                            except Exception:
                                logger.debug(
                                    "failed to verify hook tmux session",
                                    extra={
                                        "session_id": event.session_id,
                                        "pid": event.pid,
                                        "terminal_id": matched.terminal_id,
                                        "expected_tmux_session": expected_tmux_session,
                                    },
                                    exc_info=True,
                                )
                        if actual_tmux_session == expected_tmux_session and expected_tmux_session is not None:
                            ownership = OwnershipResult(
                                owner_user_id=matched.user_id,
                                origin=ExternalSessionOrigin.TMUX,
                                ownership_state="owned",
                            )
                            logger.info(
                                "unbound hook event matched active tmux session",
                                extra={
                                    "session_id": event.session_id,
                                    "owner_user_id": matched.user_id,
                                    "terminal_id": matched.terminal_id,
                                    "workdir": matched.workdir,
                                    "pid": event.pid,
                                    "tmux_session": actual_tmux_session,
                                },
                            )
                        else:
                            logger.info(
                                "unbound hook event rejected by tmux identity check",
                                extra={
                                    "session_id": event.session_id,
                                    "owner_user_id": matched.user_id,
                                    "terminal_id": matched.terminal_id,
                                    "workdir": matched.workdir,
                                    "pid": event.pid,
                                    "expected_tmux_session": expected_tmux_session,
                                    "actual_tmux_session": actual_tmux_session,
                                },
                            )

            if is_session_end and ownership.origin == ExternalSessionOrigin.EXTERNAL:
                if hasattr(self, "external_discovery"):
                    marker = getattr(self.external_discovery, "mark_session_ended", None)
                    if callable(marker):
                        marker(event.session_id)
                    else:
                        self.external_discovery.remove_session(event.session_id)
            logger.info(
                "hook event ownership resolved",
                extra={
                    "session_id": event.session_id,
                    "ownership_state": ownership.ownership_state,
                    "origin": ownership.origin.value,
                    "owner_user_id": ownership.owner_user_id,
                },
            )

            # Refresh activity timestamp on bound external hook events so that
            # the periodic stale-binding cleanup keeps active sessions alive.
            # Skipped for SessionEnd (which removes the binding above) and for
            # tmux-owned or unbound events.
            if (
                ownership.origin == ExternalSessionOrigin.EXTERNAL
                and ownership.ownership_state == "bound"
                and not is_session_end
                and hasattr(self, "external_binding_store")
                and self._is_current_bound_ownership(event, ownership)
            ):
                self.external_binding_store.touch_activity(event.session_id, utc_now(), pid=event.pid, tty=event.tty)

            return ownership
        except Exception:
            logger.exception(
                "hook pipeline stage failed",
                extra={
                    "stage_name": "ownership_resolution",
                    "session_id": event.session_id,
                    "event_type": event.event,
                    "hook_cwd": event.cwd,
                },
            )
            return None

    def _build_stage_list(self, event: HookEvent, ownership: OwnershipResult) -> list[tuple[str, Awaitable[None]]]:
        """Build the ordered list of pipeline stages based on ownership state.

        Returns a list of (stage_name, coroutine) tuples. Each coroutine is a
        zero-arg awaitable that captures the needed context from event/ownership.
        """
        stages: list[tuple[str, Awaitable[None]]] = []

        if ownership.ownership_state == "owned":
            # Session binding MUST run before auto-approve check so that
            # structured_session_store is updated even when short-circuited.
            stages.append(
                (
                    "session_binding",
                    self._bind_hook_session(event),
                )
            )
            # Event dispatch
            stages.append(
                (
                    "event_dispatch",
                    self._dispatch_session_event(  # type: ignore[attr-defined]
                        SessionEvent(
                            session_id=event.session_id,
                            type=SessionEventType.HOOK_RECEIVED,
                            payload=HookReceivedPayload.from_hook_event(event),
                        )
                    ),
                )
            )

            # JSONL sync scheduling (sync, not async — wrap in a trivial coroutine)
            async def _schedule_jsonl_owned() -> None:
                self._schedule_jsonl_sync(event.session_id, event.cwd)  # type: ignore[attr-defined]

            stages.append(("jsonl_sync_scheduling", _schedule_jsonl_owned()))

            # Auto-approve check — may short-circuit, skipping only auto_file_send
            stages.append(
                (  # type: ignore[arg-type]
                    "auto_approve_check",
                    self._run_auto_approve_check(
                        event,
                        origin=SessionOrigin.OWNED,
                        candidate_user_id=ownership.owner_user_id,
                    ),
                )
            )

            # Auto-file-send (sync — wrap in a trivial coroutine)
            async def _auto_file_send_owned() -> None:
                self._maybe_auto_file_send(event, ownership.owner_user_id)

            stages.append(("auto_file_send", _auto_file_send_owned()))

        elif ownership.ownership_state == "bound":
            # Event dispatch MUST run before auto-approve check
            stages.append(
                (
                    "event_dispatch",
                    self._dispatch_session_event(  # type: ignore[attr-defined]
                        SessionEvent(
                            session_id=event.session_id,
                            type=SessionEventType.HOOK_RECEIVED,
                            payload=HookReceivedPayload.from_hook_event(event),
                        )
                    ),
                )
            )

            # Stop is flushed synchronously by reply delivery; SessionEnd gets one final direct sync without re-watching.
            async def _schedule_jsonl_bound() -> None:
                if _is_session_end_event(event):
                    await self.sync_claude_session(event.session_id, event.cwd)  # type: ignore[attr-defined]
                    return
                if self.settings.external_push_reply_enabled and hasattr(self, "external_reply_delivery_pump"):
                    self.external_reply_delivery_pump.ensure(session_id=event.session_id, cwd=event.cwd)
                if event.event == "Stop" and self.settings.external_push_reply_enabled:
                    self.session_supervisor.watch(session_id=event.session_id, workdir=event.cwd)
                    return
                self._schedule_jsonl_sync(event.session_id, event.cwd)  # type: ignore[attr-defined]

            stages.append(("jsonl_sync_scheduling", _schedule_jsonl_bound()))

            # Permission decisions are serialized with unbind/rebind so an old
            # owner cannot approve after the binding generation changes.
            async def _auto_approve_bound() -> None:
                async with self._external_reply_delivery_locks.lock(event.session_id):
                    if not self._is_current_bound_ownership(event, ownership):
                        raise _StageShortCircuitError(reason="binding-changed")
                    await self._run_auto_approve_check(
                        event,
                        origin=SessionOrigin.EXTERNAL_BOUND,
                        candidate_user_id=ownership.owner_user_id,
                    )

            stages.append(("auto_approve_check", _auto_approve_bound()))

            # Permission and user-question pushes share the same generation
            # barrier. Stop reply delivery already acquires this lock internally.
            async def _push_notification_bound() -> None:
                if not hasattr(self, "push_notifier") or ownership.owner_user_id is None:
                    return
                if event.event == "Stop":
                    await self._notify_bound_external_event(event, ownership.owner_user_id)
                    return
                async with self._external_reply_delivery_locks.lock(event.session_id):
                    if not self._is_current_bound_ownership(event, ownership):
                        return
                    await self._notify_bound_external_event(event, ownership.owner_user_id)

            stages.append(("push_notification", _push_notification_bound()))

            # Auto-file-send
            async def _auto_file_send_bound() -> None:
                if (
                    event.event == "PostToolUse"
                    and event.tool == "Write"
                    and ownership.owner_user_id is not None
                    and hasattr(self, "file_sender")
                ):
                    self._background_tasks.spawn(self._send_bound_file_if_current(event, ownership))

            stages.append(("auto_file_send", _auto_file_send_bound()))

        else:
            # Unbound
            # External discovery record
            async def _external_discovery() -> None:
                if hasattr(self, "external_discovery"):
                    if _is_session_end_event(event):
                        return
                    self.external_discovery.record_event(event)

            stages.append(("external_discovery", _external_discovery()))

            # Permission handling
            async def _permission_handling() -> None:
                if event.expects_response and hasattr(self, "unbound_permission_handler"):
                    if not await self._is_unbound_tmux_event(event):
                        await self._release_unbound_non_tmux_permission(event)
                        return
                    candidate_user_id = None
                    if hasattr(self, "auto_approve_service"):
                        candidate_user_id = self.auto_approve_service.get_active_user_for_session(event.session_id)
                    outcome = await self._run_auto_approve_check(
                        event,
                        origin=SessionOrigin.EXTERNAL_UNBOUND,
                        candidate_user_id=candidate_user_id,
                    )
                    if outcome in {AutoApproveOutcome.APPROVED, AutoApproveOutcome.APPROVAL_UNKNOWN}:
                        return
                    await self.unbound_permission_handler.handle_unbound_permission(event)

            stages.append(("permission_handling", _permission_handling()))

        return stages

    async def _run_auto_approve_check(
        self,
        event: HookEvent,
        *,
        origin: SessionOrigin = SessionOrigin.EXTERNAL_BOUND,
        candidate_user_id: int | None = None,
    ) -> AutoApproveOutcome:
        """Check if the event should be auto-approved through PermissionGateway.

        Successful or unknown auto-approval short-circuits downstream prompt stages.
        Failed auto-approval falls back to normal interactive notification.
        """
        if not event.expects_response or event.tool == "AskUserQuestion":
            return AutoApproveOutcome.NOT_APPROVED
        if not event.tool_use_id or not hasattr(self, "permission_gateway"):
            return AutoApproveOutcome.NOT_APPROVED

        outcome = await self.permission_gateway.maybe_auto_approve(
            session_id=event.session_id,
            origin=origin,
            candidate_user_id=candidate_user_id,
            tool_use_id=event.tool_use_id,
            tool_name=event.tool or "unknown tool",
            tool_input=event.tool_input,
        )
        if outcome in {AutoApproveOutcome.APPROVED, AutoApproveOutcome.APPROVAL_UNKNOWN}:
            raise _StageShortCircuitError(reason="auto-approved")
        return outcome

    async def _is_unbound_tmux_event(self, event: HookEvent) -> bool:
        if event.pid is None or event.pid <= 0:
            return False
        from app.adapters.process.pty_injector import find_tmux_pane_for_pid

        settings = getattr(self, "settings", None)
        tmux_bin = getattr(settings, "tmux_bin", "tmux")
        try:
            return await find_tmux_pane_for_pid(event.pid, tmux_bin) is not None
        except Exception:
            logger.debug(
                "failed to detect tmux pane for unbound permission",
                extra={"session_id": event.session_id, "tool_use_id": event.tool_use_id, "pid": event.pid},
                exc_info=True,
            )
            return False

    async def _release_unbound_non_tmux_permission(self, event: HookEvent) -> None:
        logger.info(
            "skip unbound non-tmux permission push",
            extra={"session_id": event.session_id, "tool_use_id": event.tool_use_id, "tool": event.tool, "pid": event.pid},
        )
        if not event.tool_use_id:
            return
        hook_socket_server = getattr(self, "hook_socket_server", None)
        release_pending_permission = getattr(hook_socket_server, "release_pending_permission", None)
        if release_pending_permission is None:
            return
        try:
            await release_pending_permission(tool_use_id=event.tool_use_id)
        except Exception:
            logger.warning(
                "failed to release unbound non-tmux permission",
                extra={"session_id": event.session_id, "tool_use_id": event.tool_use_id},
                exc_info=True,
            )

    async def _send_bound_file_if_current(
        self,
        event: HookEvent,
        ownership: OwnershipResult,
    ) -> None:
        async with self._external_reply_delivery_locks.lock(event.session_id):
            owner_user_id = ownership.owner_user_id
            if owner_user_id is None or not self._is_current_bound_ownership(event, ownership):
                return
            file_path_raw = event.tool_input.get("file_path", "") if event.tool_input else ""
            file_sender = cast(Any, self).file_sender
            await file_sender.send_if_eligible(
                file_path_raw=file_path_raw,
                cwd=event.cwd,
                chat_id=owner_user_id,
            )

    def _maybe_auto_file_send(self, event: HookEvent, owner_user_id: int | None) -> None:
        if event.event == "PostToolUse" and event.tool == "Write" and owner_user_id is not None and hasattr(self, "file_sender"):
            file_path_raw = event.tool_input.get("file_path", "") if event.tool_input else ""
            self._background_tasks.spawn(
                self.file_sender.send_if_eligible(
                    file_path_raw=file_path_raw,
                    cwd=event.cwd,
                    chat_id=owner_user_id,
                )
            )

    async def _stop_background_tasks(self) -> None:
        await self._background_tasks.cancel_all()
        await self._upload_background_tasks.cancel_all()
        await self._stream_background_tasks.cancel_all()

    async def _notify_bound_external_event(self, event: HookEvent, user_id: int) -> None:
        """Send push notifications for bound external session events."""
        if not hasattr(self, "push_notifier"):
            return
        if _is_session_end_event(event):
            return
        if event.expects_response:
            # AskUserQuestion: route to an interactive external question card
            # (Ghostty or legacy tmux) when a target is available; otherwise fall
            # through to the generic permission confirmation card.
            if event.tool == "AskUserQuestion":
                prompts = extract_user_question_prompts(
                    tool_use_id=event.tool_use_id or "",
                    tool_name=event.tool,
                    tool_input=event.tool_input,
                )
                if prompts and hasattr(self, "external_uq_state"):
                    # Ghostty-bound session: inject via the verified transport.
                    ghostty_handled = await self._try_ghostty_user_question(event=event, user_id=user_id, prompts=prompts)
                    if ghostty_handled:
                        return

                    # Legacy tmux: PTY injection when a pane is reachable for the PID.
                    if event.pid is not None:
                        from app.adapters.process.pty_injector import find_tmux_pane_for_pid

                        pane_id = await find_tmux_pane_for_pid(event.pid, self.settings.tmux_bin)
                        if pane_id is not None:
                            from app.services.external_user_question_state import PendingExternalUserQuestion

                            pending = PendingExternalUserQuestion(
                                tool_use_id=event.tool_use_id or "",
                                session_id=event.session_id,
                                user_id=user_id,
                                prompts=prompts,
                                target=ExternalTmuxQuestionTarget(
                                    pane_id=pane_id,
                                    tmux_bin=self.settings.tmux_bin,
                                ),
                            )
                            self.external_uq_state.store(pending)
                            await self.push_notifier.notify_user_question(
                                user_id=user_id,
                                session_id=event.session_id,
                                prompts=prompts,
                                interactive=True,
                                origin=UserQuestionCallbackOrigin.EXTERNAL_TMUX,
                            )
                            return

                # Fallback: no tmux pane found or no PID — fall through to normal
                # permission flow (notify_permission_request below). The user sees
                # the permission request in Telegram and clicks allow; Claude Code
                # then shows the question UI in the terminal.
                pass
            # Resolve title for permission notification
            _title: str | None = None
            if hasattr(self, "claude_jsonl_parser"):
                try:
                    _title = self.claude_jsonl_parser.extract_session_title(session_id=event.session_id, cwd=event.cwd)
                except Exception:
                    pass
            await self.push_notifier.notify_permission_request(
                user_id=user_id,
                session_id=event.session_id,
                tool_name=event.tool or "",
                tool_input=event.tool_input,
                tool_use_id=event.tool_use_id or "",
                cwd=event.cwd,
                title=_title,
            )
        elif event.event == "Stop":
            await self._push_bound_assistant_replies(event, user_id)

    async def _try_ghostty_user_question(
        self,
        *,
        event: HookEvent,
        user_id: int,
        prompts: tuple[UserQuestionPrompt, ...],
    ) -> bool:
        """Route an AskUserQuestion to a bound, paired Ghostty session.

        Returns True when a Ghostty target exists for the bound session and the
        interactive question card has been pushed (the Hook permission is held
        until the user answers and the transport reports completion). Returns
        False to let the caller fall through to the tmux / generic card.
        """
        if not hasattr(self, "external_binding_store"):
            return False
        binding = self.external_binding_store.get_binding(event.session_id)
        if binding is None or binding.ended_at is not None or binding.user_id != user_id:
            return False
        ghostty_target = binding.ghostty_target
        if ghostty_target is None:
            return False
        from app.services.external_user_question_state import (
            ExternalUserQuestionState,
            PendingExternalUserQuestion,
        )

        state: ExternalUserQuestionState = self.external_uq_state
        question_target = ExternalGhosttyQuestionTarget(
            binding_id=ghostty_target.binding_id,
            terminal_id=ghostty_target.terminal_id,
            paired_tty=ghostty_target.paired_tty,
            paired_at=ghostty_target.paired_at,
        )
        pending = PendingExternalUserQuestion(
            tool_use_id=event.tool_use_id or "",
            session_id=event.session_id,
            user_id=user_id,
            prompts=prompts,
            target=question_target,
        )
        state.store(pending)
        await self.push_notifier.notify_user_question(
            user_id=user_id,
            session_id=event.session_id,
            prompts=prompts,
            interactive=True,
            origin=UserQuestionCallbackOrigin.EXTERNAL_GHOSTTY,
        )
        return True

    async def _push_bound_assistant_replies(self, event: HookEvent, user_id: int) -> ExternalReplyDrainResult:
        if not self.settings.external_push_reply_enabled:
            return ExternalReplyDrainResult.NO_NEW_REPLY

        if hasattr(self, "external_reply_delivery_pump"):
            self.external_reply_delivery_pump.ensure(session_id=event.session_id, cwd=event.cwd)

        try:
            await self.sync_claude_session(event.session_id, event.cwd)  # type: ignore[attr-defined]
        except Exception:
            logger.exception(
                "bound assistant reply sync failed",
                extra={"session_id": event.session_id, "user_id": user_id},
            )
            self._schedule_jsonl_sync(event.session_id, event.cwd)  # type: ignore[attr-defined]
            if hasattr(self, "external_reply_delivery_pump"):
                self.external_reply_delivery_pump.request_settle(session_id=event.session_id, cwd=event.cwd)
            return ExternalReplyDrainResult.DELIVERY_FAILED

        result = await self._drain_bound_assistant_replies(event.session_id)
        if hasattr(self, "external_reply_delivery_pump"):
            self.external_reply_delivery_pump.request_settle(session_id=event.session_id, cwd=event.cwd)
        return result

    async def _drain_bound_assistant_replies(self, session_id: str) -> ExternalReplyDrainResult:
        if not self.settings.external_push_reply_enabled:
            return ExternalReplyDrainResult.NO_NEW_REPLY

        async with self._external_reply_delivery_locks.lock(session_id):
            async with self._session_event_locks.lock(session_id):
                state = self.structured_session_store.get(session_id)
                turns = tuple(state.turns) if state is not None else ()
            binding = self.external_binding_store.get_binding(session_id)
            if binding is None:
                return ExternalReplyDrainResult.NO_NEW_REPLY
            user_id = binding.user_id
            if not binding.reply_cursor_initialized:
                bound_at = ensure_aware_utc(binding.bound_at)
                latest_reply = next(
                    (
                        turn
                        for turn in reversed(turns)
                        if _is_completed_assistant_reply(turn) and ensure_aware_utc(turn.ended_at or turn.started_at) <= bound_at
                    ),
                    None,
                )
                cursor_id = latest_reply.turn_id if latest_reply is not None else None
                self.external_binding_store.set_reply_cursor(session_id, cursor_id)
                logger.info(
                    "legacy bound assistant reply cursor initialized",
                    extra={"session_id": session_id, "user_id": user_id},
                )
            else:
                cursor_id = binding.last_pushed_reply_turn_id
            cursor_index = next(
                (index for index, turn in enumerate(turns) if turn.turn_id == cursor_id),
                None,
            )
            if cursor_index is not None:
                pending_replies = [turn for turn in turns[cursor_index + 1 :] if _is_completed_assistant_reply(turn)]
            else:
                if cursor_id is not None:
                    logger.warning(
                        "bound assistant reply cursor not found; recovering from bind time",
                        extra={"session_id": session_id, "user_id": user_id, "turn_id": cursor_id},
                    )
                bound_at = ensure_aware_utc(binding.bound_at)
                pending_replies = [
                    turn
                    for turn in turns
                    if _is_completed_assistant_reply(turn) and ensure_aware_utc(turn.ended_at or turn.started_at) > bound_at
                ]

            if not pending_replies:
                return ExternalReplyDrainResult.NO_NEW_REPLY

            for turn in pending_replies:
                delivered = await self.push_notifier.notify_assistant_reply(
                    user_id=user_id,
                    session_id=session_id,
                    text=turn.text,
                    title=binding.title,
                    turn_id=turn.turn_id,
                )
                if not delivered:
                    logger.warning(
                        "bound assistant reply delivery failed",
                        extra={"session_id": session_id, "user_id": user_id, "turn_id": turn.turn_id},
                    )
                    return ExternalReplyDrainResult.DELIVERY_FAILED
                self.external_binding_store.set_reply_cursor(session_id, turn.turn_id)
            return ExternalReplyDrainResult.DELIVERED

    async def _handle_permission_failure(self, session_id: str, tool_use_id: str) -> None:
        logger.warning(
            "permission response failed",
            extra={"session_id": session_id, "tool_use_id": tool_use_id},
        )
        await self._dispatch_session_event(  # type: ignore[attr-defined]
            SessionEvent(
                session_id=session_id,
                type=SessionEventType.PERMISSION_RESPONSE_FAILED,
                payload=PermissionResponseFailedPayload(tool_use_id=tool_use_id),
            )
        )
        # Update permission callback registry and edit Telegram message
        if hasattr(self, "permission_callback_registry"):
            # Get the record before invalidating to preserve message info
            record = await self.permission_callback_registry.get_record_by_tool_use_id(session_id, tool_use_id)
            await self.permission_callback_registry.invalidate_pending_for_tool(
                session_id=session_id,
                tool_use_id=tool_use_id,
                reason="permission_response_failed",
            )
            # Edit the Telegram message if we have the message info
            if record and record.telegram_chat_id and record.telegram_message_id:
                from app.bot.handlers.callback_utils import build_approval_message

                approval_text = "⚠️ 响应失败（超时或连接断开）"
                try:
                    original_text = record.telegram_message_text or ""
                    new_text = build_approval_message(original_text, approval_text)
                    await self.message_sender.edit_message(
                        chat_id=record.telegram_chat_id,
                        message_id=record.telegram_message_id,
                        text=new_text,
                        parse_mode="HTML",
                    )
                    logger.info(
                        "telegram message updated for permission failure",
                        extra={"session_id": session_id, "tool_use_id": tool_use_id},
                    )
                except Exception:
                    logger.warning(
                        "failed to edit Telegram message for permission failure",
                        extra={"session_id": session_id, "tool_use_id": tool_use_id},
                        exc_info=True,
                    )

    async def _handle_permission_resolved(self, session_id: str, tool_use_id: str, reason: str) -> None:
        """Handle permission resolved in terminal (not via Telegram)."""
        logger.info(
            "permission resolved in terminal session_id=%s tool_use_id=%s reason=%s",
            session_id,
            tool_use_id,
            reason,
        )
        is_approved = reason == "terminal_approved"
        # Dispatch permission event to update session state
        await self._dispatch_session_event(  # type: ignore[attr-defined]
            SessionEvent(
                session_id=session_id,
                type=SessionEventType.PERMISSION_APPROVED if is_approved else SessionEventType.PERMISSION_DENIED,
                payload=PermissionDecisionPayload(tool_use_id=tool_use_id, source="terminal"),
            )
        )
        # Update permission callback registry and edit Telegram message
        if hasattr(self, "permission_callback_registry"):
            # Get the record before invalidating to preserve message info
            record = await self.permission_callback_registry.get_record_by_tool_use_id(session_id, tool_use_id)
            if record is None:
                logger.warning(
                    "permission record not found for terminal resolution session_id=%s tool_use_id=%s reason=%s",
                    session_id,
                    tool_use_id,
                    reason,
                )
            else:
                logger.info(
                    "permission record found for terminal resolution session_id=%s tool_use_id=%s reason=%s has_telegram_message=%s record_status=%s record_decision=%s chat_id=%s message_id=%s",
                    session_id,
                    tool_use_id,
                    reason,
                    bool(record.telegram_chat_id and record.telegram_message_id),
                    record.status,
                    record.decision,
                    record.telegram_chat_id,
                    record.telegram_message_id,
                )
            transitioned_count = await self.permission_callback_registry.invalidate_pending_for_tool(
                session_id=session_id,
                tool_use_id=tool_use_id,
                reason=reason,
            )
            # Edit the Telegram message only when this resolution actually transitioned
            # the callback record. Late duplicate terminal events should not rewrite it.
            if transitioned_count > 0 and record and record.telegram_chat_id and record.telegram_message_id:
                from app.bot.handlers.callback_utils import build_approval_message

                approval_text = "✅ 已在终端批准" if is_approved else "❌ 已在终端拒绝"
                try:
                    original_text = record.telegram_message_text or ""
                    new_text = build_approval_message(original_text, approval_text)
                    await self.message_sender.edit_message(
                        chat_id=record.telegram_chat_id,
                        message_id=record.telegram_message_id,
                        text=new_text,
                        parse_mode="HTML",
                    )
                    logger.info(
                        "telegram message updated for terminal resolution session_id=%s tool_use_id=%s approval_text=%s chat_id=%s message_id=%s",
                        session_id,
                        tool_use_id,
                        approval_text,
                        record.telegram_chat_id,
                        record.telegram_message_id,
                    )
                except Exception:
                    logger.warning(
                        "failed to edit Telegram message for terminal resolution",
                        extra={"session_id": session_id, "tool_use_id": tool_use_id},
                        exc_info=True,
                    )

    async def _bind_hook_session(self, event: HookEvent) -> None:
        if not event.session_id:
            return
        # Use per-session lock to prevent concurrent modifications to the same SessionState
        async with self._session_event_locks.lock(event.session_id):
            matched = await self._match_session_context(event)  # type: ignore[attr-defined]
            logger.info(
                "hook session match result",
                extra={
                    "hook_session_id": event.session_id,
                    "hook_event": event.event,
                    "hook_status": event.status,
                    "hook_cwd": event.cwd,
                    "matched_user_id": matched.user_id if matched is not None else None,
                    "matched_workdir": matched.workdir if matched is not None else None,
                    "matched_terminal_id": matched.terminal_id if matched is not None else None,
                    "matched_claude_session_id": matched.claude_session_id if matched is not None else None,
                },
            )
            if matched is None:
                return
            workdir = event.cwd or matched.workdir
            await self.task_service.bind_claude_session(
                user_id=matched.user_id,
                claude_session_id=event.session_id,
                workdir=workdir,
            )
            state = self.structured_session_store.get_or_create(
                session_id=event.session_id,
                provider="claude_code",
                workdir=workdir,
                terminal_id=matched.terminal_id,
                user_id=matched.user_id,
                claude_session_id=event.session_id,
            )
            state.terminal_id = matched.terminal_id
            state.user_id = matched.user_id
            state.workdir = workdir
            state.claude_session_id = event.session_id
            self.structured_session_store.save(state)


class SessionMatchingMixin(AppContainerBase):
    """Session matching: bind hook events to user sessions."""

    async def _match_session_context(self, event: HookEvent) -> SessionContext | None:
        # O(1) index lookup by claude_session_id (most common match path)
        indexed = await self.session_service.lookup_by_claude_session_id(event.session_id)
        if indexed is not None:
            logger.info(
                "matched hook session by claude_session_id (index)",
                extra={
                    "hook_session_id": event.session_id,
                    "user_id": indexed.user_id,
                    "workdir": indexed.workdir,
                    "terminal_id": indexed.terminal_id,
                },
            )
            return indexed

        # Index miss — fall back to full-scan matching logic
        sessions = await self.session_service.list_all()
        logger.info(
            "matching hook session context (fallback)",
            extra={
                "hook_session_id": event.session_id,
                "hook_cwd": event.cwd,
                "session_count": len(sessions),
            },
        )

        state = self.structured_session_store.get(event.session_id)
        if state is not None:
            for session in sessions:
                if session.user_id != state.user_id:
                    continue
                if session.terminal_id and state.terminal_id and session.terminal_id == state.terminal_id:
                    logger.info(
                        "matched hook session by terminal_id",
                        extra={
                            "hook_session_id": event.session_id,
                            "user_id": session.user_id,
                            "terminal_id": session.terminal_id,
                        },
                    )
                    return session

        event_workdir = str(Path(event.cwd).resolve()) if event.cwd else None
        eligible_sessions: list[SessionContext] = []
        workdir_matches: list[SessionContext] = []
        for session in sessions:
            session_workdir = str(Path(session.workdir).resolve()) if session.workdir else None
            logger.info(
                "evaluating hook session candidate",
                extra={
                    "hook_session_id": event.session_id,
                    "user_id": session.user_id,
                    "provider": session.provider,
                    "claude_chat_active": session.claude_chat_active,
                    "session_workdir": session.workdir,
                    "resolved_session_workdir": session_workdir,
                    "resolved_event_workdir": event_workdir,
                    "session_claude_session_id": session.claude_session_id,
                    "session_terminal_id": session.terminal_id,
                },
            )
            if not self.cli_factory.capabilities(session.provider).persistent_terminal or not session.claude_chat_active:
                continue
            eligible_sessions.append(session)
            if event_workdir and session_workdir == event_workdir:
                workdir_matches.append(session)

        if workdir_matches:
            active_task_matches: list[SessionContext] = []
            for session in workdir_matches:
                if await self._has_active_interactive_task(user_id=session.user_id, workdir=session.workdir):
                    active_task_matches.append(session)
            if len(active_task_matches) == 1:
                session = active_task_matches[0]
                logger.info(
                    "matched hook session by active interactive task",
                    extra={
                        "hook_session_id": event.session_id,
                        "user_id": session.user_id,
                        "terminal_id": session.terminal_id,
                        "resolved_event_workdir": event_workdir,
                    },
                )
                return session
            if len(active_task_matches) > 1:
                logger.warning(
                    "failed to match hook session context",
                    extra={
                        "hook_session_id": event.session_id,
                        "hook_cwd": event.cwd,
                        "reason": "ambiguous_active_interactive_task",
                        "candidate_user_ids": [session.user_id for session in active_task_matches],
                        "resolved_event_workdir": event_workdir,
                    },
                )
                return None

        if len(workdir_matches) == 1:
            session = workdir_matches[0]
            can_bind_chat, bind_reason, terminal_state = self._can_bind_unique_workdir_claude_chat(
                session=session,
                resolved_event_workdir=event_workdir,
            )
            if can_bind_chat:
                logger.info(
                    "matched hook session by active claude chat",
                    extra={
                        "hook_session_id": event.session_id,
                        "user_id": session.user_id,
                        "terminal_id": session.terminal_id,
                        "resolved_event_workdir": event_workdir,
                        "terminal_state_id": terminal_state.session_id if terminal_state is not None else None,
                        "terminal_state_phase": terminal_state.phase.value if terminal_state is not None else None,
                        "reason": bind_reason,
                    },
                )
                return session
            logger.warning(
                "failed to match hook session context",
                extra={
                    "hook_session_id": event.session_id,
                    "hook_cwd": event.cwd,
                    "reason": "workdir_only_match_blocked",
                    "candidate_user_ids": [session.user_id],
                    "resolved_event_workdir": event_workdir,
                    "terminal_id": session.terminal_id,
                    "has_active_interactive_task": False,
                    "claude_chat_bind_reason": bind_reason,
                    "terminal_state_id": terminal_state.session_id if terminal_state is not None else None,
                    "terminal_state_phase": terminal_state.phase.value if terminal_state is not None else None,
                },
            )
            return None

        if len(workdir_matches) > 1:
            logger.warning(
                "failed to match hook session context",
                extra={
                    "hook_session_id": event.session_id,
                    "hook_cwd": event.cwd,
                    "reason": "ambiguous_workdir",
                    "candidate_user_ids": [session.user_id for session in workdir_matches],
                },
            )
            return None

        logger.warning(
            "failed to match hook session context",
            extra={
                "hook_session_id": event.session_id,
                "hook_cwd": event.cwd,
                "reason": "no_match",
                "eligible_user_ids": [session.user_id for session in eligible_sessions],
            },
        )
        return None

    async def _has_active_interactive_task(self, *, user_id: int, workdir: str) -> bool:
        tasks = await self.task_store.iter_all()
        for task in tasks:
            if task.user_id != user_id:
                continue
            if not self.cli_factory.capabilities(task.provider).interactive_input:
                continue
            if task.workdir != workdir:
                continue
            if task.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELED}:
                continue
            return True
        return False

    def _can_bind_unique_workdir_claude_chat(
        self,
        *,
        session: SessionContext,
        resolved_event_workdir: str | None,
    ) -> tuple[bool, str, SessionState | None]:
        if not self.cli_factory.capabilities(session.provider).persistent_terminal or not session.claude_chat_active:
            return False, "inactive_claude_chat", None
        if not session.terminal_mode or not session.terminal_id:
            return False, "terminal_not_ready", None

        terminal_state = self.structured_session_store.find_by_terminal_id(session.terminal_id)
        if terminal_state is None:
            return True, "terminal_missing_state", None

        if terminal_state.user_id is not None and terminal_state.user_id != session.user_id:
            return False, "terminal_user_mismatch", terminal_state

        terminal_workdir = str(Path(terminal_state.workdir).resolve()) if terminal_state.workdir else None
        if resolved_event_workdir and terminal_workdir and terminal_workdir != resolved_event_workdir:
            return False, "terminal_workdir_mismatch", terminal_state

        has_content = bool(terminal_state.turns or terminal_state.tool_calls or terminal_state.pending_permission is not None)
        if has_content:
            return True, "terminal_has_content", terminal_state

        if terminal_state.phase in IDLE_PHASES:
            return True, "terminal_waiting", terminal_state

        if terminal_state.phase == SessionPhase.PROCESSING and terminal_state.session_id.startswith("tgcli_"):
            return True, "terminal_empty_processing_fallback", terminal_state

        return False, "terminal_empty_not_waiting", terminal_state


class WatcherMixin(AppContainerBase):
    """Session watcher management (unified interrupt + file + JSONL sync)."""

    def _start_session_watchers(self) -> None:
        """Start session supervisor watchers for all structured-session-capable sessions."""
        sessions = self.structured_session_store.values()
        for state in sessions:
            if not self.cli_factory.capabilities(state.provider).session_state:
                continue
            self.session_supervisor.watch(session_id=state.session_id, workdir=state.workdir)

    def _start_session_watchers_by_session_id(self, session_id: str, workdir: str) -> None:
        """Start session supervisor watcher for a specific session."""
        self.session_supervisor.watch(session_id=session_id, workdir=workdir)


class PeriodicRecheckMixin(AppContainerBase):
    """Periodic recheck of active Claude sessions."""

    async def _periodic_recheck_loop(self) -> None:
        interval_sec = self.settings.claude_periodic_recheck_ms / 1000
        try:
            while True:
                await asyncio.sleep(interval_sec)
                await self._recheck_active_claude_sessions()
        except asyncio.CancelledError:
            raise

    async def _recheck_active_claude_sessions(self) -> None:
        sessions = await self.session_service.list_all()
        for session in sessions:
            if not self.cli_factory.capabilities(session.provider).session_state or not session.claude_chat_active:
                continue
            if not session.claude_session_id:
                continue
            state = self.structured_session_store.get(session.claude_session_id)
            if state is None:
                continue
            if state.phase not in INTERRUPTIBLE_PHASES:
                continue
            logger.info(
                "periodic recheck syncing",
                extra={
                    "user_id": session.user_id,
                    "claude_session_id": session.claude_session_id,
                    "phase": state.phase.value,
                    "workdir": session.workdir,
                },
            )
            await self.sync_claude_session(session.claude_session_id, session.workdir)  # type: ignore[attr-defined]


class SessionRestoreMixin(AppContainerBase):
    """Session restoration on startup."""

    async def _restore_session_bindings(self) -> None:
        sessions = await self.session_service.list_all()
        for session in sessions:
            claude_session_id = session.claude_session_id
            if not claude_session_id:
                continue
            state = self.structured_session_store.get_or_create(
                session_id=claude_session_id,
                provider="claude_code",
                workdir=session.workdir,
                terminal_id=session.terminal_id,
                user_id=session.user_id,
                claude_session_id=claude_session_id,
            )
            session_file = self.claude_jsonl_parser.session_file_path(session_id=claude_session_id, cwd=session.workdir)
            if session_file.exists():
                await self.sync_claude_session(claude_session_id, session.workdir)  # type: ignore[attr-defined]
                self.session_supervisor.watch(session_id=state.session_id, workdir=state.workdir)
                continue
            if state.turns or state.tool_calls or state.pending_permission is not None:
                self.session_supervisor.watch(session_id=state.session_id, workdir=state.workdir)
                continue
            terminal_state = self.structured_session_store.find_by_terminal_id(session.terminal_id) if session.terminal_id else None
            if (
                terminal_state is not None
                and terminal_state.phase in INTERRUPTIBLE_PHASES
                and (terminal_state.turns or terminal_state.tool_calls or terminal_state.pending_permission is not None)
            ):
                self.session_supervisor.watch(session_id=terminal_state.session_id, workdir=terminal_state.workdir)
                continue
            # Clean up orphaned session state: clear binding and delete state files
            await self.session_service.clear_claude_session(user_id=session.user_id)
            self.structured_session_store.delete_session(claude_session_id)

    async def _restore_external_reply_delivery_pumps(self) -> None:
        for binding in self.external_binding_store.list_all():
            if not self.settings.external_push_reply_enabled and binding.ended_at is None:
                continue
            self.structured_session_store.get_or_create(
                session_id=binding.session_id,
                provider="claude_code",
                workdir=binding.cwd,
                user_id=binding.user_id,
                claude_session_id=binding.session_id,
            )
            self.session_supervisor.watch(session_id=binding.session_id, workdir=binding.cwd)
            self.external_reply_delivery_pump.ensure(session_id=binding.session_id, cwd=binding.cwd)
            self.session_supervisor.schedule_jsonl_sync(binding.session_id, binding.cwd)


class EventDispatchMixin(AppContainerBase):
    """Session event dispatch with per-session locking."""

    async def _dispatch_session_event(self, event: SessionEvent) -> None:
        is_session_end = self._is_session_end_dispatch_event(event)
        payload = event.payload_dict()
        cwd = payload.get("cwd")
        async with self._session_event_locks.lock(event.session_id):
            existing = self.structured_session_store.get(event.session_id)
            workdir = str(cwd) if cwd else (existing.workdir if existing and existing.workdir else ".")
            self.structured_session_store.get_or_create(
                session_id=event.session_id,
                provider="claude_code",
                workdir=workdir,
                claude_session_id=event.session_id,
            )
            self.structured_session_store.process(event)
        if is_session_end:
            await self._reconcile_session_context_after_session_end(event.session_id)
            await self._session_event_locks.cleanup_key(event.session_id, require_expired=False)

    @staticmethod
    def _is_session_end_dispatch_event(event: SessionEvent) -> bool:
        if event.type == SessionEventType.SESSION_ENDED:
            return True
        if event.type != SessionEventType.HOOK_RECEIVED:
            return False
        payload = event.payload_dict()
        return payload.get("event") == "SessionEnd" or payload.get("status") == "ended"

    async def _reconcile_session_context_after_session_end(self, session_id: str) -> None:
        session = await self.session_service.lookup_by_claude_session_id(session_id)
        if session is None or session.claude_session_id != session_id:
            return
        if session.terminal_id:
            terminal_id = session.terminal_id
            async with self.session_service.terminal_group_lock(terminal_id):
                current = await self.session_service.lookup_by_claude_session_id(session_id)
                if current is None or current.claude_session_id != session_id or current.terminal_id != terminal_id:
                    return
                if current.claude_chat_active:
                    logger.info(
                        "active terminal chat context retained after session end",
                        extra={"session_id": session_id, "terminal_id": terminal_id, "user_id": current.user_id},
                    )
                    return
                await self.session_service.clear_terminal_group(terminal_id)
            logger.info("session context cleared after session end", extra={"session_id": session_id, "terminal_id": terminal_id})
            return
        if not session.claude_chat_active:
            await self.session_service.clear_claude_session(user_id=session.user_id)
            return
        current, _ = await self.session_service.switch(
            user_id=session.user_id,
            terminal_mode=False,
            claude_chat_active=False,
        )
        current.claude_session_id = None
        await self.session_service.save_session_context(current)
        logger.info("terminal-less session context cleared after session end", extra={"session_id": session_id, "user_id": session.user_id})
