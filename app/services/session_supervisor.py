"""Unified per-session watcher combining interrupt detection, file sync, and JSONL sync.

Replaces three independent per-session polling tasks (InterruptWatcher,
AgentFileWatcher, debounced JSONL sync) with a single loop per session.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

from app.domain.session_models import (
    INTERRUPTIBLE_PHASES,
    InterruptDetectedPayload,
    SessionEvent,
    SessionEventType,
    SessionState,
    ToolStatus,
)
from app.infra.file_mtime_utils import clear_seen_mtimes, refresh_seen_mtimes

logger = logging.getLogger(__name__)

_MainJSONLSignature = tuple[str, int | None, int | None]


class _SessionStore(Protocol):
    def get(self, session_id: str) -> SessionState | None: ...


class _JSONLParser(Protocol):
    def parse_incremental(self, *, session_id: str, cwd: str) -> Any: ...

    def session_file_path(self, *, session_id: str, cwd: str) -> Path: ...

    def subagent_file_path(self, *, session_id: str, agent_id: str, cwd: str) -> Path: ...

    def reset_state(self, session_id: str) -> None: ...


class _JSONLFileWatcher(Protocol):
    @property
    def is_available(self) -> bool: ...

    def watch_files(self, *, session_id: str, cwd: str, paths: Iterable[Path]) -> None: ...

    def replace_session_files(self, *, session_id: str, cwd: str, paths: Iterable[Path]) -> None: ...

    def unwatch_session(self, session_id: str) -> None: ...

    def clear(self) -> None: ...


class SessionSupervisor:
    """Unified per-session watcher combining interrupt detection, file sync, and JSONL sync."""

    def __init__(
        self,
        *,
        session_store: _SessionStore,
        claude_jsonl_parser: _JSONLParser,
        on_jsonl_sync: Callable[[str, str], Awaitable[None]],
        on_dispatch_event: Callable[[SessionEvent], Awaitable[None]],
        poll_interval_sec: float = 0.2,
        idle_poll_interval_sec: float = 10.0,
        debounce_sec: float = 0.1,
        jsonl_file_watcher: _JSONLFileWatcher | None = None,
    ) -> None:
        self._session_store = session_store
        self._claude_jsonl_parser = claude_jsonl_parser
        self._on_jsonl_sync = on_jsonl_sync
        self._on_dispatch_event = on_dispatch_event
        self._poll_interval_sec = poll_interval_sec
        self._idle_poll_interval_sec = idle_poll_interval_sec
        self._debounce_sec = debounce_sec
        self._jsonl_file_watcher = jsonl_file_watcher
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._wake_events: dict[str, asyncio.Event] = {}
        self._jsonl_sync_requests: dict[str, tuple[str, float]] = {}  # session_id -> (cwd, requested_at)
        self._main_jsonl_signatures: dict[str, _MainJSONLSignature] = {}
        self._seen_mtimes: dict[str, float] = {}
        self._session_mtime_keys: dict[str, set[str]] = {}  # session_id -> tracked file paths
        self._active = False
        self._closing = False

    def reopen(self) -> None:
        self._closing = False

    def watch(self, *, session_id: str, workdir: str) -> None:
        """Start watching a session if not already watched."""
        if self._closing:
            return
        self._active = True
        self._wake_events.setdefault(session_id, asyncio.Event())
        self._watch_main_jsonl_file(session_id=session_id, workdir=workdir)
        task = self._tasks.get(session_id)
        if task is not None and not task.done():
            self._wake(session_id)
            return
        self._baseline_main_jsonl_signature(session_id=session_id, workdir=workdir)
        self._start_watch_task(session_id=session_id, workdir=workdir)

    def _start_watch_task(self, *, session_id: str, workdir: str) -> None:
        task = asyncio.create_task(self._watch_session(session_id=session_id, workdir=workdir))
        self._tasks[session_id] = task
        task.add_done_callback(lambda done: self._on_watch_done(done, session_id=session_id, workdir=workdir))

    def _on_watch_done(self, task: asyncio.Task[None], *, session_id: str, workdir: str) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        logger.exception(
            "session supervisor watcher crashed",
            extra={"session_id": session_id, "workdir": workdir},
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        current = self._tasks.get(session_id)
        if self._active and (current is None or current is task):
            self._start_watch_task(session_id=session_id, workdir=workdir)

    def schedule_jsonl_sync(self, session_id: str, cwd: str) -> None:
        """Request a debounced JSONL sync and wake the session watcher."""
        if self._closing:
            return
        existing = self._jsonl_sync_requests.get(session_id)
        if existing is not None:
            _, requested_at = existing
            self._jsonl_sync_requests[session_id] = (cwd, requested_at)
            return
        self._jsonl_sync_requests[session_id] = (cwd, asyncio.get_running_loop().time())
        self._wake(session_id)

    async def forget(self, session_id: str) -> None:
        """Stop watching a session and clean up state."""
        task = self._tasks.pop(session_id, None)
        await self._clear_session_mtimes(session_id)
        self._jsonl_sync_requests.pop(session_id, None)
        self._main_jsonl_signatures.pop(session_id, None)
        self._locks.pop(session_id, None)
        self._wake_events.pop(session_id, None)
        if self._jsonl_file_watcher is not None:
            self._jsonl_file_watcher.unwatch_session(session_id)
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def stop_all(self) -> None:
        """Cancel all watched sessions and wait for termination."""
        self._closing = True
        self._active = False
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        self._locks.clear()
        self._wake_events.clear()
        self._seen_mtimes.clear()
        self._session_mtime_keys.clear()
        self._jsonl_sync_requests.clear()
        self._main_jsonl_signatures.clear()
        if self._jsonl_file_watcher is not None:
            self._jsonl_file_watcher.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _watch_session(self, *, session_id: str, workdir: str) -> None:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        task = asyncio.current_task()
        try:
            while self._active:
                active_state = True
                main_jsonl_signature: _MainJSONLSignature | None = None
                subagent_files_changed = False
                try:
                    async with lock:
                        state = self._session_store.get(session_id)
                        if state is None:
                            return

                        active_state = self._is_active_state(state)
                        self._watch_jsonl_files_for_state(state)

                        # Interrupt detection (PROCESSING, WAITING_FOR_APPROVAL)
                        if state.phase in INTERRUPTIBLE_PHASES:
                            await self._maybe_detect_interrupt(state)

                        # Agent file sync (any phase with subagent containers)
                        if self._has_subagent_files(state):
                            if await self._sync_files_if_needed(state):
                                subagent_files_changed = True

                        if not self._is_file_watcher_available() and session_id not in self._jsonl_sync_requests:
                            main_jsonl_signature = self._changed_main_jsonl_signature(state)

                    # JSONL sync must stay outside the watcher lock because it dispatches events.
                    # Mark subagent files seen only after sync succeeds, otherwise a failed
                    # sync is treated as consumed and never retried.
                    if subagent_files_changed or main_jsonl_signature is not None:
                        await self._on_jsonl_sync(session_id, state.workdir)
                        if subagent_files_changed:
                            await self._refresh_seen_mtimes(state)
                        if main_jsonl_signature is not None:
                            self._main_jsonl_signatures[session_id] = main_jsonl_signature
                    await self._maybe_process_jsonl_sync(session_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("session supervisor tick failed", extra={"session_id": session_id, "workdir": workdir})

                await self._wait_for_next_tick(session_id=session_id, active_state=active_state)
        except asyncio.CancelledError:
            raise
        finally:
            if task is not None and self._tasks.get(session_id) is task:
                self._tasks.pop(session_id, None)
                await self._clear_session_mtimes(session_id)
                self._locks.pop(session_id, None)
                self._wake_events.pop(session_id, None)
                self._main_jsonl_signatures.pop(session_id, None)
                if self._jsonl_file_watcher is not None:
                    self._jsonl_file_watcher.unwatch_session(session_id)

    def _is_active_state(self, state: SessionState) -> bool:
        return state.phase in INTERRUPTIBLE_PHASES or self._has_active_subagent_files(state)

    def _wake(self, session_id: str) -> None:
        event = self._wake_events.get(session_id)
        if event is None:
            return
        event.set()

    async def _wait_for_next_tick(self, *, session_id: str, active_state: bool) -> None:
        timeout = self._next_wait_timeout(session_id=session_id, active_state=active_state)
        event = self._wake_events.setdefault(session_id, asyncio.Event())
        if event.is_set():
            event.clear()
            return
        if timeout <= 0:
            await asyncio.sleep(0)
            return
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except TimeoutError:
            pass
        finally:
            event.clear()

    def _next_wait_timeout(self, *, session_id: str, active_state: bool) -> float:
        entry = self._jsonl_sync_requests.get(session_id)
        if entry is not None:
            _, requested_at = entry
            elapsed = asyncio.get_running_loop().time() - requested_at
            return max(0.0, self._debounce_sec - elapsed)
        if active_state:
            return self._poll_interval_sec
        if self._jsonl_file_watcher is not None and self._jsonl_file_watcher.is_available:
            return self._idle_poll_interval_sec
        return self._poll_interval_sec

    # ── JSONL file watcher registration ───────────────────────────────────────

    def _is_file_watcher_available(self) -> bool:
        return self._jsonl_file_watcher is not None and self._jsonl_file_watcher.is_available

    def _baseline_main_jsonl_signature(self, *, session_id: str, workdir: str) -> None:
        signature = self._main_jsonl_signature(session_id=session_id, workdir=workdir)
        previous = self._main_jsonl_signatures.get(session_id)
        if previous is None or previous[0] != signature[0]:
            self._main_jsonl_signatures[session_id] = signature

    def _changed_main_jsonl_signature(self, state: SessionState) -> _MainJSONLSignature | None:
        signature = self._main_jsonl_signature(
            session_id=state.claude_session_id or state.session_id,
            workdir=state.workdir,
        )
        if self._main_jsonl_signatures.get(state.session_id) == signature:
            return None
        return signature

    def _main_jsonl_signature(self, *, session_id: str, workdir: str) -> _MainJSONLSignature:
        try:
            path = self._claude_jsonl_parser.session_file_path(session_id=session_id, cwd=workdir)
        except ValueError:
            return ("", None, None)
        try:
            stat = path.stat()
        except OSError:
            return (str(path), None, None)
        return (str(path), stat.st_mtime_ns, stat.st_size)

    def _watch_main_jsonl_file(self, *, session_id: str, workdir: str) -> None:
        watcher = self._jsonl_file_watcher
        if watcher is None:
            return
        try:
            path = self._claude_jsonl_parser.session_file_path(session_id=session_id, cwd=workdir)
        except ValueError:
            logger.debug("skip jsonl file watcher registration for invalid session id", extra={"session_id": session_id})
            return
        watcher.watch_files(session_id=session_id, cwd=workdir, paths=(path,))

    def _watch_jsonl_files_for_state(self, state: SessionState) -> None:
        watcher = self._jsonl_file_watcher
        if watcher is None:
            return
        claude_session_id = state.claude_session_id or state.session_id
        try:
            paths = set(self._jsonl_file_paths_for_state(state, claude_session_id=claude_session_id))
        except ValueError:
            logger.debug("skip jsonl file watcher registration for invalid session state", extra={"session_id": state.session_id})
            return
        watcher.replace_session_files(session_id=state.session_id, cwd=state.workdir, paths=paths)

    def _jsonl_file_paths_for_state(self, state: SessionState, *, claude_session_id: str) -> set[Path]:
        session_file = self._claude_jsonl_parser.session_file_path(session_id=claude_session_id, cwd=state.workdir)
        paths = {session_file}
        for tool in state.tool_calls.values():
            if not tool.is_subagent_container:
                continue
            result = tool.structured_result or {}
            agent_id = str(result.get("agentId") or "")
            if not agent_id:
                continue
            agent_file = self._claude_jsonl_parser.subagent_file_path(
                session_id=claude_session_id,
                agent_id=agent_id,
                cwd=state.workdir,
            )
            paths.add(agent_file)
            paths.add(session_file.parent / claude_session_id / "subagents" / agent_file.name)
            paths.add(session_file.parent / agent_file.name)
        return paths

    # ── Interrupt detection ───────────────────────────────────────────────────

    async def _maybe_detect_interrupt(self, state: SessionState) -> None:
        if state.interrupted:
            return
        claude_session_id = state.claude_session_id or state.session_id
        if not claude_session_id:
            return
        snapshot = self._claude_jsonl_parser.parse_incremental(session_id=claude_session_id, cwd=state.workdir)
        if not snapshot.interrupt_detected:
            return
        await self._on_dispatch_event(
            SessionEvent(
                session_id=claude_session_id,
                type=SessionEventType.INTERRUPT_DETECTED,
                payload=InterruptDetectedPayload.from_mapping(snapshot.to_payload()),
            )
        )

    # ── Agent file sync ───────────────────────────────────────────────────────

    def _has_subagent_files(self, state: SessionState) -> bool:
        return any(tool.is_subagent_container for tool in state.tool_calls.values())

    def _has_active_subagent_files(self, state: SessionState) -> bool:
        active_statuses = {ToolStatus.RUNNING, ToolStatus.WAITING_FOR_APPROVAL}
        return any(tool.is_subagent_container and tool.status in active_statuses for tool in state.tool_calls.values())

    async def _clear_session_mtimes(self, session_id: str) -> None:
        """Remove all tracked mtime entries for a session using the key index."""
        keys = self._session_mtime_keys.pop(session_id, set())
        await clear_seen_mtimes(keys, self._seen_mtimes)

    async def _sync_files_if_needed(self, state: SessionState) -> bool:
        changed = False
        for tool in state.tool_calls.values():
            if not tool.is_subagent_container:
                continue
            result = tool.structured_result or {}
            agent_id = str(result.get("agentId") or "")
            if not agent_id:
                continue
            agent_file = self._claude_jsonl_parser.subagent_file_path(
                session_id=state.claude_session_id or state.session_id,
                agent_id=agent_id,
                cwd=state.workdir,
            )
            if not agent_file.exists():
                continue
            file_path = str(agent_file)
            mtime = agent_file.stat().st_mtime
            previous = self._seen_mtimes.get(file_path)
            if previous is not None and mtime <= previous:
                continue
            changed = True
            break
        if changed:
            self._claude_jsonl_parser.reset_state(state.claude_session_id or state.session_id)
        return changed

    async def _refresh_seen_mtimes(self, state: SessionState) -> None:
        session_key = state.session_id
        # Build path set from current tool calls
        paths: set[str] = set()
        for tool in state.tool_calls.values():
            if not tool.is_subagent_container:
                continue
            result = tool.structured_result or {}
            agent_id = str(result.get("agentId") or "")
            if not agent_id:
                continue
            agent_file = self._claude_jsonl_parser.subagent_file_path(
                session_id=state.claude_session_id or state.session_id,
                agent_id=agent_id,
                cwd=state.workdir,
            )
            if not agent_file.exists():
                continue
            paths.add(str(agent_file))
        # Clear stale keys via tracked index and refresh current ones
        await self._clear_session_mtimes(session_key)
        await refresh_seen_mtimes(paths, self._seen_mtimes)
        self._session_mtime_keys[session_key] = set(paths)

    # ── Debounced JSONL sync ──────────────────────────────────────────────────

    async def _maybe_process_jsonl_sync(self, session_id: str) -> None:
        entry = self._jsonl_sync_requests.get(session_id)
        if entry is None:
            return
        cwd, requested_at = entry
        now = asyncio.get_running_loop().time()
        if now - requested_at < self._debounce_sec:
            return
        self._jsonl_sync_requests.pop(session_id, None)
        try:
            await self._on_jsonl_sync(session_id, cwd)
        except Exception:
            logger.exception("jsonl sync failed", extra={"session_id": session_id})
