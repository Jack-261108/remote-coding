from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from app.config.settings import is_workdir_allowed
from app.domain.hook_models import HookEvent, HookResponse, PendingPermissionRequest
from app.domain.models import utc_now

HookEventHandler = Callable[[HookEvent], Awaitable[None] | None]
PermissionFailureHandler = Callable[[str, str], Awaitable[None] | None]
PermissionResolvedHandler = Callable[[str, str, str], Awaitable[None] | None]  # session_id, tool_use_id, reason
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _CachedToolUseId:
    exact_key: str
    tool_use_id: str
    tool_input: dict[str, Any] | None
    cached_at: datetime


def _summarize_tool_input(tool_input: dict[str, Any] | None) -> dict[str, Any] | None:
    if tool_input is None:
        return None
    summary: dict[str, Any] = {"key_count": len(tool_input)}
    try:
        summary["approx_bytes"] = len(json.dumps(tool_input, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError):
        pass
    return summary


@dataclass(slots=True)
class _DisconnectedPermission:
    session_id: str
    tool_use_id: str
    event: HookEvent
    received_at: datetime
    disconnected_at: datetime


class HookSocketServer:
    def __init__(
        self,
        socket_path: str,
        *,
        allowed_workdirs: Sequence[str] | None = None,
        max_message_bytes: int = 1_048_576,
        pending_permission_ttl_sec: int = 600,
        max_pending_permissions: int = 64,
        max_tool_use_id_cache_entries: int = 256,
        permission_disconnect_grace_sec: float = 0.75,
    ) -> None:
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes 必须大于 0")
        if pending_permission_ttl_sec <= 0:
            raise ValueError("pending_permission_ttl_sec 必须大于 0")
        if max_pending_permissions <= 0:
            raise ValueError("max_pending_permissions 必须大于 0")
        if max_tool_use_id_cache_entries <= 0:
            raise ValueError("max_tool_use_id_cache_entries 必须大于 0")
        if permission_disconnect_grace_sec <= 0:
            raise ValueError("permission_disconnect_grace_sec 必须大于 0")
        self._socket_path = Path(socket_path)
        self._allowed_workdirs = list(allowed_workdirs) if allowed_workdirs is not None else None
        self._max_message_bytes = max_message_bytes
        self._pending_permission_ttl_sec = pending_permission_ttl_sec
        self._max_pending_permissions = max_pending_permissions
        self._max_tool_use_id_cache_entries = max_tool_use_id_cache_entries
        self._permission_disconnect_grace_sec = permission_disconnect_grace_sec
        self._server: asyncio.AbstractServer | None = None
        self._event_handler: HookEventHandler | None = None
        self._permission_failure_handler: PermissionFailureHandler | None = None
        self._permission_resolved_handler: PermissionResolvedHandler | None = None
        self._pending_permissions: dict[str, PendingPermissionRequest] = {}
        self._pending_expiration_tasks: dict[str, asyncio.Task[None]] = {}
        self._pending_disconnect_tasks: dict[str, asyncio.Task[None]] = {}
        self._disconnected_permissions: dict[str, _DisconnectedPermission] = {}
        self._disconnect_grace_tasks: dict[str, asyncio.Task[None]] = {}
        self._tool_use_id_cache: dict[str, list[_CachedToolUseId]] = {}
        self._client_tasks: set[asyncio.Task[None]] = set()
        self._stopping = False
        self._ingress_ready = asyncio.Event()
        self._ingress_ready.set()
        self._lock = asyncio.Lock()

    def pause_ingress(self) -> None:
        self._ingress_ready.clear()

    def resume_ingress(self) -> None:
        self._ingress_ready.set()

    async def start(
        self,
        on_event: HookEventHandler,
        on_permission_failure: PermissionFailureHandler | None = None,
        on_permission_resolved: PermissionResolvedHandler | None = None,
    ) -> None:
        if self._server is not None:
            return
        self._stopping = False
        self._event_handler = on_event
        self._permission_failure_handler = on_permission_failure
        self._permission_resolved_handler = on_permission_resolved
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(FileNotFoundError):
            self._socket_path.unlink()
        previous_umask = os.umask(0o177)
        try:
            self._server = await asyncio.start_unix_server(self._accept_client, path=str(self._socket_path))
        finally:
            os.umask(previous_umask)
        self._socket_path.chmod(0o600)

    async def stop(self) -> None:
        self._stopping = True
        server = self._server
        self._server = None
        if server is not None:
            server.close()
        # 先取消并等待 client handlers 结束。Python 3.12+ 的 Server.wait_closed()
        # 会等待活跃连接退出；若 handler 卡在 paused ingress 或等待 EOF 读上，需先取消。
        client_tasks = list(self._client_tasks)
        for task in client_tasks:
            task.cancel()
        await asyncio.gather(*client_tasks, return_exceptions=True)
        self._client_tasks.clear()
        # 取消 pending permissions 相关的定时/断线任务，并在 wait_closed 之前关闭 writer。
        # 在 Python 3.12+ 中，wait_closed() 会等待所有连接断开；keep-open 的权限连接若在
        # wait_closed 之后才关闭，会导致停机死锁。
        async with self._lock:
            pending = list(self._pending_permissions.values())
            expiration_tasks = list(self._pending_expiration_tasks.values())
            disconnect_tasks = list(self._pending_disconnect_tasks.values())
            grace_tasks = list(self._disconnect_grace_tasks.values())
            self._pending_permissions.clear()
            self._pending_expiration_tasks.clear()
            self._pending_disconnect_tasks.clear()
            self._disconnected_permissions.clear()
            self._disconnect_grace_tasks.clear()
            self._tool_use_id_cache.clear()
        for task in [*expiration_tasks, *disconnect_tasks, *grace_tasks]:
            task.cancel()
        for task in [*expiration_tasks, *disconnect_tasks, *grace_tasks]:
            with suppress(asyncio.CancelledError):
                await task
        await self._close_pending_permissions(pending, emit_failure=False)
        if server is not None:
            await server.wait_closed()
        with suppress(FileNotFoundError):
            self._socket_path.unlink()

    async def respond_to_permission(self, *, tool_use_id: str, decision: str, reason: str | None = None) -> bool:
        expired: list[PendingPermissionRequest]
        async with self._lock:
            expired = self._pop_expired_pending_permissions_locked()
            pending = self._pending_permissions.pop(tool_use_id, None)
            if pending is not None:
                self._cancel_pending_expiration_locked(tool_use_id)
            self._cancel_pending_disconnect_watch_locked(tool_use_id)
        await self._expire_pending_permissions(expired)
        if pending is None:
            return False
        return await self._write_response(pending=pending, decision=decision, reason=reason)

    async def respond_to_permission_outcome(
        self, *, tool_use_id: str, decision: str, reason: str | None = None
    ) -> Literal["wrote", "not_pending", "write_failed"]:
        """Like ``respond_to_permission`` but distinguishes WHY it returned False.

        External question approval needs to tell apart two False cases that the
        boolean cannot:

        * ``"not_pending"`` — no pending connection exists for ``tool_use_id``
          (PostToolUse already closed it after reading the TUI answer). The answer
          reached Claude; this is the race-tolerant success the external flow must
          keep treating as approved.
        * ``"write_failed"`` — the connection existed but the response write/drain
          raised. Claude is still blocked on this ``allow``; the external flow must
          NOT claim success and must converge to ``INDETERMINATE``.

        ``"wrote"`` means the allow was written; identical to the True branch of
        ``respond_to_permission``.
        """
        expired: list[PendingPermissionRequest]
        async with self._lock:
            expired = self._pop_expired_pending_permissions_locked()
            pending = self._pending_permissions.pop(tool_use_id, None)
            if pending is not None:
                self._cancel_pending_expiration_locked(tool_use_id)
                self._cancel_pending_disconnect_watch_locked(tool_use_id)
        await self._expire_pending_permissions(expired)
        if pending is None:
            return "not_pending"
        return "wrote" if await self._write_response(pending=pending, decision=decision, reason=reason) else "write_failed"

    async def respond_to_permission_by_session(self, *, session_id: str, decision: str, reason: str | None = None) -> bool:
        expired: list[PendingPermissionRequest]
        async with self._lock:
            expired = self._pop_expired_pending_permissions_locked()
            candidates = [item for item in self._pending_permissions.values() if item.session_id == session_id]
            pending = max(candidates, key=lambda item: item.received_at, default=None)
            if pending is not None:
                self._pending_permissions.pop(pending.tool_use_id, None)
                self._cancel_pending_expiration_locked(pending.tool_use_id)
                self._cancel_pending_disconnect_watch_locked(pending.tool_use_id)
        await self._expire_pending_permissions(expired)
        if pending is None:
            return False
        return await self._write_response(pending=pending, decision=decision, reason=reason)

    async def release_pending_permission(self, *, tool_use_id: str) -> bool:
        async with self._lock:
            pending = self._pending_permissions.pop(tool_use_id, None)
            disconnected = self._disconnected_permissions.pop(tool_use_id, None)
            if pending is not None:
                self._cancel_pending_expiration_locked(tool_use_id)
                self._cancel_pending_disconnect_watch_locked(tool_use_id)
            if disconnected is not None:
                self._cancel_disconnect_grace_locked(tool_use_id)
        if pending is not None:
            await self._close_writer(pending.writer)
            return True
        return disconnected is not None

    async def cancel_pending_permissions(self, *, session_id: str) -> None:
        async with self._lock:
            expired = self._pop_expired_pending_permissions_locked()
            matching = [item for item in self._pending_permissions.values() if item.session_id == session_id]
            disconnected_ids = [
                tool_use_id for tool_use_id, item in self._disconnected_permissions.items() if item.session_id == session_id
            ]
            for item in matching:
                self._pending_permissions.pop(item.tool_use_id, None)
                self._cancel_pending_expiration_locked(item.tool_use_id)
                self._cancel_pending_disconnect_watch_locked(item.tool_use_id)
            for tool_use_id in disconnected_ids:
                self._disconnected_permissions.pop(tool_use_id, None)
                self._cancel_disconnect_grace_locked(tool_use_id)
        await self._expire_pending_permissions(expired)
        await self._close_pending_permissions(matching, emit_failure=False)

    async def has_pending_permission(self, *, session_id: str) -> bool:
        async with self._lock:
            expired = self._pop_expired_pending_permissions_locked()
            result = any(item.session_id == session_id for item in self._pending_permissions.values())
        await self._expire_pending_permissions(expired)
        return result

    async def get_pending_permission(self, *, session_id: str) -> tuple[str | None, str | None, dict[str, Any] | None] | None:
        async with self._lock:
            expired = self._pop_expired_pending_permissions_locked()
            found = None
            for item in self._pending_permissions.values():
                if item.session_id == session_id:
                    found = (item.event.tool, item.tool_use_id, item.event.tool_input)
                    break
        await self._expire_pending_permissions(expired)
        return found

    async def get_session_id_for_tool_use_id(self, tool_use_id: str) -> str | None:
        """Look up the session_id for a pending permission by tool_use_id."""
        async with self._lock:
            pending = self._pending_permissions.get(tool_use_id)
            return pending.session_id if pending is not None else None

    async def _read_hook_payload(self, reader: asyncio.StreamReader) -> tuple[bytes, bool]:
        try:
            raw = await reader.readuntil(b"\n")
            return raw.rstrip(b"\n"), True
        except asyncio.IncompleteReadError as exc:
            return exc.partial, False
        except asyncio.LimitOverrunError:
            raw = await reader.read(self._max_message_bytes + 1)
            return raw, False

    def _accept_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._stopping:
            writer.close()
            return
        task = asyncio.create_task(self._run_client(reader, writer))
        self._client_tasks.add(task)
        task.add_done_callback(self._client_tasks.discard)

    async def _run_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await self._handle_client(reader, writer)
        except asyncio.CancelledError:
            await self._close_writer(writer)
            raise
        except Exception:
            logger.exception("hook socket client handler failed")
            await self._close_writer(writer)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        raw, is_framed = await self._read_hook_payload(reader)
        raw_size = len(raw)
        if not raw:
            await self._close_writer(writer)
            return
        if raw_size > self._max_message_bytes:
            logger.warning(
                "hook message rejected: too large",
                extra={"max_message_bytes": self._max_message_bytes, "raw_size": raw_size, "is_framed": is_framed},
            )
            await self._close_writer(writer)
            return

        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            logger.warning(
                "hook payload rejected",
                extra={"reason": "utf8_decode_failed", "error_type": type(exc).__name__, "raw_size": raw_size, "is_framed": is_framed},
            )
            await self._close_writer(writer)
            return

        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError as exc:
            logger.warning(
                "hook payload rejected",
                extra={"reason": "json_decode_failed", "error_type": type(exc).__name__, "raw_size": raw_size, "is_framed": is_framed},
            )
            await self._close_writer(writer)
            return
        if not isinstance(payload, dict):
            logger.warning(
                "hook payload rejected",
                extra={"reason": "non_object_payload", "error_type": type(payload).__name__, "raw_size": raw_size, "is_framed": is_framed},
            )
            await self._close_writer(writer)
            return
        try:
            event = HookEvent.from_dict(payload)
        except (KeyError, ValueError) as exc:
            logger.warning(
                "hook payload rejected",
                extra={
                    "reason": "validation_failed",
                    "error_type": type(exc).__name__,
                    "raw_size": raw_size,
                    "is_framed": is_framed,
                    "payload_key_count": len(payload),
                },
            )
            await self._close_writer(writer)
            return

        if not self._is_event_workdir_allowed(event):
            logger.warning(
                "hook event rejected by workdir allowlist",
                extra={"session_id": event.session_id, "cwd": event.cwd, "event": event.event},
            )
            await self._close_writer(writer)
            return

        await self._ingress_ready.wait()
        if self._stopping:
            await self._close_writer(writer)
            return

        logger.info(
            "hook event received",
            extra={
                "session_id": event.session_id,
                "event": event.event,
                "status": event.status,
                "tool": event.tool,
                "tool_use_id": event.tool_use_id,
                "expects_response": event.expects_response,
                "tool_input_summary": _summarize_tool_input(event.tool_input),
            },
        )

        if event.event == "PreToolUse" and event.tool_use_id:
            await self._cache_tool_use_id(event)
        if event.event in {"PostToolUse", "PostToolUseFailure", "PermissionDenied"}:
            await self._remove_cached_tool_use_id(event)
            if event.event in {"PostToolUse", "PermissionDenied"}:
                # Check if this tool had a pending permission (terminal-side resolution).
                # PostToolUseFailure is not a permission decision: the user may have
                # allowed the tool and the tool itself failed afterwards.
                decision = "terminal_approved" if event.event == "PostToolUse" else "terminal_denied"
                tool_use_id = event.tool_use_id
                if not await self._has_resolvable_permission(tool_use_id):
                    tool_use_id = await self._find_terminal_permission_tool_use_id(
                        event,
                        allow_latest_fallback=decision == "terminal_denied",
                    )
                if tool_use_id:
                    logger.info(
                        "terminal permission resolution detected",
                        extra={
                            "session_id": event.session_id,
                            "event": event.event,
                            "decision": decision,
                            "resolved_tool_use_id": tool_use_id,
                            "original_tool_use_id": event.tool_use_id,
                            "tool": event.tool,
                            "tool_input_summary": _summarize_tool_input(event.tool_input),
                        },
                    )
                    await self._check_terminal_permission_resolved(event.session_id, tool_use_id, decision)
        if event.event == "SessionEnd":
            await self._cleanup_cache(event.session_id)
            await self.cancel_pending_permissions(session_id=event.session_id)

        keep_open = False
        emit_event = event
        if event.expects_response:
            tool_use_id = event.tool_use_id or await self._pop_cached_tool_use_id(event)
            if tool_use_id:
                emit_event = event.with_tool_use_id(tool_use_id)
            else:
                fallback_tool_use_id = f"hookperm-{uuid.uuid4().hex}"
                emit_event = event.with_tool_use_id(fallback_tool_use_id)
                logger.warning(
                    "permission request missing tool_use_id match; using synthetic id",
                    extra={
                        "session_id": event.session_id,
                        "tool": event.tool,
                        "tool_input_summary": _summarize_tool_input(event.tool_input),
                        "synthetic_tool_use_id": fallback_tool_use_id,
                    },
                )

            async with self._lock:
                expired = self._pop_expired_pending_permissions_locked()
                if len(self._pending_permissions) >= self._max_pending_permissions:
                    over_limit = True
                    previous = None
                else:
                    over_limit = False
                    tool_id = emit_event.tool_use_id or ""
                    previous = self._pending_permissions.pop(tool_id, None)
                    if previous is not None:
                        self._cancel_pending_expiration_locked(tool_id)
                        self._cancel_pending_disconnect_watch_locked(tool_id)
                    self._disconnected_permissions.pop(tool_id, None)
                    self._cancel_disconnect_grace_locked(tool_id)
                    self._pending_permissions[tool_id] = PendingPermissionRequest(
                        session_id=emit_event.session_id,
                        tool_use_id=tool_id,
                        writer=writer,
                        event=emit_event,
                    )
                    self._schedule_pending_expiration_locked(tool_id)
                    if is_framed:
                        self._schedule_pending_disconnect_watch_locked(tool_id, reader)
            await self._expire_pending_permissions(expired)
            if over_limit:
                logger.warning(
                    "permission request rejected: pending limit reached",
                    extra={"session_id": emit_event.session_id, "max_pending_permissions": self._max_pending_permissions},
                )
                await self._write_response(
                    pending=PendingPermissionRequest(
                        session_id=emit_event.session_id,
                        tool_use_id=emit_event.tool_use_id or "",
                        writer=writer,
                        event=emit_event,
                    ),
                    decision="deny",
                    reason="pending permission limit reached",
                )
                return
            if previous is not None:
                await self._expire_pending_permissions([previous], reason="permission request superseded")
            keep_open = True

        if not keep_open:
            await self._close_writer(writer)

        await self._emit_event(emit_event)

    async def _write_response(self, *, pending: PendingPermissionRequest, decision: str, reason: str | None) -> bool:
        response = HookResponse(decision=decision, reason=reason)
        data = json.dumps(response.to_dict(), ensure_ascii=False).encode("utf-8")
        success = True
        try:
            pending.writer.write(data)
            await pending.writer.drain()
        except Exception:
            success = False
            await self._emit_permission_failure(pending.session_id, pending.tool_use_id)
        finally:
            await self._close_writer(pending.writer)
        return success

    async def _invoke_handler_safely(
        self,
        *,
        kind: str,
        handler: Callable[..., Awaitable[None] | None] | None,
        args: tuple[object, ...],
        extra: dict[str, object],
    ) -> None:
        if handler is None:
            return
        try:
            result = handler(*args)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("hook socket callback failed", extra={"callback": kind, **extra})

    async def _emit_event(self, event: HookEvent) -> None:
        await self._invoke_handler_safely(
            kind="event",
            handler=self._event_handler,
            args=(event,),
            extra={"session_id": event.session_id, "event": event.event, "tool_use_id": event.tool_use_id or ""},
        )

    async def _emit_permission_failure(self, session_id: str, tool_use_id: str) -> None:
        await self._invoke_handler_safely(
            kind="permission_failure",
            handler=self._permission_failure_handler,
            args=(session_id, tool_use_id),
            extra={"session_id": session_id, "tool_use_id": tool_use_id},
        )

    async def _emit_permission_resolved(self, session_id: str, tool_use_id: str, reason: str) -> None:
        await self._invoke_handler_safely(
            kind="permission_resolved",
            handler=self._permission_resolved_handler,
            args=(session_id, tool_use_id, reason),
            extra={"session_id": session_id, "tool_use_id": tool_use_id, "reason": reason},
        )

    async def _has_resolvable_permission(self, tool_use_id: str | None) -> bool:
        if tool_use_id is None:
            return False
        async with self._lock:
            return tool_use_id in self._pending_permissions or tool_use_id in self._disconnected_permissions

    async def _check_terminal_permission_resolved(self, session_id: str, tool_use_id: str, decision: str = "terminal_approved") -> None:
        """Check if a terminal-side event indicates permission resolution."""
        async with self._lock:
            pending = self._pending_permissions.pop(tool_use_id, None)
            disconnected = None
            if pending is not None:
                self._cancel_pending_expiration_locked(tool_use_id)
                self._cancel_pending_disconnect_watch_locked(tool_use_id)
            else:
                disconnected = self._disconnected_permissions.pop(tool_use_id, None)
                if disconnected is not None:
                    self._cancel_disconnect_grace_locked(tool_use_id)
            if pending is None and disconnected is None:
                pending_summaries = [
                    {
                        "session_id": item.session_id,
                        "tool_use_id": item.tool_use_id,
                        "tool": item.event.tool,
                        "tool_input_summary": _summarize_tool_input(item.event.tool_input),
                    }
                    for item in self._pending_permissions.values()
                ]
                disconnected_summaries = [
                    {
                        "session_id": item.session_id,
                        "tool_use_id": item.tool_use_id,
                        "tool": item.event.tool,
                        "tool_input": item.event.tool_input,
                    }
                    for item in self._disconnected_permissions.values()
                ]
                logger.debug(
                    "no pending permission found for terminal resolution session_id=%s tool_use_id=%s decision=%s pending=%s disconnected=%s",
                    session_id,
                    tool_use_id,
                    decision,
                    pending_summaries,
                    disconnected_summaries,
                )
                return
        if pending is not None:
            await self._close_writer(pending.writer)
            emit_session_id = pending.session_id
        else:
            assert disconnected is not None
            emit_session_id = disconnected.session_id
        logger.info(
            "terminal permission resolution: emitting resolved event session_id=%s tool_use_id=%s decision=%s",
            emit_session_id,
            tool_use_id,
            decision,
        )
        await self._emit_permission_resolved(emit_session_id, tool_use_id, decision)

    async def _close_pending_permissions(self, items: list[PendingPermissionRequest], *, emit_failure: bool) -> None:
        for item in items:
            await self._close_writer(item.writer)
            if emit_failure:
                await self._emit_permission_failure(item.session_id, item.tool_use_id)

    async def _expire_pending_permissions(
        self, items: list[PendingPermissionRequest], *, reason: str = "permission request expired"
    ) -> None:
        for item in items:
            success = await self._write_response(pending=item, decision="deny", reason=reason)
            if success:
                await self._emit_permission_failure(item.session_id, item.tool_use_id)

    async def _close_writer(self, writer: asyncio.StreamWriter) -> None:
        writer.close()
        with suppress(Exception):
            await writer.wait_closed()

    def _pop_expired_pending_permissions_locked(self) -> list[PendingPermissionRequest]:
        now = utc_now()
        expired = [
            item
            for item in self._pending_permissions.values()
            if (now - item.received_at).total_seconds() >= self._pending_permission_ttl_sec
        ]
        for item in expired:
            self._pending_permissions.pop(item.tool_use_id, None)
            self._cancel_pending_expiration_locked(item.tool_use_id)
            self._cancel_pending_disconnect_watch_locked(item.tool_use_id)
        return expired

    def _schedule_pending_expiration_locked(self, tool_use_id: str) -> None:
        self._cancel_pending_expiration_locked(tool_use_id)
        self._pending_expiration_tasks[tool_use_id] = asyncio.create_task(self._expire_pending_permission_later(tool_use_id))

    def _schedule_pending_disconnect_watch_locked(self, tool_use_id: str, reader: asyncio.StreamReader) -> None:
        self._cancel_pending_disconnect_watch_locked(tool_use_id)
        self._pending_disconnect_tasks[tool_use_id] = asyncio.create_task(self._watch_pending_permission_disconnect(tool_use_id, reader))

    def _cancel_pending_expiration_locked(self, tool_use_id: str) -> None:
        task = self._pending_expiration_tasks.pop(tool_use_id, None)
        if task is not None:
            task.cancel()

    def _cancel_pending_disconnect_watch_locked(self, tool_use_id: str) -> None:
        task = self._pending_disconnect_tasks.pop(tool_use_id, None)
        if task is not None:
            task.cancel()

    def _schedule_disconnect_grace_locked(self, tool_use_id: str) -> None:
        self._cancel_disconnect_grace_locked(tool_use_id)
        self._disconnect_grace_tasks[tool_use_id] = asyncio.create_task(self._resolve_disconnected_permission_later(tool_use_id))

    def _cancel_disconnect_grace_locked(self, tool_use_id: str) -> None:
        task = self._disconnect_grace_tasks.pop(tool_use_id, None)
        if task is not None:
            task.cancel()

    async def _expire_pending_permission_later(self, tool_use_id: str) -> None:
        try:
            await asyncio.sleep(self._pending_permission_ttl_sec)
            async with self._lock:
                pending = self._pending_permissions.pop(tool_use_id, None)
                self._pending_expiration_tasks.pop(tool_use_id, None)
                self._cancel_pending_disconnect_watch_locked(tool_use_id)
            if pending is not None:
                await self._expire_pending_permissions([pending])
        except asyncio.CancelledError:
            raise

    async def _resolve_disconnected_permission_later(self, tool_use_id: str) -> None:
        try:
            await asyncio.sleep(self._permission_disconnect_grace_sec)
            async with self._lock:
                disconnected = self._disconnected_permissions.pop(tool_use_id, None)
                self._disconnect_grace_tasks.pop(tool_use_id, None)
            if disconnected is None:
                return
            logger.info(
                "permission disconnect grace expired, marking denied session_id=%s tool_use_id=%s",
                disconnected.session_id,
                disconnected.tool_use_id,
            )
            await self._emit_permission_resolved(disconnected.session_id, disconnected.tool_use_id, "terminal_denied")
        except asyncio.CancelledError:
            raise

    async def _watch_pending_permission_disconnect(self, tool_use_id: str, reader: asyncio.StreamReader) -> None:
        try:
            await reader.read()
            async with self._lock:
                pending = self._pending_permissions.pop(tool_use_id, None)
                self._pending_disconnect_tasks.pop(tool_use_id, None)
                if pending is not None:
                    self._cancel_pending_expiration_locked(tool_use_id)
                    self._disconnected_permissions[tool_use_id] = _DisconnectedPermission(
                        session_id=pending.session_id,
                        tool_use_id=pending.tool_use_id,
                        event=pending.event,
                        received_at=pending.received_at,
                        disconnected_at=utc_now(),
                    )
                    self._schedule_disconnect_grace_locked(tool_use_id)
            if pending is None:
                return
            logger.info(
                "pending permission request connection closed before response session_id=%s tool_use_id=%s",
                pending.session_id,
                pending.tool_use_id,
            )
            await self._close_writer(pending.writer)
        except asyncio.CancelledError:
            raise

    def _is_event_workdir_allowed(self, event: HookEvent) -> bool:
        if self._allowed_workdirs is None:
            return True
        return is_workdir_allowed(event.cwd, self._allowed_workdirs)

    async def _cache_tool_use_id(self, event: HookEvent) -> None:
        key = self._session_tool_cache_key(event)
        exact_key = self._exact_cache_key(event)
        async with self._lock:
            self._prune_tool_use_id_cache_locked()
            self._tool_use_id_cache.setdefault(key, []).append(
                _CachedToolUseId(
                    exact_key=exact_key,
                    tool_use_id=event.tool_use_id or "",
                    tool_input=event.tool_input,
                    cached_at=utc_now(),
                )
            )
            self._trim_tool_use_id_cache_locked()

    async def _remove_cached_tool_use_id(self, event: HookEvent) -> None:
        key = self._session_tool_cache_key(event)
        exact_key = self._exact_cache_key(event)
        async with self._lock:
            self._prune_tool_use_id_cache_locked()
            queue = self._tool_use_id_cache.get(key)
            if not queue:
                return
            if event.tool_use_id:
                self._tool_use_id_cache[key] = [cached for cached in queue if cached.tool_use_id != event.tool_use_id]
            else:
                self._tool_use_id_cache[key] = [cached for cached in queue if cached.exact_key != exact_key]
            if not self._tool_use_id_cache[key]:
                self._tool_use_id_cache.pop(key, None)

    async def _pop_cached_tool_use_id(self, event: HookEvent) -> str | None:
        key = self._session_tool_cache_key(event)
        exact_key = self._exact_cache_key(event)
        async with self._lock:
            self._prune_tool_use_id_cache_locked()
            queue = self._tool_use_id_cache.get(key)
            if not queue:
                return None
            exact_index = next((index for index in range(len(queue) - 1, -1, -1) if queue[index].exact_key == exact_key), None)
            if exact_index is not None:
                cached = queue.pop(exact_index)
                if not queue:
                    self._tool_use_id_cache.pop(key, None)
                return cached.tool_use_id or None

            relaxed_index = self._find_relaxed_cache_match_index(queue, event)
            if relaxed_index is None:
                return None
            cached = queue.pop(relaxed_index)
            if not queue:
                self._tool_use_id_cache.pop(key, None)
            logger.info(
                "permission request matched cached tool_use_id via relaxed tool-input comparison",
                extra={
                    "session_id": event.session_id,
                    "tool": event.tool,
                    "tool_use_id": cached.tool_use_id,
                    "tool_input_summary": _summarize_tool_input(event.tool_input),
                },
            )
            return cached.tool_use_id or None

    async def _find_terminal_permission_tool_use_id(self, event: HookEvent, *, allow_latest_fallback: bool) -> str | None:
        async with self._lock:
            candidates: list[PendingPermissionRequest | _DisconnectedPermission] = [
                item for item in self._pending_permissions.values() if item.session_id == event.session_id
            ]
            candidates.extend(item for item in self._disconnected_permissions.values() if item.session_id == event.session_id)
            if not candidates:
                return None
            tool_matches = [item for item in candidates if item.event.tool == event.tool]
            input_candidates = tool_matches or candidates
            input_matches = [item for item in input_candidates if self._tool_inputs_relaxed_match(event.tool_input, item.event.tool_input)]
            if input_matches:
                latest = max(input_matches, key=lambda item: item.received_at)
                return latest.tool_use_id
            if allow_latest_fallback:
                latest_candidates = tool_matches or candidates
                latest = max(latest_candidates, key=lambda item: item.received_at)
                return latest.tool_use_id
            return None

    async def _cleanup_cache(self, session_id: str) -> None:
        async with self._lock:
            keys = [key for key in self._tool_use_id_cache if key.startswith(f"{session_id}:")]
            for key in keys:
                self._tool_use_id_cache.pop(key, None)

    def _prune_tool_use_id_cache_locked(self) -> None:
        now = utc_now()
        for key, queue in list(self._tool_use_id_cache.items()):
            fresh = [cached for cached in queue if (now - cached.cached_at).total_seconds() < self._pending_permission_ttl_sec]
            if fresh:
                self._tool_use_id_cache[key] = fresh
            else:
                self._tool_use_id_cache.pop(key, None)

    def _trim_tool_use_id_cache_locked(self) -> None:
        overflow = sum(len(queue) for queue in self._tool_use_id_cache.values()) - self._max_tool_use_id_cache_entries
        while overflow > 0:
            oldest_key = min(
                (key for key, queue in self._tool_use_id_cache.items() if queue),
                key=lambda key: self._tool_use_id_cache[key][0].cached_at,
                default=None,
            )
            if oldest_key is None:
                return
            self._tool_use_id_cache[oldest_key].pop(0)
            if not self._tool_use_id_cache[oldest_key]:
                self._tool_use_id_cache.pop(oldest_key, None)
            overflow -= 1

    def _session_tool_cache_key(self, event: HookEvent) -> str:
        return f"{event.session_id}:{event.tool or 'unknown'}"

    def _exact_cache_key(self, event: HookEvent) -> str:
        tool_input = event.tool_input or {}
        serialized = json.dumps(tool_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"{self._session_tool_cache_key(event)}:{serialized}"

    def _find_relaxed_cache_match_index(self, queue: list[_CachedToolUseId], event: HookEvent) -> int | None:
        if not queue:
            return None
        for index in range(len(queue) - 1, -1, -1):
            if self._tool_inputs_relaxed_match(event.tool_input, queue[index].tool_input):
                return index
        return None

    def _tool_inputs_relaxed_match(
        self,
        requested: dict[str, Any] | None,
        cached: dict[str, Any] | None,
    ) -> bool:
        if requested == cached:
            return True
        if not requested or not cached:
            return False

        for key in ("command", "description", "url", "query", "file_path", "path"):
            requested_value = requested.get(key)
            cached_value = cached.get(key)
            if requested_value is None or cached_value is None:
                continue
            if str(requested_value).strip() == str(cached_value).strip():
                return True

        requested_items = {
            key: value for key, value in requested.items() if isinstance(value, (str, int, float, bool)) and value is not None
        }
        cached_items = {key: value for key, value in cached.items() if isinstance(value, (str, int, float, bool)) and value is not None}
        if requested_items and all(cached_items.get(key) == value for key, value in requested_items.items()):
            return True
        if cached_items and all(requested_items.get(key) == value for key, value in cached_items.items()):
            return True
        return False
