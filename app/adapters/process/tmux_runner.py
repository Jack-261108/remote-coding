from __future__ import annotations

import asyncio
import logging
import shlex
import shutil
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.adapters.process.tmux_commands import TmuxCommandMixin
from app.adapters.process.tmux_log import AsyncFifoReader, TmuxLogMixin
from app.adapters.process.tmux_session import TmuxSessionCheckError, TmuxSessionMixin, is_recoverable_tmux_session_error
from app.adapters.storage.file_session_store import FileSessionStore
from app.domain.models import CLIEvent, EventType, utc_now
from app.domain.protocols import SessionStoreProtocol
from app.domain.session_models import (
    ConversationTurn,
    SessionEvent,
    SessionEventType,
    SessionPhase,
    SessionState,
    ToolStatus,
    is_claude_session_id,
)
from app.infra.lock_registry import RefCountedLockRegistry

CCB_BEGIN_PREFIX = "TGCLI_BEGIN"
CCB_DONE_PREFIX = "TGCLI_DONE"
_TERMINAL_REVEAL_SCRIPT = """
on run argv
    tell application "Terminal"
        activate
        set targetTab to do script ""
        delay 0.2
        repeat with attempt from 1 to 100
            if not busy of targetTab then exit repeat
            delay 0.1
        end repeat
        if busy of targetTab then error "Terminal shell did not become ready"
        do script (item 1 of argv) in targetTab
    end tell
end run
""".strip()
logger = logging.getLogger(__name__)


@dataclass
class _TmuxTaskMeta:
    session_name: str
    log_file: Path
    exit_file: Path
    task_id: str
    workdir: str
    terminal_id: str | None = None
    claude_session_id: str | None = None
    command_file: Path | None = None
    persistent_terminal: bool = False
    cancel_requested: bool = False
    interactive: bool = False
    prompt_text: str = ""
    baseline_captured: bool = False
    baseline_offset: int = 0
    baseline_completed_turn_id: str | None = None
    command_started_at: datetime | None = None


@dataclass
class _InteractiveWatchState:
    """Mutable state for interactive completion detection across ticks."""

    watch_started_at: datetime
    completion_started_after: datetime
    latest_completed_turn_id_before_run: str | None = None
    saw_interactive_progress: bool = False
    structured_offset_before_run: int = 0
    last_interactive_revision: int | None = None
    completion_candidate_key: tuple[object, ...] | None = None
    completion_candidate_seen_at: float | None = None
    last_progress_at: float | None = None


class TmuxRunner(TmuxSessionMixin, TmuxCommandMixin, TmuxLogMixin):
    def __init__(
        self,
        *,
        tmux_bin: str = "tmux",
        data_dir: str = "/tmp/tg-cli-gateway",
        poll_interval_sec: float = 0.2,
        cancel_grace_sec: float = 1.0,
        enter_delay_sec: float = 0.2,
        partial_flush_sec: float = 0.5,
        interactive_completion_grace_sec: float = 0.1,
        exit_file_read_grace_sec: float = 1.0,
        claude_cli_bin: str = "claude",
        file_store: FileSessionStore | None = None,
        session_store: SessionStoreProtocol | None = None,
        session_lock_ttl_sec: int = 3600,
        lock_cleanup_interval_sec: int = 60,
        lock_cleanup_batch_size: int = 50,
        lock_clock: Callable[[], float] | None = None,
    ) -> None:
        self._tmux_bin = tmux_bin
        self._data_dir = Path(data_dir)
        self._poll_interval_sec = poll_interval_sec
        self._cancel_grace_sec = cancel_grace_sec
        self._enter_delay_sec = max(0.0, enter_delay_sec)
        self._partial_flush_sec = max(0.0, partial_flush_sec)
        self._interactive_completion_grace_sec = max(0.0, interactive_completion_grace_sec)
        self._exit_file_read_grace_sec = max(0.0, exit_file_read_grace_sec)
        self._interactive_idle_check_sec = 5.0
        self._claude_cli_bin = claude_cli_bin
        self._tasks: dict[str, _TmuxTaskMeta] = {}
        self._session_locks = RefCountedLockRegistry(
            ttl_sec=session_lock_ttl_sec,
            cleanup_interval_sec=lock_cleanup_interval_sec,
            cleanup_batch_size=lock_cleanup_batch_size,
            clock=lock_clock,
        )
        self._pane_io_locks = RefCountedLockRegistry(
            ttl_sec=session_lock_ttl_sec,
            cleanup_interval_sec=lock_cleanup_interval_sec,
            cleanup_batch_size=lock_cleanup_batch_size,
            clock=lock_clock,
        )
        self._lock = asyncio.Lock()
        self._file_store = file_store or FileSessionStore(str(self._data_dir))
        self._session_store_impl = session_store

    @property
    def _session_store(self) -> SessionStoreProtocol:
        return self._require_session_store()

    def _require_session_store(self) -> SessionStoreProtocol:
        if self._session_store_impl is None:
            raise RuntimeError("session_store is required for TmuxRunner session tracking")
        return self._session_store_impl

    async def _register_task(self, meta: _TmuxTaskMeta) -> None:
        async with self._lock:
            self._tasks[meta.task_id] = meta

    async def _unregister_task(self, task_id: str) -> None:
        async with self._lock:
            self._tasks.pop(task_id, None)

    def _tmux_log_extra(self, meta: _TmuxTaskMeta, **extra) -> dict[str, object]:
        payload: dict[str, object] = {
            "task_id": meta.task_id,
            "session_name": meta.session_name,
            "terminal_id": meta.terminal_id,
            "claude_session_id": meta.claude_session_id,
            "workdir": meta.workdir,
            "persistent_terminal": meta.persistent_terminal,
            "interactive": meta.interactive,
        }
        payload.update(extra)
        return payload

    def _structured_state_log_extra(self, state: SessionState | None) -> dict[str, object]:
        if state is None:
            return {}
        tools = list(state.tool_calls.values())
        return {
            "active_session_id": state.session_id,
            "active_phase": state.phase.value,
            "active_revision": state.revision,
            "active_offset": state.checkpoint.last_offset,
            "turn_count": len(state.turns),
            "tool_count": len(tools),
            "running_tool_count": sum(1 for tool in tools if tool.status == ToolStatus.RUNNING),
            "waiting_tool_count": sum(1 for tool in tools if tool.status == ToolStatus.WAITING_FOR_APPROVAL),
            "interrupted_tool_count": sum(1 for tool in tools if tool.status == ToolStatus.INTERRUPTED),
        }

    async def run(
        self,
        *,
        task_id: str,
        argv: list[str],
        workdir: str,
        timeout_sec: int,
        env: dict[str, str] | None = None,
        terminal_key: str | None = None,
        interactive: bool = False,
        claude_session_id: str | None = None,
    ) -> AsyncGenerator[CLIEvent, None]:
        if not argv:
            yield CLIEvent(type=EventType.FAILED, task_id=task_id, error="命令参数为空")
            return

        self._data_dir.mkdir(parents=True, exist_ok=True)

        session_id = terminal_key or task_id
        session_name = self.build_session_name(session_id)
        log_file = self._file_store.raw_transcript_path(session_name) if interactive else self._data_dir / f"{task_id}.log"
        exit_file = self._data_dir / f"{task_id}.exit"
        command_file = self._data_dir / f"{task_id}.cmd.sh"
        persistent_terminal = terminal_key is not None

        self._safe_unlink(exit_file)
        self._safe_unlink(command_file)
        if not interactive:
            self._safe_unlink(log_file)

        try:
            if interactive:
                if not persistent_terminal:
                    yield CLIEvent(type=EventType.FAILED, task_id=task_id, error="交互式模式仅支持持久终端")
                    return
                if len(argv) != 1:
                    yield CLIEvent(type=EventType.FAILED, task_id=task_id, error="交互式模式参数错误")
                    return
                prompt = argv[0]
                command = self._wrap_interactive_prompt(prompt=prompt)
            else:
                command = self._build_shell_command(
                    argv=argv,
                    workdir=workdir,
                    log_file=log_file,
                    exit_file=exit_file,
                    command_file=command_file,
                    hide_launcher_line=persistent_terminal,
                )
        except Exception as exc:
            yield CLIEvent(type=EventType.FAILED, task_id=task_id, error=f"任务脚本创建失败: {exc}")
            return

        meta = _TmuxTaskMeta(
            session_name=session_name,
            log_file=log_file,
            exit_file=exit_file,
            task_id=task_id,
            workdir=workdir,
            terminal_id=session_id,
            claude_session_id=claude_session_id,
            command_file=command_file,
            persistent_terminal=persistent_terminal,
            interactive=interactive,
            prompt_text=(argv[0].strip() if interactive and argv else ""),
        )

        session_store = self._require_session_store()
        session_store.get_or_create(
            session_id=session_name,
            provider="claude_code",
            workdir=workdir,
            terminal_id=session_id,
        )

        if persistent_terminal:
            await self._register_task(meta)
            try:
                async with self._session_lock(session_name):
                    async for event in self._run_task(meta=meta, timeout_sec=timeout_sec, env=env, workdir=workdir, command=command):
                        yield event
            finally:
                await self._unregister_task(meta.task_id)
            return

        async for event in self._run_task(meta=meta, timeout_sec=timeout_sec, env=env, workdir=workdir, command=command):
            yield event

    async def _run_task(self, *, meta: _TmuxTaskMeta, timeout_sec: int, env: dict[str, str] | None, workdir: str, command: str):
        session_started = False
        watch_completed = False
        fifo_reader: AsyncFifoReader | None = None
        if meta.persistent_terminal:
            if meta.cancel_requested:
                yield CLIEvent(type=EventType.CANCELED, task_id=meta.task_id, error="任务已取消")
                return
            if meta.interactive:
                async with self._pane_io_lock(meta.session_name):
                    if meta.cancel_requested:
                        yield CLIEvent(type=EventType.CANCELED, task_id=meta.task_id, error="任务已取消")
                        return
                    ready, err = await self._ensure_claude_interactive_session(session_name=meta.session_name, workdir=workdir, env=env)
                    if not ready:
                        yield CLIEvent(type=EventType.FAILED, task_id=meta.task_id, error=err)
                        return
                    if meta.cancel_requested:
                        yield CLIEvent(type=EventType.CANCELED, task_id=meta.task_id, error="任务已取消")
                        return
                    fifo_reader = AsyncFifoReader(self._fifo_path(meta.session_name))
                    await fifo_reader.start()
                    pipe_ready, pipe_err = await self._bind_interactive_pipe(
                        meta=meta,
                        workdir=workdir,
                        env=env,
                        pipe_cmd=fifo_reader.pipe_command(),
                    )
                    if not pipe_ready:
                        await fifo_reader.close()
                        yield CLIEvent(type=EventType.FAILED, task_id=meta.task_id, error=pipe_err)
                        return
                    self._capture_interactive_baseline(meta=meta)
                    if meta.cancel_requested:
                        await self._clear_interactive_pipe(meta.session_name)
                        await fifo_reader.close()
                        yield CLIEvent(type=EventType.CANCELED, task_id=meta.task_id, error="任务已取消")
                        return
                    meta.command_started_at = utc_now()
                    sent, send_err = await self._send_command(meta.session_name, command, workdir=workdir, env=env, interactive=True)
            else:
                ready, err = await self._ensure_persistent_session(meta.session_name, workdir=workdir, env=env)
                if not ready:
                    yield CLIEvent(type=EventType.FAILED, task_id=meta.task_id, error=err)
                    return
                if meta.cancel_requested:
                    yield CLIEvent(type=EventType.CANCELED, task_id=meta.task_id, error="任务已取消")
                    return
                meta.command_started_at = utc_now()
                sent, send_err = await self._send_command(meta.session_name, command, workdir=workdir, env=env, interactive=False)
            if not sent:
                if fifo_reader is not None:
                    await fifo_reader.close()
                yield CLIEvent(type=EventType.FAILED, task_id=meta.task_id, error=send_err)
                return
        else:
            started, err = await self._start_ephemeral_session(meta.session_name, workdir=workdir, env=env, command=command)
            if not started:
                yield CLIEvent(type=EventType.FAILED, task_id=meta.task_id, error=err)
                return
            session_started = True

        await self._register_task(meta)

        if meta.interactive:
            state = self._session_store.mark_interactive_turn_processing(
                terminal_id=meta.terminal_id,
                workdir=meta.workdir,
                claude_session_id=meta.claude_session_id,
                fallback_session_id=meta.session_name,
            )
            if state is not None and is_claude_session_id(state.session_id):
                meta.claude_session_id = state.session_id
        else:
            self._session_store.process(SessionEvent(session_id=meta.session_name, type=SessionEventType.SESSION_STARTED))

        try:
            yield CLIEvent(type=EventType.STARTED, task_id=meta.task_id, content=f"tmux_session={meta.session_name}")
            async for event in self._watch_task(meta=meta, timeout_sec=timeout_sec, fifo_reader=fifo_reader):
                yield event
            watch_completed = True
        finally:
            if fifo_reader is not None:
                await self._clear_interactive_pipe(meta.session_name)
                await fifo_reader.close()
            if session_started and not meta.persistent_terminal and not watch_completed:
                terminated = await self._terminate_session(meta.session_name)
                if not terminated:
                    logger.warning(
                        "ephemeral tmux session cleanup failed",
                        extra={"task_id": meta.task_id, "session_name": meta.session_name},
                    )
            async with self._lock:
                self._tasks.pop(meta.task_id, None)
            if meta.command_file is not None:
                self._safe_unlink(meta.command_file)

    async def _bind_interactive_pipe(
        self, *, meta: _TmuxTaskMeta, workdir: str, env: dict[str, str] | None, pipe_cmd: str | None = None
    ) -> tuple[bool, str]:
        await self._clear_interactive_pipe(meta.session_name)
        ok, err = await self._set_interactive_pipe(meta, pipe_cmd=pipe_cmd)
        if ok:
            return True, ""
        if not is_recoverable_tmux_session_error(err):
            return False, f"tmux 管道设置失败: {err}"

        rebuilt, rebuild_err = await self._ensure_claude_interactive_session(
            session_name=meta.session_name,
            workdir=workdir,
            env=env,
        )
        if not rebuilt:
            return False, "\n".join(
                [
                    f"tmux 管道设置失败: {err}",
                    f"tmux_session: {meta.session_name}",
                    "auto_rebuilt: 否",
                    f"rebuild_error: {rebuild_err}",
                    "hint: tmux 会话丢失且自动重建失败，请发送 /claude 重新建立会话后再试",
                ]
            )

        await self._clear_interactive_pipe(meta.session_name)
        ok, retry_err = await self._set_interactive_pipe(meta, pipe_cmd=pipe_cmd)
        if ok:
            return True, ""
        return False, "\n".join(
            [
                f"tmux 管道设置失败: {retry_err}",
                f"tmux_session: {meta.session_name}",
                "auto_rebuilt: 是",
                "hint: 自动重建已执行但仍失败，请发送 /claude 重新建立会话后再试",
            ]
        )

    async def _clear_interactive_pipe(self, session_name: str) -> None:
        try:
            await self._run_tmux("pipe-pane", "-t", session_name)
        except Exception:
            pass

    def _fifo_path(self, session_name: str) -> Path:
        return self._data_dir / f"{session_name}.fifo"

    async def _set_interactive_pipe(self, meta: _TmuxTaskMeta, *, pipe_cmd: str | None = None) -> tuple[bool, str]:
        if pipe_cmd is None:
            pipe_cmd = f"cat >> {shlex.quote(str(meta.log_file))}"
        try:
            code, _, err_text = await self._run_tmux("pipe-pane", "-t", meta.session_name, pipe_cmd)
        except FileNotFoundError:
            return False, f"找不到 tmux 可执行文件 ({self._tmux_bin})"
        except Exception as exc:
            return False, f"tmux pipe-pane 异常: {exc}"
        if code == 0:
            return True, ""
        return False, err_text.strip() or "unknown error"

    def _read_exit_code_with_grace(
        self,
        *,
        meta: _TmuxTaskMeta,
        now: float,
        unreadable_since: float | None,
    ) -> tuple[int | None, float | None, bool]:
        if not meta.exit_file.exists():
            return None, None, False

        exit_code = self._read_exit_code(meta.exit_file)
        if exit_code is not None:
            return exit_code, None, True

        if unreadable_since is None:
            unreadable_since = now

        extra = self._tmux_log_extra(meta, exit_file=str(meta.exit_file))
        if (now - unreadable_since) >= self._exit_file_read_grace_sec:
            logger.warning("tmux exit file unreadable", extra=extra)
            return None, unreadable_since, True

        logger.debug("tmux exit file not readable yet", extra=extra)
        return None, unreadable_since, False

    async def _watch_task(self, *, meta: _TmuxTaskMeta, timeout_sec: int, fifo_reader: AsyncFifoReader | None = None):
        if meta.interactive and fifo_reader is not None:
            async for event in self._watch_interactive_fifo(meta=meta, timeout_sec=timeout_sec, fifo_reader=fifo_reader):
                yield event
            return

        watch_started_at = utc_now()
        watch_state = _InteractiveWatchState(
            watch_started_at=watch_started_at,
            completion_started_after=meta.command_started_at or watch_started_at,
            latest_completed_turn_id_before_run=meta.baseline_completed_turn_id,
            structured_offset_before_run=meta.baseline_offset,
        )
        position = 0
        partial = ""
        timed_out = False
        exit_code: int | None = None
        started_at = asyncio.get_running_loop().time()
        timeout_anchor = started_at
        last_partial_emit = started_at
        exit_code_unreadable_since: float | None = None

        if meta.interactive:
            exit_code = self._init_interactive_watch(meta=meta, watch_state=watch_state, now=started_at)
            position = self._interactive_log_position(meta.log_file)
            if watch_state.latest_completed_turn_id_before_run is None and not meta.baseline_captured:
                watch_state.latest_completed_turn_id_before_run = self._session_store.latest_completed_assistant_turn_id(
                    terminal_id=meta.terminal_id,
                    workdir=meta.workdir,
                    claude_session_id=meta.claude_session_id,
                    fallback_session_id=meta.session_name,
                )

        while exit_code is None:
            now = asyncio.get_running_loop().time()
            text, new_position = self._read_new_text(meta.log_file, position)
            if text:
                position = new_position
                if meta.interactive:
                    timeout_anchor = now
                    self._process_interactive_chunk(meta=meta, offset=position)
                else:
                    partial, events = self._split_to_events(task_id=meta.task_id, text=partial + text)
                    for event in events:
                        yield event

            if partial and not meta.interactive and self._partial_flush_sec > 0 and (now - last_partial_emit) >= self._partial_flush_sec:
                yield CLIEvent(type=EventType.STDOUT, task_id=meta.task_id, content=partial)
                partial = ""
                last_partial_emit = now

            if meta.interactive:
                exit_code, timeout_anchor = await self._handle_interactive_tick(
                    meta=meta,
                    watch_state=watch_state,
                    started_at=started_at,
                    timeout_anchor=timeout_anchor,
                    now=now,
                )
                if exit_code is not None:
                    break

            exit_code, exit_code_unreadable_since, exit_file_done = self._read_exit_code_with_grace(
                meta=meta,
                now=now,
                unreadable_since=exit_code_unreadable_since,
            )
            if exit_file_done:
                break
            exit_file_pending = exit_code_unreadable_since is not None

            timeout_started_at = timeout_anchor if meta.interactive else started_at
            if not exit_file_pending and (now - timeout_started_at) >= timeout_sec:
                timed_out = True
                await self._handle_watch_timeout(
                    meta=meta,
                    timeout_sec=timeout_sec,
                    started_at=started_at,
                    now=now,
                    timeout_started_at=timeout_started_at,
                    position=position,
                    watch_state=watch_state,
                )
                break

            if await self._is_cancel_requested(meta.task_id):
                await self._handle_watch_cancel(meta=meta, started_at=started_at, now=now, position=position)
                break

            await asyncio.sleep(self._poll_interval_sec)

        # Drain remaining text
        drain_events, partial = self._drain_final_text(meta=meta, position=position, partial=partial)
        for event in drain_events:
            yield event

        # Determine and yield final result
        canceled = meta.cancel_requested or await self._is_cancel_requested(meta.task_id)
        if not meta.interactive and self._session_store.get(meta.session_name) is not None:
            self._session_store.process(SessionEvent(session_id=meta.session_name, type=SessionEventType.SESSION_ENDED))

        finished_at = asyncio.get_running_loop().time()
        yield self._finalize_watch_result(
            meta=meta,
            timed_out=timed_out,
            canceled=canceled,
            exit_code=exit_code,
            timeout_sec=timeout_sec,
            started_at=started_at,
            finished_at=finished_at,
            position=position,
        )

    async def _watch_interactive_fifo(self, *, meta: _TmuxTaskMeta, timeout_sec: int, fifo_reader: AsyncFifoReader):
        watch_started_at = utc_now()
        watch_state = _InteractiveWatchState(
            watch_started_at=watch_started_at,
            completion_started_after=meta.command_started_at or watch_started_at,
            latest_completed_turn_id_before_run=meta.baseline_completed_turn_id,
            structured_offset_before_run=meta.baseline_offset,
        )
        timed_out = False
        exit_code: int | None = None
        started_at = asyncio.get_running_loop().time()
        timeout_anchor = started_at
        fifo_offset = 0
        exit_code_unreadable_since: float | None = None

        exit_code = self._init_interactive_watch(meta=meta, watch_state=watch_state, now=started_at)
        if watch_state.latest_completed_turn_id_before_run is None and not meta.baseline_captured:
            watch_state.latest_completed_turn_id_before_run = self._session_store.latest_completed_assistant_turn_id(
                terminal_id=meta.terminal_id,
                workdir=meta.workdir,
                claude_session_id=meta.claude_session_id,
                fallback_session_id=meta.session_name,
            )

        stdout = await fifo_reader.readlines()

        while exit_code is None:
            now = asyncio.get_running_loop().time()
            remaining = max(0.1, timeout_sec - (now - timeout_anchor))
            eof_seen = False
            try:
                chunk = await asyncio.wait_for(stdout.read(65536), timeout=min(1.0, remaining))
            except TimeoutError:
                chunk = None
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("fifo reader error", extra={"session": meta.session_name})
                break

            if chunk:
                fifo_offset += len(chunk)
                timeout_anchor = asyncio.get_running_loop().time()
                self._process_interactive_chunk(meta=meta, offset=fifo_offset)
            elif chunk == b"":
                eof_seen = True
                if fifo_offset > 0:
                    logger.debug("fifo reader EOF", extra={"session": meta.session_name})
                elif hasattr(fifo_reader, "_process") and fifo_reader._process is not None:
                    if fifo_reader._process.returncode is not None:
                        logger.debug("fifo reader process exited with no data", extra={"session": meta.session_name})

            now = asyncio.get_running_loop().time()
            exit_code, exit_code_unreadable_since, exit_file_done = self._read_exit_code_with_grace(
                meta=meta,
                now=now,
                unreadable_since=exit_code_unreadable_since,
            )
            if exit_file_done:
                break
            exit_file_pending = exit_code_unreadable_since is not None

            exit_code, timeout_anchor = await self._handle_interactive_tick(
                meta=meta,
                watch_state=watch_state,
                started_at=started_at,
                timeout_anchor=timeout_anchor,
                now=now,
            )
            if exit_code is not None or (eof_seen and not exit_file_pending):
                break

            if not exit_file_pending and (now - timeout_anchor) >= timeout_sec:
                timed_out = True
                await self._handle_watch_timeout(
                    meta=meta,
                    timeout_sec=timeout_sec,
                    started_at=started_at,
                    now=now,
                    timeout_started_at=timeout_anchor,
                    position=fifo_offset,
                    watch_state=watch_state,
                )
                break

            if await self._is_cancel_requested(meta.task_id):
                await self._handle_watch_cancel(meta=meta, started_at=started_at, now=now, position=fifo_offset)
                break

            if eof_seen:
                await asyncio.sleep(self._poll_interval_sec)

        canceled = meta.cancel_requested or await self._is_cancel_requested(meta.task_id)
        finished_at = asyncio.get_running_loop().time()
        yield self._finalize_watch_result(
            meta=meta,
            timed_out=timed_out,
            canceled=canceled,
            exit_code=exit_code,
            timeout_sec=timeout_sec,
            started_at=started_at,
            finished_at=finished_at,
            position=fifo_offset,
        )

    async def _handle_interactive_tick(
        self,
        *,
        meta: _TmuxTaskMeta,
        watch_state: _InteractiveWatchState,
        started_at: float,
        timeout_anchor: float,
        now: float,
    ) -> tuple[int | None, float]:
        """Process one interactive watch tick. Returns (exit_code, updated_timeout_anchor)."""
        tick_result, active_state = self._tick_interactive_watch(meta=meta, watch_state=watch_state, now=now)
        if active_state is not None and active_state.interrupted:
            meta.cancel_requested = True
            return 0, timeout_anchor
        if tick_result is not None:
            return tick_result, timeout_anchor
        if active_state is not None:
            if watch_state.last_interactive_revision is None:
                watch_state.last_interactive_revision = active_state.revision
            elif active_state.revision != watch_state.last_interactive_revision:
                watch_state.last_interactive_revision = active_state.revision
                watch_state.last_progress_at = now
                timeout_anchor = now
        # Pane idle only means Claude is back at an input prompt. A normal
        # completion has the same shape while JSONL/Stop-hook sync may still be
        # catching up, so this check is observational only and must not mark the
        # task as canceled.
        idle_anchor = watch_state.last_progress_at or started_at
        if (now - started_at) >= self._interactive_idle_check_sec and (now - idle_anchor) >= self._interactive_idle_check_sec:
            if await self._is_claude_idle_in_pane(meta.session_name):
                logger.debug("interactive pane idle observed; waiting for structured completion", extra=self._tmux_log_extra(meta))
            watch_state.last_progress_at = now
        return None, timeout_anchor

    async def _handle_watch_timeout(
        self,
        *,
        meta: _TmuxTaskMeta,
        timeout_sec: int,
        started_at: float,
        now: float,
        timeout_started_at: float,
        position: int,
        watch_state: _InteractiveWatchState,
    ) -> None:
        """Handle timeout: log, interrupt/terminate session."""
        action = "interrupt" if meta.persistent_terminal else "terminate"
        logger.warning(
            "tmux task timeout",
            extra=self._tmux_log_extra(
                meta,
                timeout_sec=timeout_sec,
                elapsed_sec=round(now - started_at, 3),
                idle_sec=round(now - timeout_started_at, 3),
                action=action,
                log_position=position,
                baseline_captured=meta.baseline_captured,
                baseline_offset=meta.baseline_offset,
                last_interactive_revision=watch_state.last_interactive_revision,
                saw_interactive_progress=watch_state.saw_interactive_progress,
                **self._structured_state_log_extra(None),
            ),
        )
        if meta.persistent_terminal:
            await self._interrupt_locked(meta.session_name)
        else:
            await self._terminate_session(meta.session_name)

    async def _handle_watch_cancel(
        self,
        *,
        meta: _TmuxTaskMeta,
        started_at: float,
        now: float,
        position: int,
    ) -> None:
        """Handle cancel detection: log, interrupt/terminate session."""
        action = "interrupt" if meta.persistent_terminal else "terminate"
        logger.info(
            "tmux task cancel detected",
            extra=self._tmux_log_extra(
                meta,
                action=action,
                log_position=position,
                elapsed_sec=round(now - started_at, 3),
            ),
        )
        if meta.persistent_terminal:
            await self._interrupt_locked(meta.session_name)
        else:
            await self._terminate_session(meta.session_name)

    async def _interrupt_locked(self, session_name: str) -> bool:
        """Serialize C-c with pane paste/select. Watch-loop cancel and idle timeout must use this too."""
        async with self._pane_io_lock(session_name):
            return await self._interrupt_session(session_name)

    def _drain_final_text(
        self,
        *,
        meta: _TmuxTaskMeta,
        position: int,
        partial: str,
    ) -> tuple[list[CLIEvent], str]:
        """Read and process any remaining text after the main watch loop.

        Returns (events_to_yield, remaining_partial).
        """
        events: list[CLIEvent] = []
        text, new_position = self._read_new_text(meta.log_file, position)
        if text:
            position = new_position
            if meta.interactive:
                self._process_interactive_chunk(meta=meta, offset=position)
            else:
                partial, parsed = self._split_to_events(task_id=meta.task_id, text=partial + text)
                events.extend(parsed)

        if partial and not meta.interactive:
            events.append(CLIEvent(type=EventType.STDOUT, task_id=meta.task_id, content=partial))
            partial = ""
        return events, partial

    def _finalize_watch_result(
        self,
        *,
        meta: _TmuxTaskMeta,
        timed_out: bool,
        canceled: bool,
        exit_code: int | None,
        timeout_sec: int,
        started_at: float,
        finished_at: float,
        position: int,
    ) -> CLIEvent:
        """Determine the final CLIEvent based on watch outcome."""
        if timed_out:
            finish_extra = self._tmux_log_extra(
                meta,
                result="timeout",
                exit_code=exit_code,
                timeout_sec=timeout_sec,
                elapsed_sec=round(finished_at - started_at, 3),
                log_position=position,
                canceled=canceled,
            )
            logger.warning("tmux task finished", extra=finish_extra)
            return CLIEvent(type=EventType.TIMEOUT, task_id=meta.task_id, error=f"任务超时({timeout_sec}s)")
        elif canceled:
            finish_extra = self._tmux_log_extra(
                meta,
                result="canceled",
                exit_code=exit_code,
                timeout_sec=timeout_sec,
                elapsed_sec=round(finished_at - started_at, 3),
                log_position=position,
                canceled=True,
            )
            logger.info("tmux task finished", extra=finish_extra)
            return CLIEvent(type=EventType.CANCELED, task_id=meta.task_id, error="任务已取消")
        elif exit_code == 0:
            finish_extra = self._tmux_log_extra(
                meta,
                result="exited",
                exit_code=exit_code,
                timeout_sec=timeout_sec,
                elapsed_sec=round(finished_at - started_at, 3),
                log_position=position,
                canceled=False,
            )
            logger.info("tmux task finished", extra=finish_extra)
            return CLIEvent(type=EventType.EXITED, task_id=meta.task_id, exit_code=0)
        elif exit_code is None:
            finish_extra = self._tmux_log_extra(
                meta,
                result="failed",
                exit_code=exit_code,
                timeout_sec=timeout_sec,
                elapsed_sec=round(finished_at - started_at, 3),
                log_position=position,
                canceled=False,
            )
            logger.error("tmux task finished", extra=finish_extra)
            return CLIEvent(type=EventType.FAILED, task_id=meta.task_id, error="进程已结束，但未能读取退出码")
        else:
            finish_extra = self._tmux_log_extra(
                meta,
                result="failed",
                exit_code=exit_code,
                timeout_sec=timeout_sec,
                elapsed_sec=round(finished_at - started_at, 3),
                log_position=position,
                canceled=False,
            )
            logger.error("tmux task finished", extra=finish_extra)
            return CLIEvent(type=EventType.FAILED, task_id=meta.task_id, exit_code=exit_code, error=f"进程退出码: {exit_code}")

    def _process_interactive_chunk(self, *, meta: _TmuxTaskMeta, offset: int) -> None:
        state = self._session_store.mark_interactive_turn_processing(
            terminal_id=meta.terminal_id,
            workdir=meta.workdir,
            claude_session_id=meta.claude_session_id,
            fallback_session_id=meta.session_name,
        )
        if state is None:
            return
        session_id = state.session_id
        if is_claude_session_id(session_id):
            meta.claude_session_id = session_id

    async def cancel(self, task_id: str) -> bool:
        async with self._lock:
            meta = self._tasks.get(task_id)
            if meta is None:
                return False
            meta.cancel_requested = True
            session_name = meta.session_name
            persistent_terminal = meta.persistent_terminal
            queued = persistent_terminal and meta.command_started_at is None
            action = "dequeue" if queued else ("interrupt" if persistent_terminal else "terminate")
            log_extra = self._tmux_log_extra(meta, action=action)
        logger.info("tmux task cancel requested", extra=log_extra)
        if queued:
            return True
        if persistent_terminal:
            return await self._interrupt_locked(session_name)
        return await self._terminate_session(session_name)

    def get_session_state(self, terminal_key: str) -> SessionState | None:
        state = self._session_store.get(terminal_key)
        if state is not None:
            return state
        session_name = self.build_session_name(terminal_key)
        return self._session_store.get(session_name)

    async def close_terminal(self, terminal_key: str) -> tuple[bool, str]:
        session_name = self.build_session_name(terminal_key)
        # Cancel any tasks running on this terminal so they release the session lock.
        async with self._lock:
            for meta in self._tasks.values():
                if meta.session_name == session_name and not meta.cancel_requested:
                    meta.cancel_requested = True
        async with self._session_lock(session_name):
            try:
                exists = await self.session_exists(session_name)
            except TmuxSessionCheckError as exc:
                return False, f"终端状态检查失败: {exc}"
            if not exists:
                return False, "终端不存在"
            closed = await self._terminate_session(session_name)
            if not closed:
                return False, "终端关闭失败"
        return True, ""

    async def ensure_terminal(self, *, terminal_key: str, workdir: str, env: dict[str, str] | None = None) -> tuple[bool, str]:
        session_name = self.build_session_name(terminal_key)
        async with self._session_lock(session_name):
            return await self._ensure_persistent_session(session_name, workdir=workdir, env=env)

    async def ensure_claude_interactive_session(
        self, *, terminal_key: str, workdir: str, env: dict[str, str] | None = None
    ) -> tuple[bool, str]:
        session_name = self.build_session_name(terminal_key)
        async with self._session_lock(session_name):
            return await self._ensure_claude_interactive_session(session_name=session_name, workdir=workdir, env=env)

    async def ensure_claude_resume_session(
        self, *, terminal_key: str, workdir: str, session_id: str, env: dict[str, str] | None = None
    ) -> tuple[bool, str]:
        session_name = self.build_session_name(terminal_key)
        async with self._session_lock(session_name):
            return await self._ensure_claude_resume_session(session_name=session_name, workdir=workdir, session_id=session_id, env=env)

    async def send_interactive_input(self, *, terminal_key: str, workdir: str, text: str) -> tuple[bool, str]:
        session_name = self.build_session_name(terminal_key)
        prompt = self._wrap_interactive_prompt(prompt=text)
        async with self._session_lock(session_name):
            async with self._pane_io_lock(session_name):
                ready, err = await self._ensure_claude_interactive_session(session_name=session_name, workdir=workdir, env=None)
                if not ready:
                    return False, err
                return await self._send_command(session_name, prompt, workdir=workdir, env=None, interactive=True)

    async def select_user_question_option(
        self,
        *,
        terminal_key: str,
        workdir: str,
        option_index: int,
        submit_after: bool = False,
    ) -> tuple[bool, str]:
        session_name = self.build_session_name(terminal_key)
        async with self._pane_io_lock(session_name):
            ready, err = await self._ensure_live_claude_session_for_user_question(session_name=session_name)
            if not ready:
                return False, err
            ok, err = await self._move_user_question_cursor_to_option(session_name=session_name, target_index=option_index)
            if not ok:
                return False, err
            ok, err = await self._send_keys(session_name, "C-m")
            if not ok:
                return False, err
            if submit_after:
                await asyncio.sleep(self._enter_delay_sec)
                ok, err = await self._send_keys(session_name, "C-m")
                if not ok:
                    return False, err
            return True, ""

    async def answer_user_question_with_text(
        self,
        *,
        terminal_key: str,
        workdir: str,
        option_count: int,
        text: str,
        submit_after: bool = False,
    ) -> tuple[bool, str]:
        session_name = self.build_session_name(terminal_key)
        async with self._pane_io_lock(session_name):
            ready, err = await self._ensure_live_claude_session_for_user_question(session_name=session_name)
            if not ready:
                return False, err
            ok, err = await self._move_user_question_cursor_to_option(session_name=session_name, target_index=option_count)
            if not ok:
                return False, err
            ok, err = await self._send_keys(session_name, "C-m")
            if not ok:
                return False, err
            await asyncio.sleep(self._enter_delay_sec)
            ok, err = await self._paste_text(session_name, text)
            if not ok:
                return False, err
            ok, err = await self._send_keys(session_name, "C-m")
            if not ok:
                return False, err
            if submit_after:
                await asyncio.sleep(self._enter_delay_sec)
                ok, err = await self._send_keys(session_name, "C-m")
                if not ok:
                    return False, err
            return True, ""

    async def advance_user_question_after_multi_select(
        self,
        *,
        terminal_key: str,
        workdir: str,
        final_question: bool,
    ) -> tuple[bool, str]:
        session_name = self.build_session_name(terminal_key)
        async with self._pane_io_lock(session_name):
            ready, err = await self._ensure_live_claude_session_for_user_question(session_name=session_name)
            if not ready:
                return False, err
            ok, err = await self._send_keys(session_name, "Right")
            if not ok:
                return False, err
            if final_question:
                await asyncio.sleep(self._enter_delay_sec)
                ok, err = await self._send_keys(session_name, "C-m")
                if not ok:
                    return False, err
            return True, ""

    async def reveal_terminal(self, terminal_key: str) -> tuple[bool, str]:
        session_name = self.build_session_name(terminal_key)
        try:
            exists = await self.session_exists(session_name)
        except TmuxSessionCheckError as exc:
            return False, f"终端状态检查失败: {exc}"
        if not exists:
            return False, f"tmux 会话不存在: {session_name}\nhint: 请先发送 /claude 创建会话后再打开桌面终端"
        resolved_tmux_bin = shutil.which(self._tmux_bin)
        if resolved_tmux_bin:
            tmux_bin = str(Path(resolved_tmux_bin).resolve())
        elif Path(self._tmux_bin).exists():
            tmux_bin = str(Path(self._tmux_bin).resolve())
        else:
            tmux_bin = self._tmux_bin
        attach_command = shlex.join(
            [
                "/usr/bin/env",
                "-u",
                "TMUX",
                "-u",
                "TMUX_PANE",
                tmux_bin,
                "attach-session",
                "-t",
                session_name,
            ]
        )
        try:
            process = await asyncio.create_subprocess_exec(
                "osascript",
                "-e",
                _TERMINAL_REVEAL_SCRIPT,
                "--",
                attach_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
        except FileNotFoundError:
            return False, "找不到 osascript（仅支持 macOS 桌面）\nhint: 可手动执行 `tmux attach -t <tmux_session>`"
        except Exception as exc:
            logger.warning("reveal_terminal 启动 osascript 失败: %s", exc, exc_info=True)
            return False, f"打开桌面终端失败: {exc}"
        if process.returncode != 0:
            err = stderr.decode(errors="replace").strip() or stdout.decode(errors="replace").strip() or "unknown error"
            logger.warning("reveal_terminal osascript 退出码非 0: session=%s err=%s", session_name, err)
            return False, f"打开桌面终端失败: {err}\nhint: 可手动执行 `tmux attach -t {session_name}`"
        logger.debug("reveal_terminal 已通过 osascript 打开并附着: session=%s", session_name)
        return True, f"已在桌面打开 Terminal 并附着到 {session_name}"

    async def _is_cancel_requested(self, task_id: str) -> bool:
        async with self._lock:
            meta = self._tasks.get(task_id)
            return bool(meta and meta.cancel_requested)

    @asynccontextmanager
    async def _session_lock(self, session_name: str) -> AsyncIterator[None]:
        async with self._session_locks.lock(session_name):
            yield

    @asynccontextmanager
    async def _pane_io_lock(self, session_name: str) -> AsyncIterator[None]:
        async with self._pane_io_locks.lock(session_name):
            yield

    async def _is_claude_idle_in_pane(self, session_name: str) -> bool:
        """Check if Claude Code TUI is showing an input prompt (idle state).

        This is a fallback detection for when Esc cancellation doesn't produce
        a hook event. Claude Code shows a specific pattern when idle:
        - A prompt line starting with ❯ or ›
        - Followed by a horizontal rule (────) separator
        - Followed by a status bar (model | branch | files)
        """
        pane_text = await self._capture_pane_text(session_name, start_line=-15)
        if not pane_text:
            return False
        lines = pane_text.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith(("›", "❯")):
                continue
            # Found a prompt line; check if the next non-empty line is a separator
            for j in range(i + 1, min(i + 3, len(lines))):
                next_stripped = lines[j].strip()
                if not next_stripped:
                    continue
                if next_stripped.startswith("─") and len(next_stripped) >= 10:
                    return True
                break
        return False

    def _capture_interactive_baseline(self, *, meta: _TmuxTaskMeta) -> None:
        resolved_session_id = self._session_store.resolve_interactive_session_id(
            terminal_id=meta.terminal_id,
            claude_session_id=meta.claude_session_id,
            fallback_session_id=meta.session_name,
            require_claude_session=True,
        )
        if resolved_session_id is not None:
            meta.claude_session_id = resolved_session_id
        state = self._session_store.get_interactive_state(
            terminal_id=meta.terminal_id,
            workdir=meta.workdir,
            claude_session_id=meta.claude_session_id,
            fallback_session_id=meta.session_name,
            require_claude_session=True,
        )
        if state is None or not is_claude_session_id(state.session_id):
            meta.baseline_captured = False
            meta.baseline_offset = 0
            meta.baseline_completed_turn_id = None
            return
        meta.claude_session_id = state.session_id
        self._record_interactive_baseline(meta=meta, state=state)

    def _record_interactive_baseline(self, *, meta: _TmuxTaskMeta, state: SessionState) -> None:
        meta.baseline_captured = True
        meta.baseline_offset = state.checkpoint.last_offset
        latest_completed_turn = self._latest_completed_assistant_turn(state)
        meta.baseline_completed_turn_id = latest_completed_turn.turn_id if latest_completed_turn is not None else None

    def _latest_completed_assistant_turn(self, state: SessionState) -> ConversationTurn | None:
        return next(
            (turn for turn in reversed(state.turns) if turn.role == "assistant" and turn.is_complete),
            None,
        )

    def _interactive_completion_candidate_key(self, state: SessionState, turn: ConversationTurn) -> tuple[object, ...] | None:
        if state.phase not in {SessionPhase.PROCESSING, SessionPhase.WAITING_FOR_INPUT, SessionPhase.ENDED}:
            return None
        if state.pending_permission is not None:
            return None
        if self._has_active_tool_call(state):
            return None
        if self._has_tool_call_started_after_turn(state, turn):
            return None
        return (
            state.session_id,
            state.phase.value,
            state.checkpoint.last_offset,
            turn.turn_id,
            tuple(
                sorted(
                    (
                        tool.tool_use_id,
                        tool.status.value,
                        tool.started_at.isoformat(),
                        tool.completed_at.isoformat() if tool.completed_at else "",
                        tuple(
                            sorted(
                                (
                                    subtool.tool_use_id,
                                    subtool.status.value,
                                    subtool.started_at.isoformat(),
                                    subtool.completed_at.isoformat() if subtool.completed_at else "",
                                )
                                for subtool in tool.subagent_tools
                            )
                        ),
                    )
                    for tool in state.tool_calls.values()
                )
            ),
        )

    def _has_active_tool_call(self, state: SessionState) -> bool:
        active_statuses = {ToolStatus.RUNNING, ToolStatus.WAITING_FOR_APPROVAL}
        return any(
            tool.status in active_statuses or any(subtool.status in active_statuses for subtool in tool.subagent_tools)
            for tool in state.tool_calls.values()
        )

    def _has_tool_call_started_after_turn(self, state: SessionState, turn: ConversationTurn) -> bool:
        turn_completed_at = turn.ended_at or turn.started_at
        for tool in state.tool_calls.values():
            if tool.started_at >= turn_completed_at:
                return True
            if any(subtool.started_at >= turn_completed_at for subtool in tool.subagent_tools):
                return True
        return False

    def _init_interactive_watch(
        self,
        *,
        meta: _TmuxTaskMeta,
        watch_state: _InteractiveWatchState,
        now: float,
    ) -> int | None:
        """Initialize interactive watch state before the main loop. Returns exit_code if already complete."""
        state = self._session_store.mark_interactive_turn_processing(
            terminal_id=meta.terminal_id,
            workdir=meta.workdir,
            claude_session_id=meta.claude_session_id,
            fallback_session_id=meta.session_name,
        )
        if state is not None:
            if is_claude_session_id(state.session_id):
                meta.claude_session_id = state.session_id
                if not meta.baseline_captured:
                    latest_completed_turn = self._latest_completed_assistant_turn(state)
                    is_current = self._is_current_completed_turn(latest_completed_turn, meta=meta, watch_state=watch_state)
                    if is_current and latest_completed_turn is not None:
                        if self._is_interactive_completion_turn_ready(state, latest_completed_turn, watch_state, observed_at=now):
                            return 0
                    else:
                        self._record_interactive_baseline(meta=meta, state=state)
                        watch_state.structured_offset_before_run = meta.baseline_offset
                        watch_state.latest_completed_turn_id_before_run = meta.baseline_completed_turn_id
            elif not meta.baseline_captured:
                watch_state.structured_offset_before_run = state.checkpoint.last_offset
        return None

    def _is_interactive_completion_turn_ready(
        self,
        state: SessionState,
        turn: ConversationTurn,
        watch_state: _InteractiveWatchState,
        *,
        observed_at: float,
    ) -> bool:
        candidate_key = self._interactive_completion_candidate_key(state, turn)
        if candidate_key is None:
            watch_state.completion_candidate_key = None
            watch_state.completion_candidate_seen_at = None
            return False
        if candidate_key != watch_state.completion_candidate_key:
            watch_state.completion_candidate_key = candidate_key
            watch_state.completion_candidate_seen_at = observed_at
            return self._interactive_completion_grace_sec <= 0
        return (
            watch_state.completion_candidate_seen_at is not None
            and (observed_at - watch_state.completion_candidate_seen_at) >= self._interactive_completion_grace_sec
        )

    def _is_current_completed_turn(
        self,
        turn: ConversationTurn | None,
        *,
        meta: _TmuxTaskMeta,
        watch_state: _InteractiveWatchState,
    ) -> bool:
        if turn is None:
            return False
        if meta.command_started_at is not None:
            return turn.started_at >= watch_state.completion_started_after
        if meta.baseline_captured:
            return turn.turn_id != watch_state.latest_completed_turn_id_before_run
        return turn.started_at >= watch_state.watch_started_at

    def _tick_interactive_watch(
        self,
        *,
        meta: _TmuxTaskMeta,
        watch_state: _InteractiveWatchState,
        now: float,
    ) -> tuple[int | None, SessionState | None]:
        """Process one interactive watch tick.

        Returns (exit_code, active_state). exit_code is set if completion detected.
        active_state is returned so the caller can check revision changes for timeout reset.
        """
        resolved_session_id = self._session_store.resolve_interactive_session_id(
            terminal_id=meta.terminal_id,
            claude_session_id=meta.claude_session_id,
            fallback_session_id=meta.session_name,
            require_claude_session=True,
        )
        if resolved_session_id is not None:
            meta.claude_session_id = resolved_session_id
        active_state = self._session_store.get_interactive_state(
            terminal_id=meta.terminal_id,
            workdir=meta.workdir,
            claude_session_id=meta.claude_session_id,
            fallback_session_id=meta.session_name,
            require_claude_session=True,
        )
        latest_completed_turn = self._latest_completed_assistant_turn(active_state) if active_state is not None else None
        latest_completed_turn_is_current = self._is_current_completed_turn(
            latest_completed_turn,
            meta=meta,
            watch_state=watch_state,
        )
        if active_state is not None and is_claude_session_id(active_state.session_id) and not meta.baseline_captured:
            meta.claude_session_id = active_state.session_id
            if latest_completed_turn_is_current and latest_completed_turn is not None:
                if self._is_interactive_completion_turn_ready(active_state, latest_completed_turn, watch_state, observed_at=now):
                    return 0, active_state
            else:
                self._record_interactive_baseline(meta=meta, state=active_state)
                watch_state.structured_offset_before_run = meta.baseline_offset
                watch_state.latest_completed_turn_id_before_run = meta.baseline_completed_turn_id
        if active_state is not None and active_state.checkpoint.last_offset > watch_state.structured_offset_before_run:
            if latest_completed_turn is not None and not latest_completed_turn_is_current:
                watch_state.structured_offset_before_run = active_state.checkpoint.last_offset
                watch_state.latest_completed_turn_id_before_run = latest_completed_turn.turn_id
                meta.baseline_offset = watch_state.structured_offset_before_run
                meta.baseline_completed_turn_id = watch_state.latest_completed_turn_id_before_run
            else:
                watch_state.saw_interactive_progress = True
                watch_state.structured_offset_before_run = active_state.checkpoint.last_offset
                meta.baseline_offset = watch_state.structured_offset_before_run
        completion_phase = self._session_store.interactive_completion_phase(
            terminal_id=meta.terminal_id,
            workdir=meta.workdir,
            claude_session_id=meta.claude_session_id,
            fallback_session_id=meta.session_name,
        )
        completion_ready = latest_completed_turn is None or (
            latest_completed_turn_is_current
            and active_state is not None
            and self._is_interactive_completion_turn_ready(active_state, latest_completed_turn, watch_state, observed_at=now)
        )
        if completion_phase is not None and watch_state.saw_interactive_progress and completion_ready:
            return 0, active_state
        if latest_completed_turn is not None and latest_completed_turn_is_current:
            if active_state is not None and self._is_interactive_completion_turn_ready(
                active_state, latest_completed_turn, watch_state, observed_at=now
            ):
                return 0, active_state
        elif latest_completed_turn is not None and latest_completed_turn.turn_id != watch_state.latest_completed_turn_id_before_run:
            watch_state.latest_completed_turn_id_before_run = latest_completed_turn.turn_id
            meta.baseline_completed_turn_id = watch_state.latest_completed_turn_id_before_run
        return None, active_state
