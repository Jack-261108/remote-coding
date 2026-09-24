from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.domain.session_models import SessionEvent, SessionPhase, SessionState, ToolCallRecord
from app.services.session_supervisor import SessionSupervisor


class _Snapshot:
    def __init__(self, *, interrupt_detected: bool = False) -> None:
        self.interrupt_detected = interrupt_detected

    def to_payload(self) -> dict:
        return {}


class _FakeParser:
    def __init__(self, *, subagent_file: Path | None = None, interrupt_detected: bool = False) -> None:
        self._subagent_file = subagent_file
        self._interrupt_detected = interrupt_detected
        self.reset_calls: list[str] = []

    def parse_incremental(self, *, session_id: str, cwd: str) -> _Snapshot:
        return _Snapshot(interrupt_detected=self._interrupt_detected)

    def session_file_path(self, *, session_id: str, cwd: str) -> Path:
        return Path(cwd) / f"{session_id}.jsonl"

    def subagent_file_path(self, *, session_id: str, agent_id: str, cwd: str) -> Path:
        assert self._subagent_file is not None
        return self._subagent_file

    def reset_state(self, session_id: str) -> None:
        self.reset_calls.append(session_id)


class _FakeStore:
    def __init__(self, state: SessionState | None) -> None:
        self.state = state

    def get(self, session_id: str) -> SessionState | None:
        if self.state is not None and self.state.session_id == session_id:
            return self.state
        return None


class _FakeJSONLFileWatcher:
    def __init__(self, *, available: bool = True) -> None:
        self.is_available = available
        self.watched: dict[str, tuple[str, set[Path]]] = {}
        self.unwatched: list[str] = []
        self.cleared = False

    def watch_files(self, *, session_id: str, cwd: str, paths) -> None:
        self.watched[session_id] = (cwd, set(paths))

    def replace_session_files(self, *, session_id: str, cwd: str, paths) -> None:
        self.watched[session_id] = (cwd, set(paths))

    def unwatch_session(self, session_id: str) -> None:
        self.unwatched.append(session_id)
        self.watched.pop(session_id, None)

    def clear(self) -> None:
        self.cleared = True
        self.watched.clear()


@pytest.mark.asyncio
async def test_session_supervisor_tick_exception_does_not_stop_watcher(caplog: pytest.LogCaptureFixture) -> None:
    state = SessionState(session_id="claude-session-1", workdir="/tmp", phase=SessionPhase.PROCESSING)
    calls = 0
    dispatched = asyncio.Event()

    async def on_dispatch_event(event: SessionEvent) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("dispatch failed")
        dispatched.set()

    supervisor = SessionSupervisor(
        session_store=_FakeStore(state),
        claude_jsonl_parser=_FakeParser(interrupt_detected=True),
        on_jsonl_sync=lambda session_id, cwd: asyncio.sleep(0),
        on_dispatch_event=on_dispatch_event,
        poll_interval_sec=0.01,
        debounce_sec=0.01,
    )
    caplog.set_level("ERROR", logger="app.services.session_supervisor")

    supervisor.watch(session_id=state.session_id, workdir=state.workdir)
    try:
        await asyncio.wait_for(dispatched.wait(), timeout=1)
    finally:
        await supervisor.stop_all()

    assert calls >= 2
    assert any(record.message == "session supervisor tick failed" for record in caplog.records)


@pytest.mark.asyncio
async def test_session_supervisor_subagent_sync_exception_keeps_watcher_running(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    subagent_file = tmp_path / "agent.jsonl"
    subagent_file.write_text("first", encoding="utf-8")
    state = SessionState(session_id="claude-session-1", workdir=str(tmp_path))
    state.tool_calls["task-1"] = ToolCallRecord(
        tool_use_id="task-1",
        name="Task",
        structured_result={"agentId": "agent-1"},
    )
    sync_calls = 0
    second_sync = asyncio.Event()

    async def on_jsonl_sync(session_id: str, cwd: str) -> None:
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 1:
            subagent_file.write_text("second", encoding="utf-8")
            raise RuntimeError("sync failed")
        second_sync.set()

    supervisor = SessionSupervisor(
        session_store=_FakeStore(state),
        claude_jsonl_parser=_FakeParser(subagent_file=subagent_file),
        on_jsonl_sync=on_jsonl_sync,
        on_dispatch_event=lambda event: asyncio.sleep(0),
        poll_interval_sec=0.01,
        debounce_sec=0.01,
    )
    caplog.set_level("ERROR", logger="app.services.session_supervisor")

    supervisor.watch(session_id=state.session_id, workdir=state.workdir)
    try:
        await asyncio.wait_for(second_sync.wait(), timeout=1)
    finally:
        await supervisor.stop_all()

    assert sync_calls >= 2
    assert any(record.message == "session supervisor tick failed" for record in caplog.records)


@pytest.mark.asyncio
async def test_session_supervisor_subagent_sync_retries_when_mtime_unchanged(
    tmp_path: Path,
) -> None:
    subagent_file = tmp_path / "agent.jsonl"
    subagent_file.write_text("first", encoding="utf-8")
    state = SessionState(session_id="claude-session-1", workdir=str(tmp_path))
    state.tool_calls["task-1"] = ToolCallRecord(
        tool_use_id="task-1",
        name="Task",
        structured_result={"agentId": "agent-1"},
    )
    sync_calls = 0
    second_sync = asyncio.Event()

    async def on_jsonl_sync(session_id: str, cwd: str) -> None:
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 1:
            raise RuntimeError("sync failed")
        second_sync.set()

    supervisor = SessionSupervisor(
        session_store=_FakeStore(state),
        claude_jsonl_parser=_FakeParser(subagent_file=subagent_file),
        on_jsonl_sync=on_jsonl_sync,
        on_dispatch_event=lambda event: asyncio.sleep(0),
        poll_interval_sec=0.01,
        debounce_sec=0.01,
    )

    supervisor.watch(session_id=state.session_id, workdir=state.workdir)
    try:
        await asyncio.wait_for(second_sync.wait(), timeout=1)
    finally:
        await supervisor.stop_all()

    assert sync_calls >= 2


@pytest.mark.asyncio
async def test_session_supervisor_subagent_sync_runs_outside_watcher_lock(tmp_path: Path) -> None:
    subagent_file = tmp_path / "agent.jsonl"
    subagent_file.write_text("initial", encoding="utf-8")
    state = SessionState(session_id="claude-session-1", workdir=str(tmp_path))
    state.tool_calls["task-1"] = ToolCallRecord(
        tool_use_id="task-1",
        name="Task",
        structured_result={"agentId": "agent-1"},
    )
    synced = asyncio.Event()
    was_locked_during_sync: bool | None = None

    supervisor: SessionSupervisor | None = None

    async def on_jsonl_sync(session_id: str, cwd: str) -> None:
        nonlocal was_locked_during_sync
        assert supervisor is not None
        lock = supervisor._locks.get(session_id)
        was_locked_during_sync = lock.locked() if lock is not None else False
        synced.set()

    supervisor = SessionSupervisor(
        session_store=_FakeStore(state),
        claude_jsonl_parser=_FakeParser(subagent_file=subagent_file),
        on_jsonl_sync=on_jsonl_sync,
        on_dispatch_event=lambda event: asyncio.sleep(0),
        poll_interval_sec=0.01,
        debounce_sec=0.01,
    )

    supervisor.watch(session_id=state.session_id, workdir=state.workdir)
    try:
        await asyncio.wait_for(synced.wait(), timeout=1)
    finally:
        await supervisor.stop_all()

    assert was_locked_during_sync is False


@pytest.mark.asyncio
async def test_session_supervisor_done_callback_restarts_crashed_watcher(monkeypatch: pytest.MonkeyPatch) -> None:
    state = SessionState(session_id="claude-session-1", workdir="/tmp")
    supervisor = SessionSupervisor(
        session_store=_FakeStore(state),
        claude_jsonl_parser=_FakeParser(),
        on_jsonl_sync=lambda session_id, cwd: asyncio.sleep(0),
        on_dispatch_event=lambda event: asyncio.sleep(0),
        poll_interval_sec=0.01,
        debounce_sec=0.01,
    )
    calls = 0

    async def fake_watch_session(*, session_id: str, workdir: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("watcher crashed")
        await asyncio.sleep(10)

    monkeypatch.setattr(supervisor, "_watch_session", fake_watch_session)

    supervisor.watch(session_id=state.session_id, workdir=state.workdir)
    try:
        for _ in range(100):
            if calls >= 2:
                break
            await asyncio.sleep(0.01)
        assert calls >= 2
        restarted = supervisor._tasks[state.session_id]
        assert restarted.done() is False
    finally:
        await supervisor.stop_all()


@pytest.mark.asyncio
async def test_session_supervisor_missing_state_exits_without_restart() -> None:
    supervisor = SessionSupervisor(
        session_store=_FakeStore(None),
        claude_jsonl_parser=_FakeParser(),
        on_jsonl_sync=lambda session_id, cwd: asyncio.sleep(0),
        on_dispatch_event=lambda event: asyncio.sleep(0),
        poll_interval_sec=0.01,
        debounce_sec=0.01,
    )

    supervisor.watch(session_id="missing", workdir="/tmp")
    await asyncio.sleep(0.05)

    assert supervisor._tasks == {}
    assert supervisor._locks == {}


@pytest.mark.asyncio
async def test_session_supervisor_stop_all_cleans_state() -> None:
    state = SessionState(session_id="claude-session-1", workdir="/tmp")
    file_watcher = _FakeJSONLFileWatcher()
    supervisor = SessionSupervisor(
        session_store=_FakeStore(state),
        claude_jsonl_parser=_FakeParser(),
        on_jsonl_sync=lambda session_id, cwd: asyncio.sleep(0),
        on_dispatch_event=lambda event: asyncio.sleep(0),
        poll_interval_sec=0.01,
        debounce_sec=0.01,
        jsonl_file_watcher=file_watcher,
    )

    supervisor.watch(session_id=state.session_id, workdir=state.workdir)
    supervisor.schedule_jsonl_sync(state.session_id, state.workdir)
    supervisor._seen_mtimes["file"] = 1.0
    supervisor._session_mtime_keys[state.session_id] = {"file"}

    await supervisor.stop_all()

    assert supervisor._active is False
    assert supervisor._tasks == {}
    assert supervisor._locks == {}
    assert supervisor._wake_events == {}
    assert supervisor._jsonl_sync_requests == {}
    assert supervisor._main_jsonl_signatures == {}
    assert supervisor._seen_mtimes == {}
    assert supervisor._session_mtime_keys == {}
    assert file_watcher.cleared is True


@pytest.mark.asyncio
async def test_session_supervisor_stop_blocks_new_watchers_until_reopened() -> None:
    state = SessionState(session_id="claude-session-1", workdir="/tmp")
    supervisor = SessionSupervisor(
        session_store=_FakeStore(state),
        claude_jsonl_parser=_FakeParser(),
        on_jsonl_sync=lambda session_id, cwd: asyncio.sleep(0),
        on_dispatch_event=lambda event: asyncio.sleep(0),
        poll_interval_sec=0.01,
        debounce_sec=0.01,
    )

    await supervisor.stop_all()
    supervisor.watch(session_id=state.session_id, workdir=state.workdir)
    supervisor.schedule_jsonl_sync(state.session_id, state.workdir)

    assert supervisor._tasks == {}
    assert supervisor._jsonl_sync_requests == {}

    supervisor.reopen()
    supervisor.watch(session_id=state.session_id, workdir=state.workdir)
    assert state.session_id in supervisor._tasks
    await supervisor.stop_all()


@pytest.mark.asyncio
async def test_session_supervisor_forget_cleans_session_state() -> None:
    state = SessionState(session_id="claude-session-1", workdir="/tmp")
    file_watcher = _FakeJSONLFileWatcher()
    supervisor = SessionSupervisor(
        session_store=_FakeStore(state),
        claude_jsonl_parser=_FakeParser(),
        on_jsonl_sync=lambda session_id, cwd: asyncio.sleep(0),
        on_dispatch_event=lambda event: asyncio.sleep(0),
        poll_interval_sec=0.01,
        debounce_sec=0.01,
        jsonl_file_watcher=file_watcher,
    )

    supervisor.watch(session_id=state.session_id, workdir=state.workdir)
    supervisor.schedule_jsonl_sync(state.session_id, state.workdir)
    supervisor._seen_mtimes["file"] = 1.0
    supervisor._session_mtime_keys[state.session_id] = {"file"}
    assert state.session_id in supervisor._tasks

    await supervisor.forget(state.session_id)
    await supervisor.stop_all()

    assert state.session_id not in supervisor._tasks
    assert state.session_id not in supervisor._locks
    assert state.session_id not in supervisor._wake_events
    assert state.session_id not in supervisor._jsonl_sync_requests
    assert "file" not in supervisor._seen_mtimes
    assert state.session_id not in supervisor._session_mtime_keys
    assert file_watcher.unwatched == [state.session_id]


@pytest.mark.asyncio
async def test_session_supervisor_schedule_jsonl_sync_wakes_idle_watcher() -> None:
    state = SessionState(session_id="claude-session-1", workdir="/tmp", phase=SessionPhase.IDLE)
    synced = asyncio.Event()
    sync_calls = 0

    async def on_jsonl_sync(session_id: str, cwd: str) -> None:
        nonlocal sync_calls
        sync_calls += 1
        synced.set()

    supervisor = SessionSupervisor(
        session_store=_FakeStore(state),
        claude_jsonl_parser=_FakeParser(),
        on_jsonl_sync=on_jsonl_sync,
        on_dispatch_event=lambda event: asyncio.sleep(0),
        poll_interval_sec=1.0,
        idle_poll_interval_sec=60.0,
        debounce_sec=0.01,
        jsonl_file_watcher=_FakeJSONLFileWatcher(available=True),
    )

    supervisor.watch(session_id=state.session_id, workdir=state.workdir)
    await asyncio.sleep(0)
    supervisor.schedule_jsonl_sync(state.session_id, state.workdir)
    try:
        await asyncio.wait_for(synced.wait(), timeout=0.5)
    finally:
        await supervisor.stop_all()

    assert sync_calls == 1


@pytest.mark.asyncio
async def test_session_supervisor_repeated_jsonl_sync_keeps_first_debounce_deadline() -> None:
    state = SessionState(session_id="claude-session-1", workdir="/tmp", phase=SessionPhase.IDLE)
    supervisor = SessionSupervisor(
        session_store=_FakeStore(state),
        claude_jsonl_parser=_FakeParser(),
        on_jsonl_sync=lambda session_id, cwd: asyncio.sleep(0),
        on_dispatch_event=lambda event: asyncio.sleep(0),
        poll_interval_sec=1.0,
        idle_poll_interval_sec=60.0,
        debounce_sec=0.05,
        jsonl_file_watcher=_FakeJSONLFileWatcher(available=True),
    )

    supervisor.schedule_jsonl_sync(state.session_id, state.workdir)
    await asyncio.sleep(0.01)
    remaining_before = supervisor._next_wait_timeout(session_id=state.session_id, active_state=False)
    supervisor.schedule_jsonl_sync(state.session_id, f"{state.workdir}/next")
    remaining_after = supervisor._next_wait_timeout(session_id=state.session_id, active_state=False)

    assert remaining_after <= remaining_before
    assert supervisor._jsonl_sync_requests[state.session_id][0] == f"{state.workdir}/next"


@pytest.mark.asyncio
async def test_session_supervisor_uses_idle_interval_when_file_watcher_available() -> None:
    state = SessionState(session_id="claude-session-1", workdir="/tmp", phase=SessionPhase.IDLE)
    supervisor = SessionSupervisor(
        session_store=_FakeStore(state),
        claude_jsonl_parser=_FakeParser(),
        on_jsonl_sync=lambda session_id, cwd: asyncio.sleep(0),
        on_dispatch_event=lambda event: asyncio.sleep(0),
        poll_interval_sec=0.25,
        idle_poll_interval_sec=11.0,
        debounce_sec=0.1,
        jsonl_file_watcher=_FakeJSONLFileWatcher(available=True),
    )

    assert supervisor._next_wait_timeout(session_id=state.session_id, active_state=False) == 11.0
    assert supervisor._next_wait_timeout(session_id=state.session_id, active_state=True) == 0.25


@pytest.mark.asyncio
async def test_session_supervisor_falls_back_to_active_interval_when_file_watcher_unavailable() -> None:
    state = SessionState(session_id="claude-session-1", workdir="/tmp", phase=SessionPhase.IDLE)
    supervisor = SessionSupervisor(
        session_store=_FakeStore(state),
        claude_jsonl_parser=_FakeParser(),
        on_jsonl_sync=lambda session_id, cwd: asyncio.sleep(0),
        on_dispatch_event=lambda event: asyncio.sleep(0),
        poll_interval_sec=0.25,
        idle_poll_interval_sec=11.0,
        debounce_sec=0.1,
        jsonl_file_watcher=_FakeJSONLFileWatcher(available=False),
    )

    assert supervisor._next_wait_timeout(session_id=state.session_id, active_state=False) == 0.25


@pytest.mark.asyncio
async def test_session_supervisor_polls_main_jsonl_when_file_watcher_unavailable(tmp_path: Path) -> None:
    state = SessionState(session_id="claude-session-1", workdir=str(tmp_path), phase=SessionPhase.IDLE)
    session_file = tmp_path / "claude-session-1.jsonl"
    session_file.write_text("partial", encoding="utf-8")
    sync_calls = 0
    synced = asyncio.Event()

    async def on_jsonl_sync(session_id: str, cwd: str) -> None:
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 1:
            raise RuntimeError("transient sync failure")
        synced.set()

    supervisor = SessionSupervisor(
        session_store=_FakeStore(state),
        claude_jsonl_parser=_FakeParser(),
        on_jsonl_sync=on_jsonl_sync,
        on_dispatch_event=lambda event: asyncio.sleep(0),
        poll_interval_sec=0.01,
        idle_poll_interval_sec=60.0,
        debounce_sec=0.01,
        jsonl_file_watcher=_FakeJSONLFileWatcher(available=False),
    )

    supervisor.watch(session_id=state.session_id, workdir=state.workdir)
    await asyncio.sleep(0.02)
    session_file.write_text("completed reply", encoding="utf-8")
    try:
        await asyncio.wait_for(synced.wait(), timeout=1)
    finally:
        await supervisor.stop_all()

    assert sync_calls == 2


@pytest.mark.asyncio
async def test_session_supervisor_registers_main_and_subagent_jsonl_paths(tmp_path: Path) -> None:
    subagent_file = tmp_path / "agent-agent-1.jsonl"
    state = SessionState(session_id="claude-session-1", workdir=str(tmp_path), claude_session_id="real-session")
    state.tool_calls["task-1"] = ToolCallRecord(
        tool_use_id="task-1",
        name="Task",
        structured_result={"agentId": "agent-1"},
    )
    file_watcher = _FakeJSONLFileWatcher()
    supervisor = SessionSupervisor(
        session_store=_FakeStore(state),
        claude_jsonl_parser=_FakeParser(subagent_file=subagent_file),
        on_jsonl_sync=lambda session_id, cwd: asyncio.sleep(0),
        on_dispatch_event=lambda event: asyncio.sleep(0),
        poll_interval_sec=0.01,
        debounce_sec=0.01,
        jsonl_file_watcher=file_watcher,
    )

    supervisor.watch(session_id=state.session_id, workdir=state.workdir)
    try:
        await asyncio.sleep(0.02)
        cwd, paths = file_watcher.watched[state.session_id]
    finally:
        await supervisor.stop_all()

    assert cwd == state.workdir
    assert tmp_path / "real-session.jsonl" in paths
    assert subagent_file in paths
    assert tmp_path / "real-session" / "subagents" / "agent-agent-1.jsonl" in paths
