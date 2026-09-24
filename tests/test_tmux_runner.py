import asyncio
import logging
import shlex
from datetime import timedelta
from pathlib import Path

import pytest

from app.adapters.process import tmux_runner as tmux_runner_module
from app.adapters.process.tmux_runner import TmuxRunner, _InteractiveWatchState, _TmuxTaskMeta
from app.adapters.storage.file_session_store import FileSessionStore
from app.domain.models import CLIEvent, EventType, utc_now
from app.domain.session_models import ConversationTurn, PendingPermission, SessionPhase, ToolCallRecord, ToolStatus
from app.services.session_store import SessionStore
from tests.fakes.subprocess_process import FakeSubprocessProcess


async def _collect_events(stream):
    return [event async for event in stream]


def _runner_with_session_store(tmp_path: Path, **kwargs) -> TmuxRunner:
    file_store = FileSessionStore(str(tmp_path))
    return TmuxRunner(data_dir=str(tmp_path), file_store=file_store, session_store=SessionStore(file_store), **kwargs)


def test_build_shell_command_writes_script_file(tmp_path: Path) -> None:
    runner = TmuxRunner(data_dir=str(tmp_path))
    command_file = tmp_path / "x.cmd.sh"
    cmd = runner._build_shell_command(
        argv=["claude", "-p", "hello"],
        workdir="/tmp",
        log_file=tmp_path / "x.log",
        exit_file=tmp_path / "x.exit",
        command_file=command_file,
        hide_launcher_line=False,
    )
    assert cmd.startswith("bash ")
    script = command_file.read_text(encoding="utf-8")
    assert "tee -a" in script
    assert "PIPESTATUS[0]" in script


def test_build_shell_command_hides_launcher_in_persistent_mode(tmp_path: Path) -> None:
    runner = TmuxRunner(data_dir=str(tmp_path))
    command_file = tmp_path / "hidden.cmd.sh"
    cmd = runner._build_shell_command(
        argv=["claude", "-p", "hello"],
        workdir="/tmp",
        log_file=tmp_path / "x.log",
        exit_file=tmp_path / "x.exit",
        command_file=command_file,
        hide_launcher_line=True,
    )

    assert cmd.startswith("bash ")
    assert 'exec "${SHELL:-bash}" -l' in cmd


def test_tmux_runner_enter_delay_default_is_positive() -> None:
    runner = TmuxRunner()
    assert runner._enter_delay_sec > 0


def test_format_send_failure_contains_actionable_hint() -> None:
    runner = TmuxRunner()
    msg = runner._format_send_failure(
        base="tmux 粘贴命令失败",
        raw_err="pane not found",
        session_name="tgcli_user_1",
        rebuilt=False,
        rebuild_err="旧会话关闭失败",
    )

    assert "tmux 粘贴命令失败: pane not found" in msg
    assert "tmux_session: tgcli_user_1" in msg
    assert "auto_rebuilt: 否" in msg
    assert "rebuild_error: 旧会话关闭失败" in msg
    assert "hint:" in msg


def test_format_send_failure_after_rebuild_has_specific_hint() -> None:
    runner = TmuxRunner()
    msg = runner._format_send_failure(
        base="tmux 执行命令失败",
        raw_err="target pane missing",
        session_name="tgcli_user_2",
        rebuilt=True,
    )

    assert "auto_rebuilt: 是" in msg
    assert "自动重建已执行但仍失败" in msg


@pytest.mark.asyncio
async def test_ensure_persistent_session_rejects_missing_workdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = TmuxRunner()
    calls: list[tuple[str, ...]] = []

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        calls.append(args)
        return 0, "", ""

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)

    ok, err = await runner._ensure_persistent_session(
        "tgcli_user_1",
        workdir=str(tmp_path / "不存在的目录"),
        env=None,
    )

    assert ok is False
    assert err.startswith("workdir 不存在或不是目录:")
    assert calls == []


@pytest.mark.asyncio
async def test_respawn_and_send_command_success(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner()
    calls: list[tuple[str, ...]] = []

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        calls.append(args)
        return 0, "", ""

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)
    ok, err = await runner._respawn_and_send_command(
        session_name="tgcli_user_1",
        command="bash '/tmp/test.cmd.sh'; exec \"${SHELL:-bash}\" -l",
        workdir="/tmp",
    )

    assert ok is True
    assert err == ""
    assert calls[0][0:3] == ("respawn-pane", "-k", "-t")
    assert calls[0][3] == "tgcli_user_1"


@pytest.mark.asyncio
async def test_respawn_and_send_command_failure_has_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner()

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        return 1, "", "target not found"

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)
    ok, err = await runner._respawn_and_send_command(
        session_name="tgcli_user_1",
        command="bash '/tmp/test.cmd.sh'; exec \"${SHELL:-bash}\" -l",
        workdir="/tmp",
    )

    assert ok is False
    assert "tmux respawn 失败" in err
    assert "tmux_session: tgcli_user_1" in err
    assert "hint:" in err


@pytest.mark.asyncio
async def test_ensure_claude_interactive_session_respawns_when_not_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner(claude_cli_bin="claude")

    async def fake_ensure_persistent_session(session_name: str, *, workdir: str, env=None):
        return True, ""

    async def fake_session_current_command(session_name: str) -> str:
        return "bash"

    async def fake_respawn_and_send_command(*, session_name: str, command: str, workdir: str):
        assert session_name == "tgcli_user_1"
        assert workdir == "/tmp"
        assert "exec claude --append-system-prompt" in command
        return True, ""

    monkeypatch.setattr(runner, "_ensure_persistent_session", fake_ensure_persistent_session)
    monkeypatch.setattr(runner, "_session_current_command", fake_session_current_command)
    monkeypatch.setattr(runner, "_respawn_and_send_command", fake_respawn_and_send_command)

    ok, err = await runner.ensure_claude_interactive_session(terminal_key="user_1", workdir="/tmp")
    assert ok is True
    assert err == ""


@pytest.mark.asyncio
async def test_send_command_uses_ctrl_m_in_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner()
    seen_enter_key: list[str] = []

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        if args and args[0] == "send-keys" and len(args) >= 4 and args[1] == "-t" and args[2] == "tgcli_user_1":
            key = args[3]
            if key in {"Enter", "C-m"}:
                seen_enter_key.append(key)
        return 0, "", ""

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)

    ok, err = await runner._send_command(
        "tgcli_user_1",
        "hello",
        workdir="/tmp",
        env=None,
        interactive=True,
    )

    assert ok is True
    assert err == ""
    assert seen_enter_key == ["C-m"]


@pytest.mark.asyncio
async def test_send_command_reloads_buffer_after_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner()
    paste_calls = 0
    load_calls = 0

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        nonlocal paste_calls, load_calls
        if args and args[0] == "load-buffer":
            load_calls += 1
            return 0, "", ""
        if args and args[0] == "paste-buffer":
            paste_calls += 1
            if paste_calls == 1:
                return 1, "", "lost buffer"
            return 0, "", ""
        return 0, "", ""

    async def fake_force_rebuild_session(session_name: str, *, workdir: str, env, interactive: bool = False):
        assert interactive is True
        return True, ""

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)
    monkeypatch.setattr(runner, "_force_rebuild_session", fake_force_rebuild_session)

    ok, err = await runner._send_command("tgcli_user_1", "hello", workdir="/tmp", env=None, interactive=True)

    assert ok is True
    assert err == ""
    assert load_calls == 2
    assert paste_calls == 2


@pytest.mark.asyncio
async def test_send_interactive_input_waits_for_pane_io_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner()
    entered = asyncio.Event()
    allow_finish = asyncio.Event()
    calls: list[str] = []

    async def fake_ensure(*, session_name: str, workdir: str, env):
        calls.append("ensure")
        return True, ""

    async def fake_send(session_name: str, command: str, *, workdir: str, env, interactive: bool = False):
        calls.append("send")
        return True, ""

    monkeypatch.setattr(runner, "_ensure_claude_interactive_session", fake_ensure)
    monkeypatch.setattr(runner, "_send_command", fake_send)

    async def hold_lock() -> None:
        async with runner._pane_io_lock("tgcli_user_1"):
            entered.set()
            await allow_finish.wait()

    holder = asyncio.create_task(hold_lock())
    await asyncio.wait_for(entered.wait(), timeout=1)
    task = asyncio.create_task(runner.send_interactive_input(terminal_key="user_1", workdir="/tmp", text="hello"))
    await asyncio.sleep(0)
    assert calls == []
    allow_finish.set()

    ok, err = await task
    await holder

    assert ok is True
    assert err == ""
    assert calls == ["ensure", "send"]


@pytest.mark.asyncio
async def test_select_user_question_option_sends_expected_tmux_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner()
    key_calls: list[tuple[str, ...]] = []

    async def fake_session_exists(session_name: str) -> bool:
        return session_name == "tgcli_user_1"

    async def fake_session_current_command(session_name: str) -> str:
        assert session_name == "tgcli_user_1"
        return "claude"

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        if args and args[0] == "capture-pane":
            return 0, "› 1. 今天\n  2. 明天\n  3. 后天\nEnter to select · Tab/Arrow keys to navigate · Esc to cancel\n", ""
        if args and args[0] == "send-keys":
            key_calls.append(args)
        return 0, "", ""

    monkeypatch.setattr(runner, "session_exists", fake_session_exists)
    monkeypatch.setattr(runner, "_session_current_command", fake_session_current_command)
    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)

    ok, err = await runner.select_user_question_option(
        terminal_key="user_1",
        workdir="/tmp",
        option_index=2,
        submit_after=True,
    )

    assert ok is True
    assert err == ""
    assert key_calls == [
        ("send-keys", "-t", "tgcli_user_1", "Down", "Down"),
        ("send-keys", "-t", "tgcli_user_1", "C-m"),
        ("send-keys", "-t", "tgcli_user_1", "C-m"),
    ]


@pytest.mark.asyncio
async def test_answer_user_question_with_text_uses_type_something_path(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner()
    key_calls: list[tuple[str, ...]] = []
    buffer_inputs: list[bytes] = []

    async def fake_session_exists(session_name: str) -> bool:
        return session_name == "tgcli_user_1"

    async def fake_session_current_command(session_name: str) -> str:
        assert session_name == "tgcli_user_1"
        return "claude"

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        if args and args[0] == "capture-pane":
            return (
                0,
                "› 1. 今天\n  2. 明天\n  3. 后天\n  4. Type something.\nEnter to select · Tab/Arrow keys to navigate · Esc to cancel\n",
                "",
            )
        if args and args[0] == "load-buffer" and input_data is not None:
            buffer_inputs.append(input_data)
        if args and args[0] == "send-keys":
            key_calls.append(args)
        return 0, "", ""

    monkeypatch.setattr(runner, "session_exists", fake_session_exists)
    monkeypatch.setattr(runner, "_session_current_command", fake_session_current_command)
    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)

    ok, err = await runner.answer_user_question_with_text(
        terminal_key="user_1",
        workdir="/tmp",
        option_count=3,
        text="2026-05-01",
        submit_after=False,
    )

    assert ok is True
    assert err == ""
    assert key_calls == [
        ("send-keys", "-t", "tgcli_user_1", "Down", "Down", "Down"),
        ("send-keys", "-t", "tgcli_user_1", "C-m"),
        ("send-keys", "-t", "tgcli_user_1", "C-m"),
    ]
    assert buffer_inputs == [b"2026-05-01"]


@pytest.mark.asyncio
async def test_select_user_question_option_returns_text_fallback_when_not_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner()

    async def fake_session_exists(session_name: str) -> bool:
        return session_name == "tgcli_user_1"

    async def fake_session_current_command(session_name: str) -> str:
        return "claude"

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        if args and args[0] == "capture-pane":
            return 0, "请直接按选项回复我，格式示例：2 / 3 / 3\n", ""
        return 0, "", ""

    monkeypatch.setattr(runner, "session_exists", fake_session_exists)
    monkeypatch.setattr(runner, "_session_current_command", fake_session_current_command)
    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)

    ok, err = await runner.select_user_question_option(
        terminal_key="user_1",
        workdir="/tmp",
        option_index=1,
        submit_after=False,
    )

    assert ok is False
    assert err == "当前问题不是 Claude 选择框界面，将回退为文本回答"


def test_selected_user_question_option_index_uses_latest_tui_block() -> None:
    runner = TmuxRunner()
    pane_text = (
        "旧问题\n"
        "  1. 今天\n"
        "  2. 明天\n"
        "› 3. 后天\n"
        "Enter to select · Tab/Arrow keys to navigate · Esc to cancel\n"
        "当前问题\n"
        "› 1. 郑州站\n"
        "  2. 郑州东\n"
        "Enter to select · Tab/Arrow keys to navigate · Esc to cancel\n"
    )

    assert runner._selected_user_question_option_index(pane_text) == 0


@pytest.mark.asyncio
async def test_interactive_run_rebinds_pipe_to_session_transcript(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    seen_pipe_calls: list[tuple[str, ...]] = []

    async def fake_ensure(*, session_name: str, workdir: str, env=None):
        return True, ""

    async def fake_send(*args, **kwargs):
        return True, ""

    async def fake_watch(*, meta, timeout_sec: int, fifo_reader=None):
        yield CLIEvent(type=EventType.EXITED, task_id=meta.task_id, exit_code=0)

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        if args and args[0] == "pipe-pane":
            seen_pipe_calls.append(args)
        return 0, "", ""

    monkeypatch.setattr(runner, "_ensure_claude_interactive_session", fake_ensure)
    monkeypatch.setattr(runner, "_send_command", fake_send)
    monkeypatch.setattr(runner, "_watch_task", fake_watch)
    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)

    events = [
        e
        async for e in runner.run(
            task_id="t-pipe",
            argv=["hello"],
            workdir=str(tmp_path),
            timeout_sec=10,
            terminal_key="user_1",
            interactive=True,
        )
    ]

    assert [e.type for e in events] == [EventType.STARTED, EventType.EXITED]
    # 3 calls: _bind clears old pipe, _bind sets new FIFO pipe, finally clears pipe before FIFO cleanup
    assert len(seen_pipe_calls) == 3
    assert seen_pipe_calls[0] == ("pipe-pane", "-t", "tgcli_user_1")
    assert seen_pipe_calls[1][0:3] == ("pipe-pane", "-t", "tgcli_user_1")
    assert "cat >" in seen_pipe_calls[1][3]
    assert ".fifo" in seen_pipe_calls[1][3]
    assert seen_pipe_calls[2] == ("pipe-pane", "-t", "tgcli_user_1")


@pytest.mark.asyncio
async def test_interactive_run_rebuilds_and_retries_pipe_when_tmux_server_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    ensure_calls: list[str] = []
    pipe_bind_calls: list[tuple[str, ...]] = []

    async def fake_ensure(*, session_name: str, workdir: str, env=None):
        ensure_calls.append(session_name)
        return True, ""

    async def fake_send(*args, **kwargs):
        return True, ""

    async def fake_watch(*, meta, timeout_sec: int, fifo_reader=None):
        yield CLIEvent(type=EventType.EXITED, task_id=meta.task_id, exit_code=0)

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        if args and args[0] == "pipe-pane" and len(args) >= 4:
            pipe_bind_calls.append(args)
            if len(pipe_bind_calls) == 1:
                return 1, "", "no server running on /private/tmp/tmux-501/default"
        return 0, "", ""

    monkeypatch.setattr(runner, "_ensure_claude_interactive_session", fake_ensure)
    monkeypatch.setattr(runner, "_send_command", fake_send)
    monkeypatch.setattr(runner, "_watch_task", fake_watch)
    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)

    events = [
        e
        async for e in runner.run(
            task_id="t-pipe-retry",
            argv=["hello"],
            workdir=str(tmp_path),
            timeout_sec=10,
            terminal_key="user_1",
            interactive=True,
        )
    ]

    assert [e.type for e in events] == [EventType.STARTED, EventType.EXITED]
    assert ensure_calls == ["tgcli_user_1", "tgcli_user_1"]
    assert len(pipe_bind_calls) == 2


@pytest.mark.asyncio
async def test_watch_task_flushes_partial_content(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    task_id = "t1"
    log_file = tmp_path / f"{task_id}.log"
    exit_file = tmp_path / f"{task_id}.exit"

    meta = _TmuxTaskMeta(
        session_name="tgcli_user_1",
        log_file=log_file,
        exit_file=exit_file,
        task_id=task_id,
        workdir=str(tmp_path),
        persistent_terminal=False,
    )

    async def writer() -> None:
        await asyncio.sleep(0.02)
        log_file.write_text("partial", encoding="utf-8")
        await asyncio.sleep(0.03)
        exit_file.write_text("0", encoding="utf-8")

    write_task = asyncio.create_task(writer())
    events = [event async for event in runner._watch_task(meta=meta, timeout_sec=1)]
    await write_task

    stdout_events = [e for e in events if e.type == EventType.STDOUT]
    assert stdout_events
    assert any((e.content or "") for e in stdout_events)
    assert events[-1].type == EventType.EXITED


def test_read_new_text_recovers_after_log_truncation(tmp_path: Path) -> None:
    runner = TmuxRunner(data_dir=str(tmp_path))
    log_file = tmp_path / "truncated.log"
    log_file.write_text("new-output", encoding="utf-8")

    text, position = runner._read_new_text(log_file, position=100)

    assert text == "new-output"
    assert position == len(b"new-output")


@pytest.mark.asyncio
async def test_process_interactive_chunk_does_not_overwrite_structured_checkpoint(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01)
    meta = _TmuxTaskMeta(
        session_name="tgcli_user_1",
        log_file=runner._file_store.raw_transcript_path("tgcli_user_1"),
        exit_file=tmp_path / "x.exit",
        task_id="t1",
        workdir=str(tmp_path),
        terminal_id="user_1",
        persistent_terminal=True,
        interactive=True,
    )
    runner._session_store.get_or_create(
        session_id=meta.session_name,
        provider="claude_code",
        workdir=meta.workdir,
        terminal_id=meta.terminal_id,
    )
    state = runner._session_store.get(meta.session_name)
    assert state is not None
    state.checkpoint.last_offset = 7
    runner._session_store.save_checkpoint(meta.session_name, state.checkpoint)
    text = "TGCLI_BEGIN\n冒泡排序说明\nTGCLI_DONE\n"
    meta.log_file.write_text(text, encoding="utf-8")

    runner._process_interactive_chunk(
        meta=meta,
        offset=len(text.encode("utf-8")),
    )

    state = runner._session_store.get(meta.session_name)
    assert state is not None
    assert state.phase == SessionPhase.PROCESSING
    assert state.turns == []
    assert state.checkpoint.last_offset == 7

    raw_text = meta.log_file.read_text(encoding="utf-8")
    assert "冒泡排序说明" in raw_text
    session_files = {path.name for path in runner._file_store.session_dir(meta.session_name).iterdir()}
    assert "transcript.raw.log" in session_files
    assert "parser.cursor.json" in session_files
    assert "transcript.events.jsonl" not in session_files


@pytest.mark.asyncio
async def test_interactive_watch_starts_from_current_transcript_end(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    session_name = "tgcli_user_1"
    claude_session = "claude-session-1"
    log_file = runner._file_store.raw_transcript_path(session_name)
    old_text = "TGCLI_BEGIN\n旧回复\nTGCLI_DONE\n"
    new_text = "TGCLI_BEGIN\n新回复\nTGCLI_DONE\n"
    log_file.write_text(old_text, encoding="utf-8")

    runner._session_store.get_or_create(
        session_id=session_name,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_id="user_1",
    )
    bound = runner._session_store.get_or_create(
        session_id=claude_session,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_id="user_1",
    )
    bound.phase = SessionPhase.PROCESSING
    runner._session_store._persist(bound)

    meta = _TmuxTaskMeta(
        session_name=session_name,
        log_file=log_file,
        exit_file=tmp_path / "x.exit",
        task_id="t2",
        workdir=str(tmp_path),
        terminal_id="user_1",
        claude_session_id=claude_session,
        persistent_terminal=True,
        interactive=True,
    )

    async def append_and_complete() -> None:
        await asyncio.sleep(0.03)
        log_file.write_text(old_text + new_text, encoding="utf-8")
        bound = runner._session_store.get(claude_session)
        assert bound is not None
        bound.phase = SessionPhase.WAITING_FOR_INPUT
        bound.turns.append(
            ConversationTurn(
                turn_id="turn-new",
                role="assistant",
                text="\n新回复\n",
                is_complete=True,
                source="jsonl",
            )
        )
        bound.checkpoint.last_offset = 1
        runner._session_store._persist(bound)

    writer = asyncio.create_task(append_and_complete())
    events = [event async for event in runner._watch_task(meta=meta, timeout_sec=1)]
    await writer

    assert [event.type for event in events] == [EventType.EXITED]
    state = runner._session_store.get(claude_session)
    assert state is not None
    assert state.checkpoint.last_offset == 1


@pytest.mark.asyncio
async def test_interactive_timeout_keeps_session_alive(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="app.adapters.process.tmux_runner")
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    session_name = "tgcli_user_1"
    state = runner._session_store.get_or_create(session_id=session_name, workdir=str(tmp_path), terminal_id="user_1")
    state.phase = SessionPhase.PROCESSING
    runner._session_store._persist(state)

    meta = _TmuxTaskMeta(
        session_name=session_name,
        log_file=runner._file_store.raw_transcript_path(session_name),
        exit_file=tmp_path / "x.exit",
        task_id="t3",
        workdir=str(tmp_path),
        terminal_id="user_1",
        claude_session_id=session_name,
        persistent_terminal=True,
        interactive=True,
    )

    events = [event async for event in runner._watch_task(meta=meta, timeout_sec=0)]

    assert events[-1].type == EventType.TIMEOUT
    assert any(record.message == "tmux task timeout" for record in caplog.records)
    assert any(record.message == "tmux task finished" and record.result == "timeout" for record in caplog.records)
    state = runner._session_store.get(session_name)
    assert state is not None
    assert state.phase == SessionPhase.PROCESSING


@pytest.mark.asyncio
async def test_interactive_timeout_resets_after_structured_progress(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"
    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    state.phase = SessionPhase.PROCESSING
    runner._session_store._persist(state)

    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x-progress.exit",
        task_id="t3-progress",
        workdir=str(tmp_path),
        terminal_id="user_1",
        claude_session_id=claude_session,
        persistent_terminal=True,
        interactive=True,
    )

    async def update_progress_and_complete() -> None:
        await asyncio.sleep(0.05)
        state = runner._session_store.get(claude_session)
        assert state is not None
        state.checkpoint.last_offset = 1
        runner._session_store._persist(state)
        await asyncio.sleep(0.07)
        state.turns.append(
            ConversationTurn(
                turn_id="turn-progress",
                role="assistant",
                text="完成",
                is_complete=True,
                source="jsonl",
            )
        )
        state.phase = SessionPhase.WAITING_FOR_INPUT
        state.checkpoint.last_offset = 2
        runner._session_store._persist(state)

    writer = asyncio.create_task(update_progress_and_complete())
    events = [event async for event in runner._watch_task(meta=meta, timeout_sec=0.1)]
    await writer

    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_interactive_watch_uses_recent_explicit_session_over_stale_pending_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner_with_session_store(
        tmp_path,
        poll_interval_sec=0.01,
        partial_flush_sec=0.01,
        interactive_completion_grace_sec=0,
    )
    terminal_session = "tgcli_user_1"
    terminal_id = "user_1"
    current_session = "4550fd41-474b-463e-8db8-2435c71c2f10"
    stale_pending_session = "29766ba6-468c-484c-a759-7c440c2e2c75"
    now = utc_now()

    async def fake_interrupt(session_name: str) -> bool:
        return True

    monkeypatch.setattr(runner, "_interrupt_session", fake_interrupt)
    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id=terminal_id)

    stale_pending = runner._session_store.get_or_create(
        session_id=stale_pending_session,
        workdir=str(tmp_path),
        terminal_id=terminal_id,
    )
    stale_pending.phase = SessionPhase.WAITING_FOR_APPROVAL
    stale_pending.last_activity = now - timedelta(days=1)
    stale_pending.pending_permission = PendingPermission(
        tool_use_id="tool-stale",
        tool_name="Bash",
        tool_input={"command": "pwd"},
    )
    stale_pending.turns.append(
        ConversationTurn(
            turn_id="turn-stale",
            role="assistant",
            text="旧授权请求",
            is_complete=True,
            source="jsonl",
        )
    )
    runner._session_store._persist(stale_pending)

    current = runner._session_store.get_or_create(
        session_id=current_session,
        workdir=str(tmp_path),
        terminal_id=terminal_id,
    )
    current.phase = SessionPhase.WAITING_FOR_INPUT
    current.last_activity = now
    runner._session_store._persist(current)

    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x-stale-pending.exit",
        task_id="t-stale-pending",
        workdir=str(tmp_path),
        terminal_id=terminal_id,
        claude_session_id=current_session,
        persistent_terminal=True,
        interactive=True,
    )

    async def complete_current_session() -> None:
        await asyncio.sleep(0.03)
        active = runner._session_store.get(current_session)
        assert active is not None
        active.turns.append(
            ConversationTurn(
                turn_id="turn-current",
                role="assistant",
                text="当前回复",
                is_complete=True,
                source="jsonl",
            )
        )
        active.phase = SessionPhase.WAITING_FOR_INPUT
        active.checkpoint.last_offset = 2
        active.last_activity = utc_now()
        runner._session_store._persist(active)

    writer = asyncio.create_task(complete_current_session())
    events = [event async for event in runner._watch_task(meta=meta, timeout_sec=0.3)]
    await writer

    assert [event.type for event in events] == [EventType.EXITED]
    assert meta.claude_session_id == current_session


@pytest.mark.asyncio
async def test_interactive_completion_does_not_end_terminal_session_state(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"
    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)

    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x.exit",
        task_id="t3b",
        workdir=str(tmp_path),
        terminal_id="user_1",
        claude_session_id=claude_session,
        persistent_terminal=True,
        interactive=True,
    )

    task = asyncio.create_task(_collect_events(runner._watch_task(meta=meta, timeout_sec=1)))
    await asyncio.sleep(0.03)
    state = runner._session_store.get(claude_session)
    assert state is not None
    assert state.phase == SessionPhase.PROCESSING
    state.checkpoint.last_offset = 1
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)

    events = await task

    assert [event.type for event in events] == [EventType.EXITED]
    terminal_state = runner._session_store.get(terminal_session)
    assert terminal_state is not None
    assert terminal_state.phase != SessionPhase.ENDED
    state = runner._session_store.get(claude_session)
    assert state is not None
    assert state.phase == SessionPhase.WAITING_FOR_INPUT


@pytest.mark.asyncio
async def test_watch_task_follows_late_bound_claude_session_on_first_turn(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")

    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x1.exit",
        task_id="t4",
        workdir=str(tmp_path),
        terminal_id="user_1",
        persistent_terminal=True,
        interactive=True,
    )

    task = asyncio.create_task(_collect_events(runner._watch_task(meta=meta, timeout_sec=1)))
    await asyncio.sleep(0.03)
    bound = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    bound.phase = SessionPhase.PROCESSING
    runner._session_store._persist(bound)
    await asyncio.sleep(0.03)
    bound.turns.append(
        ConversationTurn(
            turn_id="turn-first",
            role="assistant",
            text="\n首轮回复\n",
            is_complete=True,
            source="jsonl",
        )
    )
    bound.checkpoint.last_offset = 1
    bound.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(bound)

    events = await task

    assert meta.claude_session_id == claude_session
    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_watch_task_follows_late_bound_uuid_claude_session_on_first_turn(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    claude_session = "2185ae1c-14e5-4423-8f0d-1b76fcd893d6"

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")

    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x1-uuid.exit",
        task_id="t4-uuid",
        workdir=str(tmp_path),
        terminal_id="user_1",
        persistent_terminal=True,
        interactive=True,
    )

    task = asyncio.create_task(_collect_events(runner._watch_task(meta=meta, timeout_sec=1)))
    await asyncio.sleep(0.03)
    bound = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    bound.phase = SessionPhase.PROCESSING
    runner._session_store._persist(bound)
    await asyncio.sleep(0.03)
    bound.turns.append(
        ConversationTurn(
            turn_id="turn-first-uuid",
            role="assistant",
            text="\n首轮 UUID 回复\n",
            is_complete=True,
            source="jsonl",
        )
    )
    bound.checkpoint.last_offset = 1
    bound.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(bound)

    events = await task

    assert meta.claude_session_id == claude_session
    assert [event.type for event in events] == [EventType.EXITED]


def test_capture_interactive_baseline_does_not_mark_captured_without_real_claude_session(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x-baseline.exit",
        task_id="t-baseline",
        workdir=str(tmp_path),
        terminal_id="user_1",
        persistent_terminal=True,
        interactive=True,
    )

    runner._capture_interactive_baseline(meta=meta)

    assert meta.baseline_captured is False
    assert meta.baseline_offset == 0
    assert meta.baseline_completed_turn_id is None


@pytest.mark.asyncio
async def test_watch_task_late_bound_existing_history_does_not_exit_until_new_turn(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x-late-history.exit",
        task_id="t-late-history",
        workdir=str(tmp_path),
        terminal_id="user_1",
        persistent_terminal=True,
        interactive=True,
    )

    old_turn_started_at = utc_now() - timedelta(minutes=1)
    task = asyncio.create_task(_collect_events(runner._watch_task(meta=meta, timeout_sec=1)))
    await asyncio.sleep(0.03)
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    state.phase = SessionPhase.WAITING_FOR_INPUT
    state.turns.append(
        ConversationTurn(
            turn_id="turn-old",
            role="assistant",
            text="\n旧回复\n",
            is_complete=True,
            source="jsonl",
            started_at=old_turn_started_at,
        )
    )
    state.checkpoint.last_offset = 10
    runner._session_store._persist(state)
    await asyncio.sleep(0.05)

    assert task.done() is False

    state.turns.append(
        ConversationTurn(
            turn_id="turn-new",
            role="assistant",
            text="\n新回复\n",
            is_complete=True,
            source="jsonl",
        )
    )
    state.checkpoint.last_offset = 20
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)

    events = await task

    assert meta.claude_session_id == claude_session
    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_watch_task_late_bound_completed_first_turn_exits_without_waiting_for_another_turn(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x-late-completed.exit",
        task_id="t-late-completed",
        workdir=str(tmp_path),
        terminal_id="user_1",
        persistent_terminal=True,
        interactive=True,
    )

    task = asyncio.create_task(_collect_events(runner._watch_task(meta=meta, timeout_sec=0.2)))
    await asyncio.sleep(0.03)
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    state.turns.append(
        ConversationTurn(
            turn_id="turn-first",
            role="assistant",
            text="\n首轮回复\n",
            is_complete=True,
            source="jsonl",
        )
    )
    state.checkpoint.last_offset = 10
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)

    events = await task

    assert meta.claude_session_id == claude_session
    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_watch_task_waits_for_stable_completion_candidate_before_exit(tmp_path: Path) -> None:
    runner = _runner_with_session_store(
        tmp_path,
        poll_interval_sec=0.01,
        partial_flush_sec=0.01,
        interactive_completion_grace_sec=0.5,
    )
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x-stable-completion.exit",
        task_id="t-stable-completion",
        workdir=str(tmp_path),
        terminal_id="user_1",
        persistent_terminal=True,
        interactive=True,
    )

    task = asyncio.create_task(_collect_events(runner._watch_task(meta=meta, timeout_sec=2)))
    await asyncio.sleep(0.03)
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    progress_turn_started_at = utc_now()
    state.turns.append(
        ConversationTurn(
            turn_id="turn-progress",
            role="assistant",
            text="\n我先查一下。\n",
            is_complete=True,
            source="jsonl",
            started_at=progress_turn_started_at,
            ended_at=progress_turn_started_at,
        )
    )
    state.checkpoint.last_offset = 10
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)
    await asyncio.sleep(0.05)

    assert task.done() is False

    state.tool_calls["tool-after-progress"] = ToolCallRecord(
        tool_use_id="tool-after-progress",
        name="Read",
        status=ToolStatus.SUCCESS,
        started_at=progress_turn_started_at + timedelta(milliseconds=1),
        completed_at=progress_turn_started_at + timedelta(milliseconds=2),
    )
    state.checkpoint.last_offset = 20
    runner._session_store._persist(state)
    await asyncio.sleep(0.08)

    assert task.done() is False

    final_turn_started_at = utc_now()
    state.turns.append(
        ConversationTurn(
            turn_id="turn-final",
            role="assistant",
            text="\n最终结果。\n",
            is_complete=True,
            source="jsonl",
            started_at=final_turn_started_at,
            ended_at=final_turn_started_at,
        )
    )
    state.checkpoint.last_offset = 30
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)

    events = await task

    assert meta.claude_session_id == claude_session
    assert [event.type for event in events] == [EventType.EXITED]


def test_interactive_completion_candidate_survives_revision_only_updates(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, interactive_completion_grace_sec=0.1)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"
    command_started_at = utc_now() - timedelta(seconds=1)
    turn_started_at = command_started_at + timedelta(milliseconds=100)

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    state.turns.append(
        ConversationTurn(
            turn_id="turn-current",
            role="assistant",
            text="\n你好！\n",
            is_complete=True,
            source="jsonl",
            started_at=turn_started_at,
            ended_at=turn_started_at,
        )
    )
    state.checkpoint.last_offset = 10
    state.phase = SessionPhase.WAITING_FOR_INPUT

    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x-revision-only.exit",
        task_id="t-revision-only",
        workdir=str(tmp_path),
        terminal_id="user_1",
        claude_session_id=claude_session,
        persistent_terminal=True,
        interactive=True,
        baseline_captured=True,
        baseline_offset=0,
        baseline_completed_turn_id=None,
        command_started_at=command_started_at,
    )
    watch_state = _InteractiveWatchState(
        watch_started_at=command_started_at,
        completion_started_after=command_started_at,
        structured_offset_before_run=0,
    )

    exit_code, _ = runner._tick_interactive_watch(meta=meta, watch_state=watch_state, now=1.0)
    assert exit_code is None

    state.revision += 1
    state.structured_reply_turn_id = "turn-current"
    exit_code, _ = runner._tick_interactive_watch(meta=meta, watch_state=watch_state, now=1.2)

    assert exit_code == 0


def test_first_text_completion_survives_revision_only_updates_before_baseline_capture(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, interactive_completion_grace_sec=0.1)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"
    command_started_at = utc_now() - timedelta(seconds=1)
    turn_started_at = command_started_at + timedelta(milliseconds=100)

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x-first-text-revision-only.exit",
        task_id="t-first-text-revision-only",
        workdir=str(tmp_path),
        terminal_id="user_1",
        persistent_terminal=True,
        interactive=True,
        baseline_captured=False,
        baseline_offset=0,
        baseline_completed_turn_id=None,
        command_started_at=command_started_at,
    )
    watch_state = _InteractiveWatchState(
        watch_started_at=command_started_at,
        completion_started_after=command_started_at,
        structured_offset_before_run=0,
    )

    exit_code, active_state = runner._tick_interactive_watch(meta=meta, watch_state=watch_state, now=1.0)
    assert exit_code is None
    assert active_state is None

    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    state.turns.append(
        ConversationTurn(
            turn_id="turn-first-text",
            role="assistant",
            text="\n首条文字回复\n",
            is_complete=True,
            source="jsonl",
            started_at=turn_started_at,
            ended_at=turn_started_at,
        )
    )
    state.checkpoint.last_offset = 10
    state.phase = SessionPhase.WAITING_FOR_INPUT

    exit_code, active_state = runner._tick_interactive_watch(meta=meta, watch_state=watch_state, now=1.1)
    assert exit_code is None
    assert active_state is state
    assert meta.claude_session_id == claude_session
    assert meta.baseline_captured is False

    state.revision += 1
    state.structured_reply_turn_id = "turn-first-text"
    exit_code, _ = runner._tick_interactive_watch(meta=meta, watch_state=watch_state, now=1.3)

    assert exit_code == 0


@pytest.mark.asyncio
async def test_watch_task_does_not_finish_on_progress_turn_followed_by_tool(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"
    command_started_at = utc_now() - timedelta(seconds=1)

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x-progress-tool.exit",
        task_id="t-progress-tool",
        workdir=str(tmp_path),
        terminal_id="user_1",
        persistent_terminal=True,
        interactive=True,
        command_started_at=command_started_at,
    )

    task = asyncio.create_task(_collect_events(runner._watch_task(meta=meta, timeout_sec=1)))
    await asyncio.sleep(0.03)
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    progress_turn_started_at = command_started_at + timedelta(milliseconds=100)
    state.turns.append(
        ConversationTurn(
            turn_id="turn-progress",
            role="assistant",
            text="\n我先查一下。\n",
            is_complete=True,
            source="jsonl",
            started_at=progress_turn_started_at,
            ended_at=progress_turn_started_at,
        )
    )
    state.tool_calls["tool-after-progress"] = ToolCallRecord(
        tool_use_id="tool-after-progress",
        name="Read",
        status=ToolStatus.SUCCESS,
        started_at=progress_turn_started_at,
        completed_at=progress_turn_started_at + timedelta(milliseconds=100),
    )
    state.checkpoint.last_offset = 10
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)
    await asyncio.sleep(0.05)

    assert task.done() is False

    final_turn_started_at = progress_turn_started_at + timedelta(seconds=1)
    state.turns.append(
        ConversationTurn(
            turn_id="turn-final",
            role="assistant",
            text="\n最终结果。\n",
            is_complete=True,
            source="jsonl",
            started_at=final_turn_started_at,
            ended_at=final_turn_started_at,
        )
    )
    state.checkpoint.last_offset = 20
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)

    events = await task

    assert meta.claude_session_id == claude_session
    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_watch_task_does_not_finish_while_permission_is_pending(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01, interactive_completion_grace_sec=0)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"
    command_started_at = utc_now() - timedelta(seconds=1)

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x-pending-permission.exit",
        task_id="t-pending-permission",
        workdir=str(tmp_path),
        terminal_id="user_1",
        persistent_terminal=True,
        interactive=True,
        command_started_at=command_started_at,
    )

    task = asyncio.create_task(_collect_events(runner._watch_task(meta=meta, timeout_sec=1)))
    await asyncio.sleep(0.03)
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    progress_turn_started_at = command_started_at + timedelta(milliseconds=100)
    state.turns.append(
        ConversationTurn(
            turn_id="turn-progress",
            role="assistant",
            text="\n需要执行命令。\n",
            is_complete=True,
            source="jsonl",
            started_at=progress_turn_started_at,
            ended_at=progress_turn_started_at,
        )
    )
    state.tool_calls["tool-pending"] = ToolCallRecord(
        tool_use_id="tool-pending",
        name="Bash",
        status=ToolStatus.WAITING_FOR_APPROVAL,
        started_at=progress_turn_started_at + timedelta(milliseconds=1),
    )
    state.pending_permission = PendingPermission(tool_use_id="tool-pending", tool_name="Bash", tool_input={"command": "pwd"})
    state.checkpoint.last_offset = 10
    state.phase = SessionPhase.WAITING_FOR_APPROVAL
    runner._session_store._persist(state)
    await asyncio.sleep(0.05)

    assert task.done() is False

    state.pending_permission = None
    state.tool_calls["tool-pending"].status = ToolStatus.SUCCESS
    state.tool_calls["tool-pending"].completed_at = progress_turn_started_at + timedelta(milliseconds=200)
    final_turn_started_at = progress_turn_started_at + timedelta(seconds=1)
    state.turns.append(
        ConversationTurn(
            turn_id="turn-final",
            role="assistant",
            text="\n命令执行完了。\n",
            is_complete=True,
            source="jsonl",
            started_at=final_turn_started_at,
            ended_at=final_turn_started_at,
        )
    )
    state.checkpoint.last_offset = 20
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)

    events = await task

    assert meta.claude_session_id == claude_session
    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_watch_task_ignores_delayed_old_completed_turn_after_captured_baseline(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"
    command_started_at = utc_now()

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    state.phase = SessionPhase.PROCESSING
    runner._session_store._persist(state)

    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x-delayed-old-turn.exit",
        task_id="t-delayed-old-turn",
        workdir=str(tmp_path),
        terminal_id="user_1",
        claude_session_id=claude_session,
        persistent_terminal=True,
        interactive=True,
        baseline_captured=True,
        baseline_offset=0,
        baseline_completed_turn_id=None,
        command_started_at=command_started_at,
    )

    task = asyncio.create_task(_collect_events(runner._watch_task(meta=meta, timeout_sec=1)))
    await asyncio.sleep(0.03)
    state.turns.append(
        ConversationTurn(
            turn_id="turn-delayed-old",
            role="assistant",
            text="\n上一轮延迟回复\n",
            is_complete=True,
            source="jsonl",
            started_at=command_started_at - timedelta(seconds=1),
        )
    )
    state.checkpoint.last_offset = 10
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)
    await asyncio.sleep(0.05)

    assert task.done() is False

    state.turns.append(
        ConversationTurn(
            turn_id="turn-current",
            role="assistant",
            text="\n本轮回复\n",
            is_complete=True,
            source="jsonl",
            started_at=command_started_at + timedelta(seconds=1),
        )
    )
    state.checkpoint.last_offset = 20
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)

    events = await task

    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_watch_task_bound_completed_turn_started_after_command_before_watch_exits(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"
    command_started_at = utc_now() - timedelta(seconds=1)

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    state.turns.append(
        ConversationTurn(
            turn_id="turn-first",
            role="assistant",
            text="\n首轮回复\n",
            is_complete=True,
            source="jsonl",
            started_at=command_started_at + timedelta(milliseconds=100),
        )
    )
    state.checkpoint.last_offset = 10
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)

    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x-bound-prewatch-completed.exit",
        task_id="t-bound-prewatch-completed",
        workdir=str(tmp_path),
        terminal_id="user_1",
        claude_session_id=claude_session,
        persistent_terminal=True,
        interactive=True,
        command_started_at=command_started_at,
    )

    events = await _collect_events(runner._watch_task(meta=meta, timeout_sec=0.2))

    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_watch_task_late_bound_completed_turn_started_after_command_before_watch_exits(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"
    command_started_at = utc_now() - timedelta(seconds=1)

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x-late-prewatch-completed.exit",
        task_id="t-late-prewatch-completed",
        workdir=str(tmp_path),
        terminal_id="user_1",
        persistent_terminal=True,
        interactive=True,
    )
    meta.command_started_at = command_started_at

    task = asyncio.create_task(_collect_events(runner._watch_task(meta=meta, timeout_sec=0.2)))
    await asyncio.sleep(0.03)
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    state.turns.append(
        ConversationTurn(
            turn_id="turn-first",
            role="assistant",
            text="\n首轮回复\n",
            is_complete=True,
            source="jsonl",
            started_at=command_started_at + timedelta(milliseconds=100),
        )
    )
    state.checkpoint.last_offset = 10
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)

    events = await task

    assert meta.claude_session_id == claude_session
    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_watch_task_switches_from_stale_explicit_session_to_newer_bound_session(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    old_session = "2185ae1c-14e5-4423-8f0d-1b76fcd893d6"
    new_session = "f5bc22fa-0e77-42f6-a2d3-e422037296f6"

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    old_state = runner._session_store.get_or_create(session_id=old_session, workdir=str(tmp_path), terminal_id="user_1")
    old_state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(old_state)

    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x1-switch.exit",
        task_id="t4-switch",
        workdir=str(tmp_path),
        terminal_id="user_1",
        claude_session_id=old_session,
        persistent_terminal=True,
        interactive=True,
    )

    task = asyncio.create_task(_collect_events(runner._watch_task(meta=meta, timeout_sec=1)))
    await asyncio.sleep(0.03)
    new_state = runner._session_store.get_or_create(session_id=new_session, workdir=str(tmp_path), terminal_id="user_1")
    new_state.turns.append(
        ConversationTurn(
            turn_id="turn-switch",
            role="assistant",
            text="\n切换后的新回复\n",
            is_complete=True,
            source="jsonl",
        )
    )
    new_state.checkpoint.last_offset = 1
    new_state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(new_state)

    events = await task

    assert meta.claude_session_id == new_session
    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_watch_task_uses_bound_claude_session_for_second_turn(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    state.phase = SessionPhase.WAITING_FOR_INPUT
    state.turns.append(
        ConversationTurn(
            turn_id="turn-old",
            role="assistant",
            text="\n旧回复\n",
            is_complete=True,
            source="jsonl",
        )
    )
    state.checkpoint.last_offset = 10
    runner._session_store._persist(state)

    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x2.exit",
        task_id="t5",
        workdir=str(tmp_path),
        terminal_id="user_1",
        claude_session_id=claude_session,
        persistent_terminal=True,
        interactive=True,
    )

    task = asyncio.create_task(_collect_events(runner._watch_task(meta=meta, timeout_sec=1)))
    await asyncio.sleep(0.03)
    state = runner._session_store.get(claude_session)
    assert state is not None
    assert state.phase == SessionPhase.PROCESSING
    await asyncio.sleep(0.03)
    state.turns.append(
        ConversationTurn(
            turn_id="turn-new",
            role="assistant",
            text="\n第二轮回复\n",
            is_complete=True,
            source="jsonl",
        )
    )
    state.checkpoint.last_offset = 20
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)

    events = await task

    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_watch_task_exits_when_fast_reply_arrives_before_watch_baseline(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    state.phase = SessionPhase.WAITING_FOR_INPUT
    state.checkpoint.last_offset = 1
    state.turns.append(
        ConversationTurn(
            turn_id="turn-fast",
            role="assistant",
            text="\n很快返回的新回复\n",
            is_complete=True,
            source="jsonl",
        )
    )
    runner._session_store._persist(state)

    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x-fast.exit",
        task_id="t-fast",
        workdir=str(tmp_path),
        terminal_id="user_1",
        claude_session_id=claude_session,
        persistent_terminal=True,
        interactive=True,
        baseline_captured=True,
        baseline_offset=0,
        baseline_completed_turn_id=None,
    )

    events = await _collect_events(runner._watch_task(meta=meta, timeout_sec=1))

    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_watch_task_does_not_exit_on_stale_waiting_phase_without_new_progress(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    state.phase = SessionPhase.WAITING_FOR_INPUT
    state.turns.append(
        ConversationTurn(
            turn_id="turn-old",
            role="assistant",
            text="\n旧回复\n",
            is_complete=True,
            source="jsonl",
        )
    )
    state.checkpoint.last_offset = 10
    runner._session_store._persist(state)

    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x2-stale.exit",
        task_id="t5-stale",
        workdir=str(tmp_path),
        terminal_id="user_1",
        claude_session_id=claude_session,
        persistent_terminal=True,
        interactive=True,
    )

    task = asyncio.create_task(_collect_events(runner._watch_task(meta=meta, timeout_sec=1)))
    await asyncio.sleep(0.03)
    state = runner._session_store.get(claude_session)
    assert state is not None
    assert state.phase == SessionPhase.PROCESSING

    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)
    await asyncio.sleep(0.05)
    assert task.done() is False

    state.turns.append(
        ConversationTurn(
            turn_id="turn-new",
            role="assistant",
            text="\n真正的新回复\n",
            is_complete=True,
            source="jsonl",
        )
    )
    state.checkpoint.last_offset = 20
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)

    events = await task

    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_watch_task_exits_when_new_completed_turn_arrives_without_waiting_hook(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    state.phase = SessionPhase.WAITING_FOR_INPUT
    state.turns = []
    runner._session_store._persist(state)

    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "x3.exit",
        task_id="t6",
        workdir=str(tmp_path),
        terminal_id="user_1",
        claude_session_id=claude_session,
        persistent_terminal=True,
        interactive=True,
    )

    task = asyncio.create_task(_collect_events(runner._watch_task(meta=meta, timeout_sec=1)))
    await asyncio.sleep(0.03)
    state = runner._session_store.get(claude_session)
    assert state is not None
    assert state.phase == SessionPhase.PROCESSING
    state.turns.append(
        ConversationTurn(
            turn_id="turn-new",
            role="assistant",
            text="\n补同步后的新回复\n",
            is_complete=True,
            source="jsonl",
        )
    )
    runner._session_store._persist(state)

    events = await task

    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_interactive_idle_pane_does_not_cancel_without_interrupt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    runner._interactive_idle_check_sec = 0.01
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)

    idle_checks = 0

    async def fake_idle(session_name: str) -> bool:
        nonlocal idle_checks
        assert session_name == terminal_session
        idle_checks += 1
        return True

    monkeypatch.setattr(runner, "_is_claude_idle_in_pane", fake_idle)

    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "idle.exit",
        task_id="t-idle",
        workdir=str(tmp_path),
        terminal_id="user_1",
        claude_session_id=claude_session,
        persistent_terminal=True,
        interactive=True,
    )

    task = asyncio.create_task(_collect_events(runner._watch_task(meta=meta, timeout_sec=1)))
    await asyncio.sleep(0.04)
    assert task.done() is False
    assert meta.cancel_requested is False

    state = runner._session_store.get(claude_session)
    assert state is not None
    state.turns.append(
        ConversationTurn(
            turn_id="turn-new",
            role="assistant",
            text="正常完成后的回复",
            is_complete=True,
            source="jsonl",
        )
    )
    state.checkpoint.last_offset = 20
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)

    events = await task

    assert idle_checks > 0
    assert meta.cancel_requested is False
    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_interactive_interrupted_state_marks_task_canceled(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, partial_flush_sec=0.01)
    terminal_session = "tgcli_user_1"
    claude_session = "claude-session-1"

    runner._session_store.get_or_create(session_id=terminal_session, workdir=str(tmp_path), terminal_id="user_1")
    state = runner._session_store.get_or_create(session_id=claude_session, workdir=str(tmp_path), terminal_id="user_1")
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)

    meta = _TmuxTaskMeta(
        session_name=terminal_session,
        log_file=runner._file_store.raw_transcript_path(terminal_session),
        exit_file=tmp_path / "interrupted.exit",
        task_id="t-interrupted",
        workdir=str(tmp_path),
        terminal_id="user_1",
        claude_session_id=claude_session,
        persistent_terminal=True,
        interactive=True,
    )

    task = asyncio.create_task(_collect_events(runner._watch_task(meta=meta, timeout_sec=1)))
    await asyncio.sleep(0.03)
    state = runner._session_store.get(claude_session)
    assert state is not None
    state.interrupted = True
    state.phase = SessionPhase.WAITING_FOR_INPUT
    runner._session_store._persist(state)

    events = await task

    assert meta.cancel_requested is True
    assert [event.type for event in events] == [EventType.CANCELED]


class _FakeFifoStdout:
    def __init__(self, lines: list[bytes | None]) -> None:
        self._lines = list(lines)

    async def read(self, size: int) -> bytes:
        _ = size
        if not self._lines:
            await asyncio.sleep(10)
            return b""
        line = self._lines.pop(0)
        if line is None:
            await asyncio.sleep(10)
            return b""
        return line


class _FakeFifoReader:
    _process = None

    def __init__(self, lines: list[bytes | None]) -> None:
        self._stdout = _FakeFifoStdout(lines)

    async def readlines(self) -> _FakeFifoStdout:
        return self._stdout


class _FakeChunkFifoStdout:
    def __init__(self, chunks: list[bytes | None]) -> None:
        self._chunks = list(chunks)

    async def read(self, size: int) -> bytes:
        _ = size
        if not self._chunks:
            await asyncio.sleep(10)
            return b""
        chunk = self._chunks.pop(0)
        if chunk is None:
            await asyncio.sleep(10)
            return b""
        return chunk

    async def readline(self) -> bytes:
        raise ValueError("Separator is not found, and chunk exceed the limit")


class _FakeChunkFifoReader:
    _process = None

    def __init__(self, chunks: list[bytes | None]) -> None:
        self._stdout = _FakeChunkFifoStdout(chunks)

    async def readlines(self) -> _FakeChunkFifoStdout:
        return self._stdout


@pytest.mark.asyncio
async def test_watch_interactive_fifo_timeout_after_output_polls_tick_instead_of_eof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner_with_session_store(tmp_path, interactive_completion_grace_sec=0)
    meta = _TmuxTaskMeta(
        session_name="tgcli_user_1",
        log_file=tmp_path / "fifo-timeout.log",
        exit_file=tmp_path / "fifo-timeout.exit",
        task_id="t-fifo-timeout",
        workdir=str(tmp_path),
        terminal_id="user_1",
        persistent_terminal=True,
        interactive=True,
    )
    calls = 0

    async def fake_tick(**kwargs):
        nonlocal calls
        calls += 1
        return (0 if calls == 2 else None), kwargs["timeout_anchor"]

    monkeypatch.setattr(runner, "_process_interactive_chunk", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_handle_interactive_tick", fake_tick)

    events = await _collect_events(
        runner._watch_interactive_fifo(meta=meta, timeout_sec=0.01, fifo_reader=_FakeFifoReader([b"output\n", None]))
    )

    assert calls == 2
    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_watch_interactive_fifo_eof_runs_final_tick_before_finalize(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner_with_session_store(tmp_path, interactive_completion_grace_sec=0)
    meta = _TmuxTaskMeta(
        session_name="tgcli_user_1",
        log_file=tmp_path / "fifo-eof.log",
        exit_file=tmp_path / "fifo-eof.exit",
        task_id="t-fifo-eof",
        workdir=str(tmp_path),
        terminal_id="user_1",
        persistent_terminal=True,
        interactive=True,
    )
    calls = 0

    async def fake_tick(**kwargs):
        nonlocal calls
        calls += 1
        return (0 if calls == 2 else None), kwargs["timeout_anchor"]

    monkeypatch.setattr(runner, "_process_interactive_chunk", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_handle_interactive_tick", fake_tick)

    events = await _collect_events(
        runner._watch_interactive_fifo(meta=meta, timeout_sec=1, fifo_reader=_FakeFifoReader([b"output\n", b""]))
    )

    assert calls == 2
    assert [event.type for event in events] == [EventType.EXITED]


@pytest.mark.asyncio
async def test_watch_interactive_fifo_reads_long_chunk_without_line_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner_with_session_store(tmp_path, interactive_completion_grace_sec=0)
    meta = _TmuxTaskMeta(
        session_name="tgcli_user_1",
        log_file=tmp_path / "fifo-long-chunk.log",
        exit_file=tmp_path / "fifo-long-chunk.exit",
        task_id="t-fifo-long-chunk",
        workdir=str(tmp_path),
        terminal_id="user_1",
        persistent_terminal=True,
        interactive=True,
    )
    chunk = b"x" * 70_000
    offsets: list[int] = []

    async def fake_tick(**kwargs):
        return 0, kwargs["timeout_anchor"]

    monkeypatch.setattr(runner, "_process_interactive_chunk", lambda **kwargs: offsets.append(kwargs["offset"]))
    monkeypatch.setattr(runner, "_handle_interactive_tick", fake_tick)

    events = await _collect_events(runner._watch_interactive_fifo(meta=meta, timeout_sec=1, fifo_reader=_FakeChunkFifoReader([chunk])))

    assert offsets == [len(chunk)]
    assert [event.type for event in events] == [EventType.EXITED]


def test_finalize_watch_result_unknown_exit_code_message_does_not_render_none(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path)
    meta = _TmuxTaskMeta(
        session_name="tgcli_user_1",
        log_file=tmp_path / "unknown.log",
        exit_file=tmp_path / "unknown.exit",
        task_id="t-unknown-exit",
        workdir=str(tmp_path),
        persistent_terminal=False,
    )

    event = runner._finalize_watch_result(
        meta=meta,
        timed_out=False,
        canceled=False,
        exit_code=None,
        timeout_sec=1,
        started_at=1.0,
        finished_at=1.1,
        position=0,
    )

    assert event.type == EventType.FAILED
    assert event.exit_code is None
    assert event.error is not None
    assert "None" not in event.error
    assert "未能读取退出码" in event.error


@pytest.mark.asyncio
async def test_watch_task_retries_empty_exit_file_before_finalize(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, exit_file_read_grace_sec=0.5)
    meta = _TmuxTaskMeta(
        session_name="tgcli_task_empty_exit",
        log_file=tmp_path / "empty-exit.log",
        exit_file=tmp_path / "empty-exit.exit",
        task_id="t-empty-exit",
        workdir=str(tmp_path),
        persistent_terminal=False,
    )
    meta.exit_file.write_text("", encoding="utf-8")

    async def writer() -> None:
        await asyncio.sleep(0.03)
        meta.exit_file.write_text("0", encoding="utf-8")

    write_task = asyncio.create_task(writer())
    events = await _collect_events(runner._watch_task(meta=meta, timeout_sec=1))
    await write_task

    assert events[-1].type == EventType.EXITED


@pytest.mark.asyncio
async def test_cancel_waits_for_pane_io_lock_before_interrupt(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path)
    session_name = "tgcli_user_1"
    meta = _TmuxTaskMeta(
        session_name=session_name,
        log_file=runner._file_store.raw_transcript_path(session_name),
        exit_file=tmp_path / "x-cancel.exit",
        task_id="t-cancel-lock",
        workdir=str(tmp_path),
        terminal_id="user_1",
        persistent_terminal=True,
        interactive=True,
        command_started_at=utc_now(),
    )
    runner._tasks["t-cancel-lock"] = meta

    interrupt_calls: list[str] = []

    async def fake_interrupt(name: str) -> bool:
        interrupt_calls.append(name)
        return True

    runner._interrupt_session = fake_interrupt  # type: ignore[method-assign]

    # 模拟外部正在持有 pane_io_lock 进行输入
    async with runner._pane_io_lock(session_name):
        cancel_task = asyncio.create_task(runner.cancel("t-cancel-lock"))
        await asyncio.sleep(0.05)
        # 在释放锁之前，interrupt 尚未被调用
        assert interrupt_calls == []

    await cancel_task
    # 释放锁之后，interrupt 被顺利调用
    assert interrupt_calls == [session_name]


async def test_watch_interrupt_paths_wait_for_pane_io_lock(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path)
    session_name = "tgcli_user_1"
    now = utc_now()
    meta = _TmuxTaskMeta(
        session_name=session_name,
        log_file=runner._file_store.raw_transcript_path(session_name),
        exit_file=tmp_path / "x-watch-cancel.exit",
        task_id="t-watch-cancel-lock",
        workdir=str(tmp_path),
        terminal_id="user_1",
        persistent_terminal=True,
        interactive=True,
        command_started_at=now,
    )
    watch_state = _InteractiveWatchState(watch_started_at=now, completion_started_after=now)
    interrupt_calls: list[str] = []

    async def fake_interrupt(name: str) -> bool:
        interrupt_calls.append(name)
        return True

    runner._interrupt_session = fake_interrupt  # type: ignore[method-assign]

    async with runner._pane_io_lock(session_name):
        cancel_task = asyncio.create_task(runner._handle_watch_cancel(meta=meta, started_at=0.0, now=1.0, position=0))
        timeout_task = asyncio.create_task(
            runner._handle_watch_timeout(
                meta=meta,
                timeout_sec=1,
                started_at=0.0,
                now=1.0,
                timeout_started_at=0.0,
                position=0,
                watch_state=watch_state,
            )
        )
        await asyncio.sleep(0.05)
        assert interrupt_calls == []

    await cancel_task
    await timeout_task
    assert interrupt_calls == [session_name, session_name]


@pytest.mark.asyncio
async def test_watch_task_invalid_exit_file_reports_unknown_exit_code_without_none(tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, poll_interval_sec=0.01, exit_file_read_grace_sec=0)
    meta = _TmuxTaskMeta(
        session_name="tgcli_task_invalid_exit",
        log_file=tmp_path / "invalid-exit.log",
        exit_file=tmp_path / "invalid-exit.exit",
        task_id="t-invalid-exit",
        workdir=str(tmp_path),
        persistent_terminal=False,
    )
    meta.exit_file.write_text("not-an-int", encoding="utf-8")

    events = await _collect_events(runner._watch_task(meta=meta, timeout_sec=1))

    assert events[-1].type == EventType.FAILED
    assert events[-1].error is not None
    assert "None" not in events[-1].error
    assert "未能读取退出码" in events[-1].error


@pytest.mark.asyncio
async def test_cancel_returns_false_for_unknown_task() -> None:
    runner = TmuxRunner()
    canceled = await runner.cancel("not-found")
    assert canceled is False


@pytest.mark.asyncio
async def test_run_task_terminates_ephemeral_session_when_watch_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, cancel_grace_sec=0)
    terminated: list[str] = []

    async def fake_start(*args, **kwargs):
        return True, ""

    async def fake_watch(*, meta, timeout_sec: int, fifo_reader=None):
        raise RuntimeError("watch failed")
        yield

    async def fake_terminate(session_name: str) -> bool:
        terminated.append(session_name)
        return True

    monkeypatch.setattr(runner, "_start_ephemeral_session", fake_start)
    monkeypatch.setattr(runner, "_watch_task", fake_watch)
    monkeypatch.setattr(runner, "_terminate_session", fake_terminate)

    meta = _TmuxTaskMeta(
        session_name="tgcli_task_1",
        log_file=tmp_path / "task.log",
        exit_file=tmp_path / "task.exit",
        task_id="task-1",
        workdir=str(tmp_path),
        command_file=tmp_path / "task.cmd.sh",
        persistent_terminal=False,
    )

    with pytest.raises(RuntimeError, match="watch failed"):
        await _collect_events(runner._run_task(meta=meta, timeout_sec=1, env=None, workdir=str(tmp_path), command="echo hi"))

    assert terminated == ["tgcli_task_1"]


@pytest.mark.asyncio
async def test_run_task_terminates_ephemeral_session_when_generator_is_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, cancel_grace_sec=0)
    terminated: list[str] = []

    async def fake_start(*args, **kwargs):
        return True, ""

    async def fake_watch(*, meta, timeout_sec: int, fifo_reader=None):
        await asyncio.sleep(10)
        yield CLIEvent(type=EventType.EXITED, task_id=meta.task_id, exit_code=0)

    async def fake_terminate(session_name: str) -> bool:
        terminated.append(session_name)
        return True

    monkeypatch.setattr(runner, "_start_ephemeral_session", fake_start)
    monkeypatch.setattr(runner, "_watch_task", fake_watch)
    monkeypatch.setattr(runner, "_terminate_session", fake_terminate)

    meta = _TmuxTaskMeta(
        session_name="tgcli_task_close",
        log_file=tmp_path / "close.log",
        exit_file=tmp_path / "close.exit",
        task_id="task-close",
        workdir=str(tmp_path),
        command_file=tmp_path / "close.cmd.sh",
        persistent_terminal=False,
    )
    stream = runner._run_task(meta=meta, timeout_sec=1, env=None, workdir=str(tmp_path), command="echo hi")

    first_event = await anext(stream)
    await stream.aclose()

    assert first_event.type == EventType.STARTED
    assert terminated == ["tgcli_task_close"]


@pytest.mark.asyncio
async def test_run_task_does_not_terminate_persistent_session_when_watch_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner_with_session_store(tmp_path, cancel_grace_sec=0)
    terminated: list[str] = []

    async def fake_ensure(*args, **kwargs):
        return True, ""

    async def fake_send(*args, **kwargs):
        return True, ""

    async def fake_watch(*, meta, timeout_sec: int, fifo_reader=None):
        raise RuntimeError("watch failed")
        yield

    async def fake_terminate(session_name: str) -> bool:
        terminated.append(session_name)
        return True

    monkeypatch.setattr(runner, "_ensure_persistent_session", fake_ensure)
    monkeypatch.setattr(runner, "_send_command", fake_send)
    monkeypatch.setattr(runner, "_watch_task", fake_watch)
    monkeypatch.setattr(runner, "_terminate_session", fake_terminate)

    meta = _TmuxTaskMeta(
        session_name="tgcli_user_1",
        log_file=tmp_path / "persistent.log",
        exit_file=tmp_path / "persistent.exit",
        task_id="task-persistent",
        workdir=str(tmp_path),
        command_file=tmp_path / "persistent.cmd.sh",
        persistent_terminal=True,
    )

    with pytest.raises(RuntimeError, match="watch failed"):
        await _collect_events(runner._run_task(meta=meta, timeout_sec=1, env=None, workdir=str(tmp_path), command="echo hi"))

    assert terminated == []


@pytest.mark.asyncio
async def test_reveal_terminal_reports_session_check_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner(cancel_grace_sec=0)

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        if args and args[0] == "has-session":
            return 1, "", "permission denied"
        return 0, "", ""

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)

    opened, text = await runner.reveal_terminal("user_1")

    assert opened is False
    assert text == "终端状态检查失败: permission denied"


@pytest.mark.asyncio
async def test_reveal_terminal_opens_when_osascript_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tmux_bin_path = tmp_path / "custom tmux/bin/tmux"
    runner = TmuxRunner(tmux_bin=str(tmux_bin_path), cancel_grace_sec=0)
    tmux_calls: list[tuple[str, ...]] = []
    subprocess_calls: list[tuple[object, ...]] = []

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        tmux_calls.append(args)
        assert args[0] == "has-session"
        return 0, "", ""

    async def fake_create_subprocess_exec(*args: object, **_kwargs: object) -> FakeSubprocessProcess:
        subprocess_calls.append(args)
        return FakeSubprocessProcess(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)
    monkeypatch.setattr(tmux_runner_module.shutil, "which", lambda _bin: str(tmux_bin_path))
    monkeypatch.setattr(tmux_runner_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    opened, text = await runner.reveal_terminal("user_1")

    assert opened is True
    assert text == "已在桌面打开 Terminal 并附着到 tgcli_user_1"
    assert tmux_calls == [("has-session", "-t", "tgcli_user_1")]
    assert len(subprocess_calls) == 1
    osascript_args = subprocess_calls[0]
    assert osascript_args[:2] == ("osascript", "-e")
    script = osascript_args[2]
    assert isinstance(script, str)
    assert 'set targetTab to do script ""' in script
    assert "delay 0.2" in script
    assert "repeat with attempt from 1 to 100" in script
    assert 'if busy of targetTab then error "Terminal shell did not become ready"' in script
    assert "do script (item 1 of argv) in targetTab" in script
    assert "return tty of targetTab" not in script
    attach_command = osascript_args[4]
    parts = shlex.split(attach_command)
    assert parts == [
        "/usr/bin/env",
        "-u",
        "TMUX",
        "-u",
        "TMUX_PANE",
        str(tmux_bin_path),
        "attach-session",
        "-t",
        "tgcli_user_1",
    ]


@pytest.mark.asyncio
async def test_reveal_terminal_reports_osascript_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner(cancel_grace_sec=0)

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        assert args[0] == "has-session"
        return 0, "", ""

    async def fake_create_subprocess_exec(*_args: object, **_kwargs: object) -> FakeSubprocessProcess:
        return FakeSubprocessProcess(returncode=1, stderr=b"Terminal shell did not become ready")

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)
    monkeypatch.setattr(tmux_runner_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    opened, text = await runner.reveal_terminal("user_1")

    assert opened is False
    assert text == ("打开桌面终端失败: Terminal shell did not become ready\nhint: 可手动执行 `tmux attach -t tgcli_user_1`")


@pytest.mark.asyncio
async def test_reveal_terminal_reports_osascript_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner(cancel_grace_sec=0)

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        assert args[0] == "has-session"
        return 0, "", ""

    async def fake_create_subprocess_exec(*_args: object, **_kwargs: object) -> FakeSubprocessProcess:
        raise FileNotFoundError("osascript")

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)
    monkeypatch.setattr(tmux_runner_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    opened, text = await runner.reveal_terminal("user_1")

    assert opened is False
    assert text == "找不到 osascript（仅支持 macOS 桌面）\nhint: 可手动执行 `tmux attach -t <tmux_session>`"


@pytest.mark.asyncio
async def test_reveal_terminal_reports_osascript_start_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner(cancel_grace_sec=0)

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        assert args[0] == "has-session"
        return 0, "", ""

    async def fake_create_subprocess_exec(*_args: object, **_kwargs: object) -> FakeSubprocessProcess:
        raise OSError("spawn blocked")

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)
    monkeypatch.setattr(tmux_runner_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    opened, text = await runner.reveal_terminal("user_1")

    assert opened is False
    assert text == "打开桌面终端失败: spawn blocked"


@pytest.mark.asyncio
async def test_reveal_terminal_falls_back_to_resolved_tmux_bin_when_which_misses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tmux_bin_path = tmp_path / "bundled/tmux"
    tmux_bin_path.parent.mkdir(parents=True, exist_ok=True)
    tmux_bin_path.write_text("#!/bin/sh\n")
    runner = TmuxRunner(tmux_bin=str(tmux_bin_path), cancel_grace_sec=0)

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        assert args[0] == "has-session"
        return 0, "", ""

    async def fake_create_subprocess_exec(*args: object, **_kwargs: object) -> FakeSubprocessProcess:
        attach_command = args[4]
        parts = shlex.split(attach_command)
        assert parts[5] == str(tmux_bin_path)
        return FakeSubprocessProcess(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)
    monkeypatch.setattr(tmux_runner_module.shutil, "which", lambda _bin: None)
    monkeypatch.setattr(tmux_runner_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    opened, text = await runner.reveal_terminal("user_1")

    assert opened is True
    assert text == "已在桌面打开 Terminal 并附着到 tgcli_user_1"


@pytest.mark.asyncio
async def test_persistent_run_registers_before_ensure_so_close_can_cancel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner_with_session_store(tmp_path, cancel_grace_sec=0)
    ensure_entered = asyncio.Event()
    release_ensure = asyncio.Event()
    calls: list[str] = []

    async def fake_ensure(*args, **kwargs):
        calls.append("ensure")
        ensure_entered.set()
        await release_ensure.wait()
        return True, ""

    async def fake_send(*args, **kwargs):
        calls.append("send")
        return True, ""

    async def fake_session_exists(session_name: str) -> bool:
        calls.append("exists")
        return True

    async def fake_terminate(session_name: str) -> bool:
        calls.append("terminate")
        return True

    monkeypatch.setattr(runner, "_ensure_persistent_session", fake_ensure)
    monkeypatch.setattr(runner, "_send_command", fake_send)
    monkeypatch.setattr(runner, "session_exists", fake_session_exists)
    monkeypatch.setattr(runner, "_terminate_session", fake_terminate)

    run_task = asyncio.create_task(
        _collect_events(
            runner.run(
                task_id="task-close-race",
                argv=["echo", "ok"],
                workdir=str(tmp_path),
                timeout_sec=10,
                terminal_key="user_1",
            )
        )
    )
    await asyncio.wait_for(ensure_entered.wait(), timeout=1)

    close_task = asyncio.create_task(runner.close_terminal("user_1"))
    await asyncio.sleep(0)

    meta = runner._tasks["task-close-race"]
    assert meta.cancel_requested is True
    assert "exists" not in calls
    assert "terminate" not in calls

    release_ensure.set()
    events = await asyncio.wait_for(run_task, timeout=1)
    closed, text = await asyncio.wait_for(close_task, timeout=1)

    assert [event.type for event in events] == [EventType.CANCELED]
    assert closed is True
    assert text == ""
    assert "send" not in calls
    assert calls == ["ensure", "exists", "terminate"]


@pytest.mark.asyncio
async def test_close_terminal_waits_for_session_lock_after_marking_cancel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = TmuxRunner(cancel_grace_sec=0)
    session_name = runner.build_session_name("user_1")
    meta = _TmuxTaskMeta(
        session_name=session_name,
        log_file=tmp_path / "close.log",
        exit_file=tmp_path / "close.exit",
        task_id="task-close",
        workdir=str(tmp_path),
        persistent_terminal=True,
    )
    runner._tasks[meta.task_id] = meta
    calls: list[str] = []

    async def fake_session_exists(name: str) -> bool:
        calls.append(f"exists:{name}")
        return True

    async def fake_terminate(name: str) -> bool:
        calls.append(f"terminate:{name}")
        return True

    monkeypatch.setattr(runner, "session_exists", fake_session_exists)
    monkeypatch.setattr(runner, "_terminate_session", fake_terminate)

    held = runner._session_lock(session_name)
    await held.__aenter__()
    try:
        task = asyncio.create_task(runner.close_terminal("user_1"))
        await asyncio.sleep(0)
        assert meta.cancel_requested is True
        assert calls == []

        await held.__aexit__(None, None, None)
        closed, text = await asyncio.wait_for(task, timeout=1)
    finally:
        try:
            await held.__aexit__(None, None, None)
        except RuntimeError:
            pass

    assert closed is True
    assert text == ""
    assert calls == [f"exists:{session_name}", f"terminate:{session_name}"]


@pytest.mark.asyncio
async def test_send_interactive_input_waits_for_session_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner()
    calls: list[str] = []

    async def fake_ensure(*, session_name: str, workdir: str, env=None) -> tuple[bool, str]:
        calls.append("ensure")
        return True, ""

    async def fake_send(session_name: str, prompt: str, *, workdir: str, env, interactive: bool) -> tuple[bool, str]:
        calls.append("send")
        return True, ""

    monkeypatch.setattr(runner, "_ensure_claude_interactive_session", fake_ensure)
    monkeypatch.setattr(runner, "_send_command", fake_send)

    session_name = runner.build_session_name("user_1")
    held = runner._session_lock(session_name)
    await held.__aenter__()
    try:
        task = asyncio.create_task(runner.send_interactive_input(terminal_key="user_1", workdir="/tmp", text="hello"))
        await asyncio.sleep(0)
        assert calls == []

        await held.__aexit__(None, None, None)
        ok, err = await asyncio.wait_for(task, timeout=1)
    finally:
        try:
            await held.__aexit__(None, None, None)
        except RuntimeError:
            pass

    assert ok is True
    assert err == ""
    assert calls == ["ensure", "send"]


@pytest.mark.asyncio
async def test_run_requires_session_store_for_session_tracking(tmp_path: Path) -> None:
    runner = TmuxRunner(data_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="session_store is required"):
        await _collect_events(
            runner.run(
                task_id="task-no-store",
                argv=["echo", "ok"],
                workdir=str(tmp_path),
                timeout_sec=1,
                terminal_key="user_1",
            )
        )


@pytest.mark.asyncio
async def test_close_terminal_treats_missing_tmux_socket_as_missing_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner(cancel_grace_sec=0)
    calls: list[tuple[str, ...]] = []

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        calls.append(args)
        if args and args[0] == "has-session":
            return 1, "", "error connecting to /private/tmp/tmux-501/default (No such file or directory)"
        return 0, "", ""

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)

    closed, text = await runner.close_terminal("user_1")

    assert closed is False
    assert text == "终端不存在"
    assert not any(args and args[0] == "kill-session" for args in calls)


@pytest.mark.asyncio
async def test_close_terminal_reports_session_check_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner(cancel_grace_sec=0)
    calls: list[tuple[str, ...]] = []

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        calls.append(args)
        if args and args[0] == "has-session":
            return 1, "", "permission denied"
        return 0, "", ""

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)

    closed, text = await runner.close_terminal("user_1")

    assert closed is False
    assert text == "终端状态检查失败: permission denied"
    assert not any(args and args[0] == "kill-session" for args in calls)


@pytest.mark.asyncio
async def test_close_terminal_reports_tmux_binary_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner(cancel_grace_sec=0, tmux_bin="missing-tmux")

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        raise FileNotFoundError("missing-tmux")

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)

    closed, text = await runner.close_terminal("user_1")

    assert closed is False
    assert text == "终端状态检查失败: 找不到 tmux 可执行文件 (missing-tmux)"


@pytest.mark.asyncio
async def test_terminate_session_succeeds_when_session_is_already_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner(cancel_grace_sec=0)
    calls: list[tuple[str, ...]] = []

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        calls.append(args)
        if args and args[0] == "has-session":
            return 1, "", "can't find session"
        return 1, "", "can't find session"

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)

    assert await runner._terminate_session("missing-session") is True
    assert ("kill-session", "-t", "missing-session") not in calls


@pytest.mark.asyncio
async def test_terminate_session_returns_false_when_kill_fails_and_session_remains(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner(cancel_grace_sec=0)
    calls: list[tuple[str, ...]] = []

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        calls.append(args)
        if args and args[0] == "has-session":
            return 0, "", ""
        if args and args[0] == "kill-session":
            return 1, "", "kill failed"
        return 0, "", ""

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)

    assert await runner._terminate_session("stuck-session") is False
    assert ("kill-session", "-t", "stuck-session") in calls


@pytest.mark.asyncio
async def test_cancel_queued_persistent_task_does_not_interrupt_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = TmuxRunner(data_dir=str(tmp_path))
    meta = _TmuxTaskMeta(
        session_name="tgcli_user_1",
        log_file=tmp_path / "queued.log",
        exit_file=tmp_path / "queued.exit",
        task_id="task-queued",
        workdir=str(tmp_path),
        persistent_terminal=True,
    )
    runner._tasks[meta.task_id] = meta
    interrupted: list[str] = []

    async def fake_interrupt(session_name: str) -> bool:
        interrupted.append(session_name)
        return True

    monkeypatch.setattr(runner, "_interrupt_session", fake_interrupt)

    assert await runner.cancel("task-queued") is True
    assert meta.cancel_requested is True
    assert interrupted == []


@pytest.mark.asyncio
async def test_cancel_started_persistent_task_interrupts_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = TmuxRunner(data_dir=str(tmp_path))
    meta = _TmuxTaskMeta(
        session_name="tgcli_user_1",
        log_file=tmp_path / "started.log",
        exit_file=tmp_path / "started.exit",
        task_id="task-started",
        workdir=str(tmp_path),
        persistent_terminal=True,
        command_started_at=utc_now(),
    )
    runner._tasks[meta.task_id] = meta
    interrupted: list[str] = []

    async def fake_interrupt(session_name: str) -> bool:
        interrupted.append(session_name)
        return True

    monkeypatch.setattr(runner, "_interrupt_session", fake_interrupt)

    assert await runner.cancel("task-started") is True
    assert meta.cancel_requested is True
    assert interrupted == ["tgcli_user_1"]


@pytest.mark.asyncio
async def test_cancel_returns_false_when_ephemeral_terminate_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.adapters.process.tmux_runner")
    runner = TmuxRunner(data_dir=str(tmp_path))
    meta = _TmuxTaskMeta(
        session_name="tgcli_task_cancel",
        log_file=tmp_path / "cancel.log",
        exit_file=tmp_path / "cancel.exit",
        task_id="task-cancel",
        workdir=str(tmp_path),
        persistent_terminal=False,
    )
    runner._tasks[meta.task_id] = meta

    async def fake_terminate(session_name: str) -> bool:
        assert session_name == "tgcli_task_cancel"
        return False

    monkeypatch.setattr(runner, "_terminate_session", fake_terminate)

    assert await runner.cancel("task-cancel") is False
    assert meta.cancel_requested is True
    assert any(record.message == "tmux task cancel requested" for record in caplog.records)


@pytest.mark.asyncio
async def test_persistent_session_locks_are_ref_counted_and_cleaned(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    current = {"now": 0.0}

    def clock() -> float:
        return current["now"]

    def advance(seconds: float) -> None:
        current["now"] += seconds

    runner = _runner_with_session_store(
        tmp_path,
        session_lock_ttl_sec=1,
        lock_cleanup_interval_sec=1,
        lock_cleanup_batch_size=10,
        lock_clock=clock,
    )

    async def fake_run_task(*, meta, timeout_sec: int, env, workdir: str, command: str):
        yield CLIEvent(type=EventType.STARTED, task_id=meta.task_id)
        yield CLIEvent(type=EventType.EXITED, task_id=meta.task_id, exit_code=0)

    monkeypatch.setattr(runner, "_run_task", fake_run_task)

    for idx in range(3):
        events = await _collect_events(
            runner.run(
                task_id=f"task-{idx}",
                argv=["echo", "ok"],
                workdir=str(tmp_path),
                timeout_sec=10,
                terminal_key=f"user-{idx}",
            )
        )
        assert events[-1].type == EventType.EXITED
        advance(2.0)

    assert len(runner._session_locks) <= 1


@pytest.mark.asyncio
async def test_select_user_question_option_waits_for_pane_io_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner()
    calls: list[str] = []

    async def fake_ensure(*, session_name: str) -> tuple[bool, str]:
        calls.append("ensure")
        return True, ""

    async def fake_move(*, session_name: str, target_index: int) -> tuple[bool, str]:
        calls.append(f"move:{target_index}")
        return True, ""

    async def fake_send_keys(session_name: str, *keys: str) -> tuple[bool, str]:
        calls.append(f"keys:{keys}")
        return True, ""

    monkeypatch.setattr(runner, "_ensure_live_claude_session_for_user_question", fake_ensure)
    monkeypatch.setattr(runner, "_move_user_question_cursor_to_option", fake_move)
    monkeypatch.setattr(runner, "_send_keys", fake_send_keys)

    held = runner._pane_io_lock("tgcli_user_1")
    await held.__aenter__()
    try:
        task = asyncio.create_task(
            runner.select_user_question_option(terminal_key="user_1", workdir="/tmp", option_index=2, submit_after=True)
        )
        await asyncio.sleep(0)
        assert calls == []

        await held.__aexit__(None, None, None)
        ok, err = await task
    finally:
        # Ensure lock released even on assertion failure above
        try:
            await held.__aexit__(None, None, None)
        except RuntimeError:
            pass

    assert ok is True
    assert err == ""
    assert calls == ["ensure", "move:2", "keys:('C-m',)", "keys:('C-m',)"]


@pytest.mark.asyncio
async def test_answer_user_question_with_text_waits_for_pane_io_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = TmuxRunner(enter_delay_sec=0.0)
    calls: list[str] = []

    async def fake_ensure(*, session_name: str) -> tuple[bool, str]:
        calls.append("ensure")
        return True, ""

    async def fake_move(*, session_name: str, target_index: int) -> tuple[bool, str]:
        calls.append(f"move:{target_index}")
        return True, ""

    async def fake_send_keys(session_name: str, *keys: str) -> tuple[bool, str]:
        calls.append(f"keys:{keys}")
        return True, ""

    async def fake_paste_text(session_name: str, text: str) -> tuple[bool, str]:
        calls.append(f"paste:{text}")
        return True, ""

    monkeypatch.setattr(runner, "_ensure_live_claude_session_for_user_question", fake_ensure)
    monkeypatch.setattr(runner, "_move_user_question_cursor_to_option", fake_move)
    monkeypatch.setattr(runner, "_send_keys", fake_send_keys)
    monkeypatch.setattr(runner, "_paste_text", fake_paste_text)

    held = runner._pane_io_lock("tgcli_user_1")
    await held.__aenter__()
    try:
        task = asyncio.create_task(
            runner.answer_user_question_with_text(
                terminal_key="user_1",
                workdir="/tmp",
                option_count=3,
                text="2026-05-01",
                submit_after=False,
            )
        )
        await asyncio.sleep(0)
        assert calls == []

        await held.__aexit__(None, None, None)
        ok, err = await task
    finally:
        try:
            await held.__aexit__(None, None, None)
        except RuntimeError:
            pass

    assert ok is True
    assert err == ""
    assert calls == [
        "ensure",
        "move:3",
        "keys:('C-m',)",
        "paste:2026-05-01",
        "keys:('C-m',)",
    ]


@pytest.mark.asyncio
async def test_advance_user_question_after_multi_select_waits_for_pane_io_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = TmuxRunner(enter_delay_sec=0.0)
    calls: list[str] = []

    async def fake_ensure(*, session_name: str) -> tuple[bool, str]:
        calls.append("ensure")
        return True, ""

    async def fake_send_keys(session_name: str, *keys: str) -> tuple[bool, str]:
        calls.append(f"keys:{keys}")
        return True, ""

    monkeypatch.setattr(runner, "_ensure_live_claude_session_for_user_question", fake_ensure)
    monkeypatch.setattr(runner, "_send_keys", fake_send_keys)

    held = runner._pane_io_lock("tgcli_user_1")
    await held.__aenter__()
    try:
        task = asyncio.create_task(
            runner.advance_user_question_after_multi_select(terminal_key="user_1", workdir="/tmp", final_question=True)
        )
        await asyncio.sleep(0)
        assert calls == []

        await held.__aexit__(None, None, None)
        ok, err = await task
    finally:
        try:
            await held.__aexit__(None, None, None)
        except RuntimeError:
            pass

    assert ok is True
    assert err == ""
    assert calls == ["ensure", "keys:('Right',)", "keys:('C-m',)"]


@pytest.mark.asyncio
async def test_send_command_rebuilds_and_retries_when_ctrl_u_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """路径A：send-keys C-u 返回非0 → _force_rebuild_session → 重试 C-u。"""
    runner = TmuxRunner()
    cu_calls: list[tuple[str, ...]] = []
    rebuild_calls: list[dict] = []

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        if args and args[0] == "send-keys" and len(args) >= 4 and args[3] == "C-u":
            cu_calls.append(args)
            if len(cu_calls) == 1:
                return 1, "", "session not found"
            return 0, "", ""
        return 0, "", ""

    async def fake_force_rebuild_session(session_name: str, *, workdir: str, env, interactive: bool = False):
        rebuild_calls.append({"session_name": session_name, "interactive": interactive})
        return True, ""

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)
    monkeypatch.setattr(runner, "_force_rebuild_session", fake_force_rebuild_session)

    ok, err = await runner._send_command("tgcli_user_1", "hello", workdir="/tmp", env=None, interactive=True)

    assert ok is True
    assert err == ""
    # C-u sent twice: first failed, retry after rebuild succeeded
    assert len(cu_calls) == 2
    assert rebuild_calls == [{"session_name": "tgcli_user_1", "interactive": True}]


@pytest.mark.asyncio
async def test_send_command_reloads_buffer_and_retries_when_enter_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """路径C：send-keys Enter/C-m 返回非0 → rebuild → 重新 _load_and_paste_text + 重新 send-keys。"""
    runner = TmuxRunner(enter_delay_sec=0.0)
    load_calls = 0
    paste_calls = 0
    enter_calls: list[tuple[str, ...]] = []
    rebuild_calls: list[dict] = []

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        nonlocal load_calls, paste_calls
        if args and args[0] == "load-buffer":
            load_calls += 1
            return 0, "", ""
        if args and args[0] == "paste-buffer":
            paste_calls += 1
            return 0, "", ""
        if args and args[0] == "send-keys" and len(args) >= 4 and args[1] == "-t" and args[3] in {"Enter", "C-m"}:
            enter_calls.append(args)
            if len(enter_calls) == 1:
                return 1, "", "no such session"
            return 0, "", ""
        return 0, "", ""

    async def fake_force_rebuild_session(session_name: str, *, workdir: str, env, interactive: bool = False):
        rebuild_calls.append({"session_name": session_name, "interactive": interactive})
        return True, ""

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)
    monkeypatch.setattr(runner, "_force_rebuild_session", fake_force_rebuild_session)

    ok, err = await runner._send_command("tgcli_user_1", "hello", workdir="/tmp", env=None, interactive=True)

    assert ok is True
    assert err == ""
    assert rebuild_calls == [{"session_name": "tgcli_user_1", "interactive": True}]
    # Buffer reloaded and pasted twice: once before Enter failure, once after rebuild
    assert load_calls == 2
    assert paste_calls == 2
    # C-m sent twice: first failed, retry after rebuild+reload succeeded
    assert len(enter_calls) == 2
    assert all(call[3] == "C-m" for call in enter_calls)


@pytest.mark.asyncio
async def test_send_command_force_rebuild_uses_persistent_session_when_not_interactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """interactive=False 分支：paste 失败 → _force_rebuild_session(interactive=False) 走 _ensure_persistent_session。"""
    runner = TmuxRunner()
    ensure_persistent_calls: list[str] = []
    ensure_interactive_calls: list[str] = []
    rebuild_interactive_args: list[bool] = []
    paste_calls = 0

    async def fake_run_tmux(*args: str, input_data: bytes | None = None):
        nonlocal paste_calls
        # load-buffer succeeds; paste-buffer fails first time, succeeds on retry.
        if args and args[0] == "paste-buffer":
            paste_calls += 1
            if paste_calls == 1:
                return 1, "", "no such pane"
            return 0, "", ""
        return 0, "", ""

    async def fake_terminate_session(session_name: str) -> bool:
        return True

    async def fake_ensure_persistent_session(session_name: str, *, workdir: str, env):
        ensure_persistent_calls.append(session_name)
        return True, ""

    async def fake_ensure_claude_interactive_session(*, session_name: str, workdir: str, env):
        ensure_interactive_calls.append(session_name)
        return True, ""

    real_force = runner._force_rebuild_session

    async def spy_force_rebuild_session(session_name: str, *, workdir: str, env, interactive: bool = False):
        rebuild_interactive_args.append(interactive)
        # Delegate to the real method so _terminate_session + the interactive-branch
        # dispatch are exercised (both terminate and ensure methods are monkeypatched).
        return await real_force(session_name, workdir=workdir, env=env, interactive=interactive)

    monkeypatch.setattr(runner, "_run_tmux", fake_run_tmux)
    monkeypatch.setattr(runner, "_terminate_session", fake_terminate_session)
    monkeypatch.setattr(runner, "_ensure_persistent_session", fake_ensure_persistent_session)
    monkeypatch.setattr(runner, "_ensure_claude_interactive_session", fake_ensure_claude_interactive_session)
    monkeypatch.setattr(runner, "_force_rebuild_session", spy_force_rebuild_session)

    ok, err = await runner._send_command("tgcli_user_1", "hello", workdir="/tmp", env=None, interactive=False)

    assert ok is True
    assert err == ""
    assert rebuild_interactive_args == [False]
    assert ensure_persistent_calls == ["tgcli_user_1"]
    assert ensure_interactive_calls == []
