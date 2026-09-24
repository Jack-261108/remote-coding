from __future__ import annotations

import pytest

from app.adapters.storage.file_session_store import FileSessionStore
from app.domain.session_models import ConversationTurn, SessionEvent, SessionEventType, SessionPhase, ToolStatus
from app.services.session_store import SessionStore


def test_file_session_store_rejects_path_traversal_session_id(tmp_path) -> None:
    store = FileSessionStore(str(tmp_path))

    with pytest.raises(ValueError):
        store.raw_transcript_path("../evil")


def test_session_store_rejects_invalid_session_id(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))

    with pytest.raises(ValueError):
        store.get_or_create(session_id="../evil", workdir="/tmp/project")

    with pytest.raises(ValueError):
        store.process(SessionEvent(session_id="../evil", type=SessionEventType.SESSION_STARTED))


def test_session_store_tracks_permission_from_hook_events(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="s1", workdir="/tmp/project")

    state = store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "s1",
                "cwd": "/tmp/project",
                "event": "PreToolUse",
                "status": "running_tool",
                "tool": "Bash",
                "tool_input": {"command": "pwd"},
                "tool_use_id": "tool-1",
            },
        )
    )
    assert state.tool_calls["tool-1"].status == ToolStatus.RUNNING
    assert state.phase == SessionPhase.PROCESSING

    state = store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "s1",
                "cwd": "/tmp/project",
                "event": "PermissionRequest",
                "status": "waiting_for_approval",
                "tool": "Bash",
                "tool_input": {"command": "pwd"},
                "tool_use_id": "tool-1",
            },
        )
    )
    assert state.phase == SessionPhase.WAITING_FOR_APPROVAL
    assert state.pending_permission is not None
    assert state.pending_permission.tool_use_id == "tool-1"
    assert state.tool_calls["tool-1"].status == ToolStatus.WAITING_FOR_APPROVAL

    state = store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.PERMISSION_APPROVED,
            payload={"tool_use_id": "tool-1"},
        )
    )
    assert state.phase == SessionPhase.PROCESSING
    assert state.pending_permission is None
    assert state.tool_calls["tool-1"].status == ToolStatus.RUNNING


def test_session_store_permission_response_failed_only_interrupts_matching_tool(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="s1", workdir="/tmp/project")

    for tool_use_id in ("tool-1", "tool-2"):
        store.process(
            SessionEvent(
                session_id="s1",
                type=SessionEventType.HOOK_RECEIVED,
                payload={
                    "session_id": "s1",
                    "cwd": "/tmp/project",
                    "event": "PreToolUse",
                    "status": "running_tool",
                    "tool": "Bash",
                    "tool_input": {"command": tool_use_id},
                    "tool_use_id": tool_use_id,
                },
            )
        )
        store.process(
            SessionEvent(
                session_id="s1",
                type=SessionEventType.HOOK_RECEIVED,
                payload={
                    "session_id": "s1",
                    "cwd": "/tmp/project",
                    "event": "PermissionRequest",
                    "status": "waiting_for_approval",
                    "tool": "Bash",
                    "tool_input": {"command": tool_use_id},
                    "tool_use_id": tool_use_id,
                },
            )
        )

    state = store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.PERMISSION_RESPONSE_FAILED,
            payload={"tool_use_id": "tool-1"},
        )
    )

    assert state.tool_calls["tool-1"].status == ToolStatus.INTERRUPTED
    assert state.tool_calls["tool-2"].status == ToolStatus.WAITING_FOR_APPROVAL
    assert state.pending_permission is not None
    assert state.pending_permission.tool_use_id == "tool-2"
    assert state.phase == SessionPhase.WAITING_FOR_APPROVAL


def test_session_store_file_synced_preserves_pending_permission_from_hook(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="s1", workdir="/tmp/project")

    store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "s1",
                "cwd": "/tmp/project",
                "event": "PreToolUse",
                "status": "running_tool",
                "tool": "Bash",
                "tool_input": {"command": "pwd"},
                "tool_use_id": "tool-1",
            },
        )
    )
    state = store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "s1",
                "cwd": "/tmp/project",
                "event": "PermissionRequest",
                "status": "waiting_for_approval",
                "tool": "Bash",
                "tool_input": {"command": "pwd"},
                "tool_use_id": "tool-1",
            },
        )
    )
    assert state.phase == SessionPhase.WAITING_FOR_APPROVAL

    state = store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.FILE_SYNCED,
            payload={
                "turns": [],
                "tool_calls": {
                    "tool-1": {
                        "tool_use_id": "tool-1",
                        "name": "Bash",
                        "input": {"command": "pwd"},
                        "status": "running",
                        "result": None,
                        "structured_result": None,
                        "started_at": "2026-04-16T10:00:01+00:00",
                        "completed_at": None,
                    }
                },
                "last_offset": 18,
            },
        )
    )

    assert state.phase == SessionPhase.WAITING_FOR_APPROVAL
    assert state.pending_permission is not None
    assert state.pending_permission.tool_use_id == "tool-1"
    assert state.tool_calls["tool-1"].status == ToolStatus.WAITING_FOR_APPROVAL


def test_session_store_applies_jsonl_snapshot(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="s1", workdir="/tmp/project")

    state = store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.FILE_SYNCED,
            payload={
                "turns": [
                    {
                        "turn_id": "u1",
                        "role": "user",
                        "text": "\n你好\n",
                        "source": "jsonl",
                        "is_complete": True,
                        "started_at": "2026-04-16T10:00:00+00:00",
                        "ended_at": "2026-04-16T10:00:00+00:00",
                    },
                    {
                        "turn_id": "a1",
                        "role": "assistant",
                        "text": "\n这里是干净回复\n",
                        "source": "jsonl",
                        "is_complete": True,
                        "started_at": "2026-04-16T10:00:01+00:00",
                        "ended_at": "2026-04-16T10:00:01+00:00",
                    },
                ],
                "tool_calls": {
                    "tool-1": {
                        "tool_use_id": "tool-1",
                        "name": "Bash",
                        "input": {"command": "pwd"},
                        "status": "success",
                        "result": "/tmp/project",
                        "structured_result": {"stdout": "/tmp/project"},
                        "started_at": "2026-04-16T10:00:01+00:00",
                        "completed_at": "2026-04-16T10:00:02+00:00",
                    }
                },
                "summary": "查看当前目录",
                "last_reply": "这里是干净回复",
                "last_reply_role": "assistant",
                "last_offset": 123,
            },
        )
    )

    assert state.phase == SessionPhase.WAITING_FOR_INPUT
    assert state.summary == "查看当前目录"
    assert state.last_reply == "这里是干净回复"
    assert state.turns[-1].text == "\n这里是干净回复\n"
    assert state.tool_calls["tool-1"].status == ToolStatus.SUCCESS
    assert state.tool_calls["tool-1"].structured_result == {"stdout": "/tmp/project"}
    assert state.checkpoint.last_offset == 123
    assert state.checkpoint.completed_tool_ids == ["tool-1"]


def test_session_store_file_synced_replaces_old_state_and_interrupts_on_end(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="s1", workdir="/tmp/project")
    store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "s1",
                "cwd": "/tmp/project",
                "event": "PreToolUse",
                "status": "running_tool",
                "tool": "Bash",
                "tool_input": {"command": "pwd"},
                "tool_use_id": "tool-1",
            },
        )
    )

    state = store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.FILE_SYNCED,
            payload={
                "turns": [
                    {
                        "turn_id": "a2",
                        "role": "assistant",
                        "text": "\n新快照\n",
                        "source": "jsonl",
                        "is_complete": True,
                        "started_at": "2026-04-16T10:00:03+00:00",
                        "ended_at": "2026-04-16T10:00:03+00:00",
                    }
                ],
                "tool_calls": {},
                "last_reply": "新快照",
                "last_reply_role": "assistant",
                "last_offset": 222,
            },
        )
    )

    assert [turn.turn_id for turn in state.turns] == ["a2"]
    assert state.tool_calls == {}
    assert state.phase == SessionPhase.WAITING_FOR_INPUT

    state = store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "s1",
                "cwd": "/tmp/project",
                "event": "PreToolUse",
                "status": "running_tool",
                "tool": "Bash",
                "tool_input": {"command": "pwd"},
                "tool_use_id": "tool-2",
            },
        )
    )
    assert state.tool_calls["tool-2"].status == ToolStatus.RUNNING

    state = store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "s1",
                "cwd": "/tmp/project",
                "event": "SessionEnd",
                "status": "ended",
            },
        )
    )

    assert state.phase == SessionPhase.ENDED
    assert state.tool_calls["tool-2"].status == ToolStatus.INTERRUPTED


@pytest.mark.parametrize(
    "event",
    [
        SessionEvent(session_id="claude-session-ended", type=SessionEventType.SESSION_STARTED),
        SessionEvent(
            session_id="claude-session-ended",
            type=SessionEventType.TURN_STARTED,
            payload={"turn_id": "late-turn", "role": "assistant"},
        ),
        SessionEvent(
            session_id="claude-session-ended",
            type=SessionEventType.TURN_COMPLETED,
            payload={"turn_id": "late-turn"},
        ),
        SessionEvent(session_id="claude-session-ended", type=SessionEventType.CLEAR_DETECTED),
    ],
)
def test_session_store_ended_session_ignores_late_state_events(tmp_path, event: SessionEvent) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    state = store.get_or_create(session_id="claude-session-ended", workdir="/tmp/project")
    state.phase = SessionPhase.ENDED
    store._persist(state)

    state = store.process(event)

    assert state.phase == SessionPhase.ENDED
    assert state.current_turn_id is None
    assert state.turns == []


@pytest.mark.parametrize("event_type", [SessionEventType.FILE_SYNCED, SessionEventType.HISTORY_LOADED])
def test_session_store_ended_session_stays_ended_when_snapshot_syncs(tmp_path, event_type: SessionEventType) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    state = store.get_or_create(session_id="claude-session-ended", workdir="/tmp/project")
    state.phase = SessionPhase.ENDED
    store._persist(state)

    state = store.process(
        SessionEvent(
            session_id="claude-session-ended",
            type=event_type,
            payload={
                "turns": [
                    {
                        "turn_id": "a-late",
                        "role": "assistant",
                        "text": "\n结束后的最终同步\n",
                        "source": "jsonl",
                        "is_complete": True,
                        "started_at": "2026-04-16T10:00:03+00:00",
                        "ended_at": "2026-04-16T10:00:03+00:00",
                    }
                ],
                "tool_calls": {},
                "last_reply": "结束后的最终同步",
                "last_reply_role": "assistant",
                "last_offset": 18,
            },
        )
    )

    assert state.phase == SessionPhase.ENDED
    assert state.history_loaded is True
    assert state.last_reply == "结束后的最终同步"


@pytest.mark.parametrize(
    "hook_payload",
    [
        {
            "session_id": "claude-session-ended",
            "cwd": "/tmp/project",
            "event": "PreToolUse",
            "status": "running_tool",
            "tool": "Bash",
            "tool_input": {"command": "pwd"},
            "tool_use_id": "tool-late",
        },
        {
            "session_id": "claude-session-ended",
            "cwd": "/tmp/project",
            "event": "PermissionRequest",
            "status": "waiting_for_approval",
            "tool": "Bash",
            "tool_input": {"command": "pwd"},
            "tool_use_id": "tool-late",
        },
    ],
)
def test_session_store_ended_session_ignores_late_active_hooks(tmp_path, hook_payload: dict) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    state = store.get_or_create(session_id="claude-session-ended", workdir="/tmp/project")
    state.phase = SessionPhase.ENDED
    store._persist(state)

    state = store.process(
        SessionEvent(
            session_id="claude-session-ended",
            type=SessionEventType.HOOK_RECEIVED,
            payload=hook_payload,
        )
    )

    assert state.phase == SessionPhase.ENDED
    assert state.pending_permission is None
    assert "tool-late" not in state.tool_calls


@pytest.mark.parametrize("decision_type", [SessionEventType.PERMISSION_APPROVED, SessionEventType.PERMISSION_DENIED])
def test_session_store_ended_session_ignores_late_permission_decision(tmp_path, decision_type: SessionEventType) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.process(
        SessionEvent(
            session_id="claude-session-ended",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "claude-session-ended",
                "cwd": "/tmp/project",
                "event": "PermissionRequest",
                "status": "waiting_for_approval",
                "tool": "Bash",
                "tool_input": {"command": "pwd"},
                "tool_use_id": "tool-1",
            },
        )
    )
    state = store.process(
        SessionEvent(
            session_id="claude-session-ended",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "claude-session-ended",
                "cwd": "/tmp/project",
                "event": "SessionEnd",
                "status": "ended",
            },
        )
    )
    assert state.phase == SessionPhase.ENDED
    assert state.tool_calls["tool-1"].status == ToolStatus.INTERRUPTED

    state = store.process(
        SessionEvent(
            session_id="claude-session-ended",
            type=decision_type,
            payload={"tool_use_id": "tool-1"},
        )
    )

    assert state.phase == SessionPhase.ENDED
    assert state.interrupted is True
    assert state.pending_permission is None
    assert state.tool_calls["tool-1"].status == ToolStatus.INTERRUPTED


def test_session_store_session_ended_event_interrupts_active_tool(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.process(
        SessionEvent(
            session_id="claude-session-ended",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "claude-session-ended",
                "cwd": "/tmp/project",
                "event": "PreToolUse",
                "status": "running_tool",
                "tool": "Bash",
                "tool_input": {"command": "pwd"},
                "tool_use_id": "tool-1",
            },
        )
    )

    state = store.process(SessionEvent(session_id="claude-session-ended", type=SessionEventType.SESSION_ENDED))

    assert state.phase == SessionPhase.ENDED
    assert state.pending_permission is None
    assert state.tool_calls["tool-1"].status == ToolStatus.INTERRUPTED


def test_session_store_hook_session_end_preserves_existing_interrupt_flag(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.process(
        SessionEvent(
            session_id="claude-session-ended",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "claude-session-ended",
                "cwd": "/tmp/project",
                "event": "PreToolUse",
                "status": "running_tool",
                "tool": "Bash",
                "tool_input": {"command": "pwd"},
                "tool_use_id": "tool-1",
            },
        )
    )
    state = store.process(SessionEvent(session_id="claude-session-ended", type=SessionEventType.INTERRUPT_DETECTED))
    assert state.interrupted is True
    assert state.tool_calls["tool-1"].status == ToolStatus.INTERRUPTED

    state = store.process(
        SessionEvent(
            session_id="claude-session-ended",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "claude-session-ended",
                "cwd": "/tmp/project",
                "event": "SessionEnd",
                "status": "ended",
            },
        )
    )

    assert state.phase == SessionPhase.ENDED
    assert state.interrupted is True
    assert state.tool_calls["tool-1"].status == ToolStatus.INTERRUPTED


@pytest.mark.parametrize(
    "end_event",
    [
        SessionEvent(
            session_id="claude-session-ended",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "claude-session-ended",
                "cwd": "/tmp/project",
                "event": "SessionEnd",
                "status": "ended",
            },
        ),
        SessionEvent(session_id="claude-session-ended", type=SessionEventType.SESSION_ENDED),
    ],
)
def test_session_store_session_end_interrupts_active_subagent_tools(tmp_path, end_event: SessionEvent) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.process(
        SessionEvent(
            session_id="claude-session-ended",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "claude-session-ended",
                "cwd": "/tmp/project",
                "event": "PreToolUse",
                "status": "running_tool",
                "tool": "Task",
                "tool_input": {"description": "do thing"},
                "tool_use_id": "task-1",
            },
        )
    )
    store.process(
        SessionEvent(
            session_id="claude-session-ended",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "claude-session-ended",
                "cwd": "/tmp/project",
                "event": "PreToolUse",
                "status": "running_tool",
                "tool": "Read",
                "tool_input": {"file_path": "/tmp/a.py"},
                "tool_use_id": "sub-1",
            },
        )
    )

    state = store.process(end_event)

    assert state.phase == SessionPhase.ENDED
    assert state.tool_calls["task-1"].status == ToolStatus.INTERRUPTED
    assert state.tool_calls["task-1"].subagent_tools[0].status == ToolStatus.INTERRUPTED
    assert state.subagent_state.active_tasks["task-1"].subagent_tools[0].status == ToolStatus.INTERRUPTED


@pytest.mark.parametrize("tool_status", ["running", "waiting_for_approval"])
def test_session_store_ended_session_interrupts_active_tools_from_late_snapshot(tmp_path, tool_status: str) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.process(
        SessionEvent(
            session_id="claude-session-ended",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "claude-session-ended",
                "cwd": "/tmp/project",
                "event": "PreToolUse",
                "status": "running_tool",
                "tool": "Bash",
                "tool_input": {"command": "pwd"},
                "tool_use_id": "tool-1",
            },
        )
    )
    state = store.process(
        SessionEvent(
            session_id="claude-session-ended",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "claude-session-ended",
                "cwd": "/tmp/project",
                "event": "SessionEnd",
                "status": "ended",
            },
        )
    )
    assert state.phase == SessionPhase.ENDED
    assert state.tool_calls["tool-1"].status == ToolStatus.INTERRUPTED

    state = store.process(
        SessionEvent(
            session_id="claude-session-ended",
            type=SessionEventType.FILE_SYNCED,
            payload={
                "turns": [],
                "tool_calls": {
                    "tool-1": {
                        "tool_use_id": "tool-1",
                        "name": "Bash",
                        "input": {"command": "pwd"},
                        "status": tool_status,
                        "started_at": "2026-04-16T10:00:01+00:00",
                    }
                },
                "last_offset": 18,
            },
        )
    )

    assert state.phase == SessionPhase.ENDED
    assert state.pending_permission is None
    assert state.tool_calls["tool-1"].status == ToolStatus.INTERRUPTED


def test_session_store_file_synced_promotes_complete_turn_without_waiting_hook(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    state = store.get_or_create(session_id="claude-session-1", workdir="/tmp/project", terminal_id="user_1_36d00faeb25f")
    state.phase = SessionPhase.PROCESSING
    store._persist(state)

    state = store.process(
        SessionEvent(
            session_id="claude-session-1",
            type=SessionEventType.FILE_SYNCED,
            payload={
                "turns": [
                    {
                        "turn_id": "a1",
                        "role": "assistant",
                        "text": "\n补同步后的回复\n",
                        "source": "jsonl",
                        "is_complete": True,
                        "started_at": "2026-04-16T10:00:03+00:00",
                        "ended_at": "2026-04-16T10:00:03+00:00",
                    }
                ],
                "tool_calls": {},
                "last_reply": "补同步后的回复",
                "last_reply_role": "assistant",
                "last_offset": 18,
            },
        )
    )

    assert state.phase == SessionPhase.WAITING_FOR_INPUT
    assert state.last_reply == "补同步后的回复"


def test_session_store_tracks_subagent_tools_from_hook_and_snapshot(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="s1", workdir="/tmp/project")

    state = store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "s1",
                "cwd": "/tmp/project",
                "event": "PreToolUse",
                "status": "running_tool",
                "tool": "Task",
                "tool_input": {"description": "do thing"},
                "tool_use_id": "task-1",
            },
        )
    )
    state = store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "s1",
                "cwd": "/tmp/project",
                "event": "PreToolUse",
                "status": "running_tool",
                "tool": "Read",
                "tool_input": {"file_path": "/tmp/a.py"},
                "tool_use_id": "sub-1",
            },
        )
    )

    assert state.subagent_state.has_active_subagent is True
    assert state.tool_calls["task-1"].subagent_tools[0].tool_use_id == "sub-1"
    assert state.tool_calls["task-1"].subagent_tools[0].status == ToolStatus.RUNNING

    state = store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.FILE_SYNCED,
            payload={
                "turns": [],
                "tool_calls": {
                    "task-1": {
                        "tool_use_id": "task-1",
                        "name": "Task",
                        "input": {"description": "do thing"},
                        "status": "success",
                        "result": "done",
                        "structured_result": {"agentId": "agent-1", "status": "completed", "content": "done"},
                        "subagent_tools": [
                            {
                                "tool_use_id": "sub-1",
                                "name": "Read",
                                "input": {"file_path": "/tmp/a.py"},
                                "status": "success",
                                "result": "done",
                                "structured_result": {"stdout": "done"},
                                "started_at": "2026-04-16T10:00:01+00:00",
                                "completed_at": "2026-04-16T10:00:02+00:00",
                            }
                        ],
                        "started_at": "2026-04-16T10:00:00+00:00",
                        "completed_at": "2026-04-16T10:00:03+00:00",
                    }
                },
                "last_offset": 18,
            },
        )
    )

    assert state.tool_calls["task-1"].subagent_tools[0].status == ToolStatus.SUCCESS
    assert state.subagent_state.active_tasks["task-1"].subagent_tools[0].status == ToolStatus.SUCCESS


def test_session_store_file_synced_ignores_older_snapshot(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    state = store.get_or_create(session_id="claude-session-1", workdir="/tmp/project")
    state.phase = SessionPhase.WAITING_FOR_INPUT
    state.checkpoint.last_offset = 30
    state.last_reply = "最新回复"
    state.turns.append(ConversationTurn(turn_id="a-current", role="assistant", text="\n最新回复\n", is_complete=True))
    store._persist(state)

    state = store.process(
        SessionEvent(
            session_id="claude-session-1",
            type=SessionEventType.FILE_SYNCED,
            payload={
                "turns": [],
                "tool_calls": {},
                "last_reply": "旧回复",
                "last_reply_role": "assistant",
                "last_offset": 18,
            },
        )
    )

    assert state.phase == SessionPhase.WAITING_FOR_INPUT
    assert state.last_reply == "最新回复"
    assert state.checkpoint.last_offset == 30


def test_session_store_file_synced_accepts_reset_snapshot_with_smaller_offset(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    state = store.get_or_create(session_id="claude-session-1", workdir="/tmp/project")
    state.phase = SessionPhase.WAITING_FOR_INPUT
    state.checkpoint.last_offset = 30
    state.last_reply = "最新回复"
    state.turns.append(ConversationTurn(turn_id="a-current", role="assistant", text="\n最新回复\n", is_complete=True))
    store._persist(state)

    state = store.process(
        SessionEvent(
            session_id="claude-session-1",
            type=SessionEventType.FILE_SYNCED,
            payload={
                "turns": [
                    {
                        "turn_id": "a-reset",
                        "role": "assistant",
                        "text": "\n重建后的回复\n",
                        "source": "jsonl",
                        "is_complete": True,
                        "started_at": "2026-04-16T10:00:03+00:00",
                        "ended_at": "2026-04-16T10:00:03+00:00",
                    }
                ],
                "tool_calls": {},
                "last_reply": "重建后的回复",
                "last_reply_role": "assistant",
                "last_offset": 12,
                "reset_detected": True,
            },
        )
    )

    assert state.phase == SessionPhase.WAITING_FOR_INPUT
    assert state.last_reply == "重建后的回复"
    assert state.checkpoint.last_offset == 12
    assert [turn.turn_id for turn in state.turns] == ["a-reset"]


def test_session_store_file_synced_accepts_smaller_offset_when_snapshot_has_more_turns(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    state = store.get_or_create(session_id="claude-session-1", workdir="/tmp/project")
    state.phase = SessionPhase.WAITING_FOR_INPUT
    state.checkpoint.last_offset = 30
    state.last_reply = "第一条"
    state.turns.append(ConversationTurn(turn_id="a-current", role="assistant", text="\n第一条\n", is_complete=True))
    store._persist(state)

    state = store.process(
        SessionEvent(
            session_id="claude-session-1",
            type=SessionEventType.FILE_SYNCED,
            payload={
                "turns": [
                    {
                        "turn_id": "a-current",
                        "role": "assistant",
                        "text": "\n第一条\n",
                        "source": "jsonl",
                        "is_complete": True,
                        "started_at": "2026-04-16T10:00:01+00:00",
                        "ended_at": "2026-04-16T10:00:01+00:00",
                    },
                    {
                        "turn_id": "a-next",
                        "role": "assistant",
                        "text": "\n第二条\n",
                        "source": "jsonl",
                        "is_complete": True,
                        "started_at": "2026-04-16T10:00:02+00:00",
                        "ended_at": "2026-04-16T10:00:02+00:00",
                    },
                ],
                "tool_calls": {},
                "last_reply": "第二条",
                "last_reply_role": "assistant",
                "last_offset": 18,
            },
        )
    )

    assert state.phase == SessionPhase.WAITING_FOR_INPUT
    assert state.last_reply == "第二条"
    assert state.checkpoint.last_offset == 18
    assert [turn.turn_id for turn in state.turns] == ["a-current", "a-next"]


def test_subagent_stop_preserves_parent_pending_permission(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="s1", workdir="/tmp/project")

    store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "s1",
                "cwd": "/tmp/project",
                "event": "PreToolUse",
                "status": "running_tool",
                "tool": "Task",
                "tool_input": {"description": "sub task"},
                "tool_use_id": "task-1",
            },
        )
    )

    store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "s1",
                "cwd": "/tmp/project",
                "event": "PermissionRequest",
                "status": "waiting_for_approval",
                "tool": "Bash",
                "tool_input": {"command": "rm -rf tmp"},
                "tool_use_id": "bash-1",
            },
        )
    )

    state = store.get("s1")
    assert state is not None
    assert state.phase == SessionPhase.WAITING_FOR_APPROVAL
    assert state.pending_permission is not None
    assert state.pending_permission.tool_use_id == "bash-1"

    state = store.process(
        SessionEvent(
            session_id="s1",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "s1",
                "cwd": "/tmp/project",
                "event": "SubagentStop",
                "status": "processing",
            },
        )
    )

    assert state.phase == SessionPhase.WAITING_FOR_APPROVAL
    assert state.pending_permission is not None
    assert state.pending_permission.tool_use_id == "bash-1"
    assert state.tool_calls["bash-1"].status == ToolStatus.WAITING_FOR_APPROVAL
    assert state.interrupted is False
