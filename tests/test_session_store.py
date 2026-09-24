import asyncio
import json
from datetime import timedelta

import pytest

from app.adapters.storage.file_session_context_store import FileSessionContextStore
from app.adapters.storage.file_session_store import FileSessionStore
from app.domain.models import SessionContext, utc_now
from app.domain.session_models import (
    ConversationTurn,
    ParserCheckpoint,
    PendingPermission,
    SessionEvent,
    SessionEventType,
    SessionPhase,
    ToolCallRecord,
    ToolStatus,
)
from app.services.session_store import SessionStore


def test_file_store_load_session_context_degrades_on_corrupt_file(tmp_path) -> None:
    """A half-written or unreadable context file must read as absent instead
    of raising: the router's external-target filter and the pairing flow call
    this per plain-text message, and a raise silently drops the message."""
    file_store = FileSessionStore(str(tmp_path))
    file_store.save_session_context(SessionContext(user_id=42, session_id="s1", provider="claude_code", workdir="/project"))
    context_path = file_store.session_context_path(42)
    context_path.write_text("{half-written", encoding="utf-8")
    assert file_store.load_session_context(42) is None


@pytest.mark.asyncio
async def test_context_store_get_loads_disk_once_then_serves_memory(tmp_path) -> None:
    """The external-target router filter and the session-guard middleware both
    call get() for one plain-text message: only the cold start may touch disk."""
    file_store = FileSessionStore(str(tmp_path))
    file_store.save_session_context(SessionContext(user_id=42, session_id="s1", provider="claude_code", workdir="/project"))
    store = FileSessionContextStore(file_store)

    first = await store.get(42)
    assert first is not None
    assert first.session_id == "s1"

    def fail_if_disk_read(user_id: int = 0):
        raise AssertionError("get() must serve the loaded cache instead of reading disk")

    file_store.list_session_contexts = fail_if_disk_read  # type: ignore[method-assign]
    file_store.load_session_context = fail_if_disk_read  # type: ignore[method-assign]

    second = await store.get(42)
    assert second is not first  # independent snapshot: callers mutate it in place
    assert second == first
    assert await store.get(7) is None


@pytest.mark.asyncio
async def test_context_store_list_all_and_index_return_isolated_copies(tmp_path) -> None:
    file_store = FileSessionStore(str(tmp_path))
    file_store.save_session_context(
        SessionContext(user_id=42, session_id="s1", provider="claude_code", workdir="/project", claude_session_id="c1")
    )
    store = FileSessionContextStore(file_store)

    listed = await store.list_all()
    assert len(listed) == 1
    indexed = await store.get_by_claude_session_id("c1")
    assert indexed is not None

    # 验证 list_all 与 index 拿到的是隔离的对象，就地修改不污染缓存
    listed[0].attached_user_ids.append(999)
    indexed.claude_session_id = "mutated"

    fresh_listed = await store.list_all()
    assert fresh_listed[0].attached_user_ids == []

    fresh_indexed = await store.get_by_claude_session_id("c1")
    assert fresh_indexed is not None
    assert fresh_indexed.claude_session_id == "c1"


@pytest.mark.asyncio
async def test_context_store_get_tolerates_corrupt_context_file(tmp_path) -> None:
    """The cache is built via a directory scan, so a corrupt file must degrade
    to absent for its own user only — same contract as load_session_context."""
    file_store = FileSessionStore(str(tmp_path))
    file_store.save_session_context(SessionContext(user_id=42, session_id="s1", provider="claude_code", workdir="/project"))
    file_store.session_context_path(7).write_text("{half-written", encoding="utf-8")
    store = FileSessionContextStore(file_store)

    healthy = await store.get(42)
    assert healthy is not None
    assert healthy.session_id == "s1"
    assert await store.get(7) is None


def test_session_store_persists_checkpoint_and_turns(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    state = store.get_or_create(session_id="s1", user_id=1, workdir="/tmp", terminal_id="user_1_8c393341f536")

    checkpoint = ParserCheckpoint(in_reply_block=True, pending_buffer="abc", current_turn_id="turn-1")
    store.save_checkpoint("s1", checkpoint)
    store.process(SessionEvent(session_id="s1", type=SessionEventType.TURN_STARTED, payload={"turn_id": "turn-1", "role": "assistant"}))
    store.process(
        SessionEvent(
            session_id="s1", type=SessionEventType.PARSER_UPDATED, payload={"turn_id": "turn-1", "text": "\n你好\n", "is_complete": False}
        )
    )

    reloaded = SessionStore(FileSessionStore(str(tmp_path))).get("s1")

    assert state.session_id == "s1"
    assert reloaded is not None
    assert reloaded.checkpoint.pending_buffer == "abc"
    assert reloaded.current_turn_id == "turn-1"
    assert reloaded.turns[-1].text == "\n你好\n"
    assert reloaded.phase == SessionPhase.PROCESSING


def test_session_store_turn_completed_moves_to_waiting_for_input(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="s1")
    store.process(SessionEvent(session_id="s1", type=SessionEventType.TURN_STARTED, payload={"turn_id": "turn-1", "role": "assistant"}))

    state = store.process(SessionEvent(session_id="s1", type=SessionEventType.TURN_COMPLETED, payload={"turn_id": "turn-1"}))

    assert state.phase == SessionPhase.WAITING_FOR_INPUT
    assert state.current_turn_id is None
    assert state.turns[-1].is_complete is True


def test_session_store_interactive_completion_prefers_bound_claude_session(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="tgcli_user_1_8c393341f536", workdir="/tmp", terminal_id="user_1_8c393341f536")
    bound = store.get_or_create(session_id="claude-session-1", workdir="/tmp", terminal_id="user_1_8c393341f536")
    bound.phase = SessionPhase.WAITING_FOR_INPUT
    store._persist(bound)

    phase = store.interactive_completion_phase(
        terminal_id="user_1_8c393341f536",
        workdir="/tmp",
        fallback_session_id="tgcli_user_1_8c393341f536",
    )

    assert phase == SessionPhase.WAITING_FOR_INPUT


def test_session_store_interactive_completion_prefers_explicit_claude_session_id(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    stale = store.get_or_create(session_id="claude-session-stale", workdir="/tmp", terminal_id="user_1_8c393341f536")
    stale.phase = SessionPhase.PROCESSING
    target = store.get_or_create(session_id="claude-session-1", workdir="/tmp", terminal_id=None)
    target.phase = SessionPhase.WAITING_FOR_INPUT
    store._persist(stale)
    store._persist(target)

    phase = store.interactive_completion_phase(
        terminal_id="user_1_8c393341f536",
        workdir="/tmp",
        claude_session_id="claude-session-1",
        fallback_session_id="tgcli_user_1_8c393341f536",
    )

    assert phase == SessionPhase.WAITING_FOR_INPUT


def test_session_store_interactive_completion_prefers_uuid_claude_session_id(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="tgcli_user_1_8c393341f536", workdir="/tmp", terminal_id="user_1_8c393341f536")
    target = store.get_or_create(
        session_id="2185ae1c-14e5-4423-8f0d-1b76fcd893d6",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
    )
    target.phase = SessionPhase.WAITING_FOR_INPUT
    store._persist(target)

    phase = store.interactive_completion_phase(
        terminal_id="user_1_8c393341f536",
        workdir="/tmp",
        claude_session_id="2185ae1c-14e5-4423-8f0d-1b76fcd893d6",
        fallback_session_id="tgcli_user_1_8c393341f536",
    )

    assert phase == SessionPhase.WAITING_FOR_INPUT


def test_session_store_mark_interactive_turn_processing_does_not_revive_ended_session(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    state = store.get_or_create(
        session_id="2185ae1c-14e5-4423-8f0d-1b76fcd893d6",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
    )
    state.phase = SessionPhase.ENDED
    store._persist(state)

    state = store.mark_interactive_turn_processing(
        terminal_id="user_1_8c393341f536",
        workdir="/tmp",
        claude_session_id="2185ae1c-14e5-4423-8f0d-1b76fcd893d6",
        fallback_session_id="tgcli_user_1_8c393341f536",
    )

    assert state is not None
    assert state.phase == SessionPhase.ENDED
    assert store.get("2185ae1c-14e5-4423-8f0d-1b76fcd893d6").phase == SessionPhase.ENDED


def test_session_store_resolve_interactive_session_id_prefers_newer_bound_session_over_stale_explicit(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    now = utc_now()
    old_state = store.get_or_create(
        session_id="2185ae1c-14e5-4423-8f0d-1b76fcd893d6",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
    )
    old_state.last_activity = now - timedelta(days=1)
    store._persist(old_state)

    new_state = store.get_or_create(
        session_id="f5bc22fa-0e77-42f6-a2d3-e422037296f6",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
    )
    new_state.last_activity = now
    store._persist(new_state)

    resolved = store.resolve_interactive_session_id(
        terminal_id="user_1_8c393341f536",
        claude_session_id="2185ae1c-14e5-4423-8f0d-1b76fcd893d6",
        fallback_session_id="tgcli_user_1_8c393341f536",
        require_claude_session=True,
    )

    assert resolved == "f5bc22fa-0e77-42f6-a2d3-e422037296f6"


def test_session_store_resolve_interactive_session_id_keeps_recent_explicit_over_stale_pending_bound(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    now = utc_now()
    stale_pending = store.get_or_create(
        session_id="29766ba6-468c-484c-a759-7c440c2e2c75",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
    )
    stale_pending.phase = SessionPhase.WAITING_FOR_APPROVAL
    stale_pending.last_activity = now - timedelta(days=1)
    stale_pending.pending_permission = PendingPermission(
        tool_use_id="tool-stale",
        tool_name="Bash",
        tool_input={"command": "pwd"},
    )
    stale_pending.turns.append(ConversationTurn(turn_id="turn-stale", role="assistant", text="旧授权", is_complete=True))
    store._persist(stale_pending)

    recent_explicit = store.get_or_create(
        session_id="4550fd41-474b-463e-8db8-2435c71c2f10",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
    )
    recent_explicit.phase = SessionPhase.WAITING_FOR_INPUT
    recent_explicit.last_activity = now
    recent_explicit.turns.append(ConversationTurn(turn_id="turn-current", role="assistant", text="新回复", is_complete=True))
    store._persist(recent_explicit)

    resolved = store.resolve_interactive_session_id(
        terminal_id="user_1_8c393341f536",
        claude_session_id="4550fd41-474b-463e-8db8-2435c71c2f10",
        fallback_session_id="tgcli_user_1_8c393341f536",
        require_claude_session=True,
    )

    assert resolved == "4550fd41-474b-463e-8db8-2435c71c2f10"


def test_session_store_interactive_completion_waits_for_bound_claude_session(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    fallback = store.get_or_create(session_id="tgcli_user_1_8c393341f536", workdir="/tmp", terminal_id="user_1_8c393341f536")
    fallback.phase = SessionPhase.WAITING_FOR_INPUT
    store._persist(fallback)

    phase = store.interactive_completion_phase(
        terminal_id="user_1_8c393341f536",
        workdir="/tmp",
        fallback_session_id="tgcli_user_1_8c393341f536",
    )

    assert phase is None


def test_session_store_find_by_terminal_id_keeps_newer_in_memory_state(tmp_path) -> None:
    file_store = FileSessionStore(str(tmp_path))
    disk_store = SessionStore(file_store)
    disk_state = disk_store.get_or_create(session_id="claude-session-1", workdir="/tmp", terminal_id="user_1_8c393341f536")
    disk_state.phase = SessionPhase.PROCESSING
    disk_store._persist(disk_state)

    store = SessionStore(file_store)
    fresh_state = store.get_or_create(session_id="claude-session-1", workdir="/tmp", terminal_id="user_1_8c393341f536")
    fresh_state.phase = SessionPhase.WAITING_FOR_INPUT
    store._persist(fresh_state)

    matched = store.find_by_terminal_id("user_1_8c393341f536")

    assert matched is fresh_state
    assert matched is not None
    assert matched.phase == SessionPhase.WAITING_FOR_INPUT


def test_session_store_find_by_pending_tool_use_id_hits_disk_snapshot(tmp_path) -> None:
    file_store = FileSessionStore(str(tmp_path))
    disk_store = SessionStore(file_store)
    disk_state = disk_store.get_or_create(
        session_id="claude-session-1",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
    )
    disk_state.pending_permission = PendingPermission(
        tool_use_id="tool-1",
        tool_name="Bash",
        tool_input={"command": "pwd"},
    )
    disk_store._persist(disk_state)

    store = SessionStore(file_store)
    matched = store.find_by_pending_tool_use_id("tool-1")

    assert matched is not None
    assert matched.session_id == "claude-session-1"
    assert matched.pending_permission is not None
    assert matched.pending_permission.tool_use_id == "tool-1"


def test_session_store_find_by_active_user_question_tool_use_id_hits_disk_snapshot(tmp_path) -> None:
    file_store = FileSessionStore(str(tmp_path))
    disk_store = SessionStore(file_store)
    disk_state = disk_store.get_or_create(
        session_id="claude-session-ask",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
    )
    disk_state.tool_calls["tool-ask-1"] = ToolCallRecord(
        tool_use_id="tool-ask-1",
        name="AskUserQuestion",
        input={
            "questions": [
                {
                    "question": "要怎么处理？",
                    "options": [{"label": "直接删除"}],
                    "multiSelect": False,
                }
            ]
        },
        status=ToolStatus.RUNNING,
    )
    disk_store._persist(disk_state)

    store = SessionStore(file_store)
    matched = store.find_by_active_user_question_tool_use_id("tool-ask-1")

    assert matched is not None
    assert matched.session_id == "claude-session-ask"
    assert "tool-ask-1" in matched.tool_calls
    assert matched.tool_calls["tool-ask-1"].status == ToolStatus.RUNNING


def test_session_store_find_by_active_user_question_tool_use_id_accepts_waiting_for_approval_tool_without_pending_snapshot(
    tmp_path,
) -> None:
    file_store = FileSessionStore(str(tmp_path))
    disk_store = SessionStore(file_store)
    disk_state = disk_store.get_or_create(
        session_id="claude-session-ask-waiting",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
    )
    disk_state.phase = SessionPhase.WAITING_FOR_APPROVAL
    disk_state.tool_calls["tool-ask-waiting"] = ToolCallRecord(
        tool_use_id="tool-ask-waiting",
        name="AskUserQuestion",
        input={
            "questions": [
                {
                    "question": "要怎么处理？",
                    "options": [{"label": "直接删除"}],
                    "multiSelect": False,
                }
            ]
        },
        status=ToolStatus.WAITING_FOR_APPROVAL,
    )
    disk_store._persist(disk_state)

    store = SessionStore(file_store)
    matched = store.find_by_active_user_question_tool_use_id("tool-ask-waiting")

    assert matched is not None
    assert matched.session_id == "claude-session-ask-waiting"
    assert "tool-ask-waiting" in matched.tool_calls
    assert matched.tool_calls["tool-ask-waiting"].status == ToolStatus.WAITING_FOR_APPROVAL


def test_session_store_find_by_active_user_question_key_hits_disk_snapshot(tmp_path) -> None:
    file_store = FileSessionStore(str(tmp_path))
    disk_store = SessionStore(file_store)
    disk_state = disk_store.get_or_create(
        session_id="claude-session-ask-key",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
        user_id=1,
    )
    disk_state.tool_calls["tool-ask-key"] = ToolCallRecord(
        tool_use_id="tool-ask-key",
        name="AskUserQuestion",
        input={
            "questions": [
                {"question": "第一题", "options": [{"label": "A"}], "multiSelect": False},
                {"question": "第二题", "options": [{"label": "B"}], "multiSelect": False},
            ]
        },
        status=ToolStatus.RUNNING,
    )
    disk_store._persist(disk_state)

    store = SessionStore(file_store)
    matched = store.find_by_active_user_question_key("tool-ask-key:1")

    assert matched is not None
    assert matched.session_id == "claude-session-ask-key"
    assert "tool-ask-key" in matched.tool_calls


def test_mark_structured_user_question_emitted_does_not_regress_same_tool_cursor(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="claude-session-ask-key", workdir="/tmp")

    store.mark_structured_user_question_emitted("claude-session-ask-key", question_key="tool-ask-key:2")
    store.mark_structured_user_question_emitted("claude-session-ask-key", question_key="tool-ask-key:0")

    updated = store.get("claude-session-ask-key")
    assert updated is not None
    assert updated.structured_user_question_key == "tool-ask-key:2"


def test_session_store_find_by_terminal_id_strong_cached_hit_returns_without_repository_scan(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    state = store.get_or_create(
        session_id="claude-session-strong-cache",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
    )
    state.phase = SessionPhase.WAITING_FOR_APPROVAL
    state.pending_permission = PendingPermission(
        tool_use_id="tool-strong",
        tool_name="Bash",
        tool_input={"command": "pwd"},
    )
    state.turns.append(ConversationTurn(turn_id="turn-strong", role="assistant", text="等待授权", is_complete=True))

    def fail_if_repository_scanned():
        raise AssertionError("repository should not be scanned for strong cached hit")

    store._lookup._repository.list_states = fail_if_repository_scanned

    matched = store.find_by_terminal_id("user_1_8c393341f536")

    assert matched is state
    assert matched.session_id == "claude-session-strong-cache"


def test_session_store_find_by_terminal_id_scans_repository_for_non_strong_cached_hit(tmp_path) -> None:
    file_store = FileSessionStore(str(tmp_path))
    disk_store = SessionStore(file_store)
    persisted = disk_store.get_or_create(
        session_id="claude-session-persisted-pending",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
    )
    persisted.phase = SessionPhase.WAITING_FOR_APPROVAL
    persisted.pending_permission = PendingPermission(
        tool_use_id="tool-persisted",
        tool_name="Bash",
        tool_input={"command": "pwd"},
    )
    persisted.turns.append(ConversationTurn(turn_id="turn-persisted", role="assistant", text="等待授权", is_complete=True))
    disk_store._persist(persisted)

    store = SessionStore(file_store)
    cached = store.get_or_create(
        session_id="claude-session-cached-idle",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
    )
    cached.phase = SessionPhase.WAITING_FOR_INPUT

    matched = store.find_by_terminal_id("user_1_8c393341f536")

    assert matched is not None
    assert matched.session_id == "claude-session-persisted-pending"


def test_session_store_find_by_terminal_id_prefers_pending_active_state_over_newer_idle_state(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    now = utc_now()

    pending_state = store.get_or_create(
        session_id="claude-session-pending",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
    )
    pending_state.created_at = now - timedelta(minutes=10)
    pending_state.last_activity = now
    pending_state.phase = SessionPhase.WAITING_FOR_APPROVAL
    pending_state.pending_permission = PendingPermission(
        tool_use_id="tool-1",
        tool_name="Bash",
        tool_input={"command": "pwd"},
    )
    store._persist(pending_state)

    idle_state = store.get_or_create(
        session_id="claude-session-idle",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
    )
    idle_state.created_at = now
    idle_state.last_activity = now - timedelta(seconds=30)
    idle_state.phase = SessionPhase.WAITING_FOR_INPUT
    store._persist(idle_state)

    matched = store.find_by_terminal_id("user_1_8c393341f536")

    assert matched is not None
    assert matched.session_id == "claude-session-pending"
    assert matched.pending_permission is not None
    assert matched.pending_permission.tool_use_id == "tool-1"


def test_session_store_find_by_terminal_id_prefers_newer_waiting_session_over_stale_processing_session(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    now = utc_now()

    stale_processing = store.get_or_create(
        session_id="claude-session-stale-processing",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
    )
    stale_processing.created_at = now - timedelta(hours=2)
    stale_processing.last_activity = now - timedelta(hours=1)
    stale_processing.phase = SessionPhase.PROCESSING
    stale_processing.turns.append(
        ConversationTurn(
            turn_id="turn-old",
            role="assistant",
            text="旧回复",
            is_complete=True,
        )
    )
    store._persist(stale_processing)

    fresh_waiting = store.get_or_create(
        session_id="claude-session-fresh-waiting",
        workdir="/tmp",
        terminal_id="user_1_8c393341f536",
    )
    fresh_waiting.created_at = now - timedelta(minutes=5)
    fresh_waiting.last_activity = now
    fresh_waiting.phase = SessionPhase.WAITING_FOR_INPUT
    fresh_waiting.turns.append(
        ConversationTurn(
            turn_id="turn-new",
            role="assistant",
            text="新回复",
            is_complete=True,
        )
    )
    store._persist(fresh_waiting)

    matched = store.find_by_terminal_id("user_1_8c393341f536")

    assert matched is not None
    assert matched.session_id == "claude-session-fresh-waiting"


def test_session_store_returns_latest_completed_assistant_turn_id(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    state = store.get_or_create(session_id="claude-session-1", workdir="/tmp", terminal_id="user_1_8c393341f536")
    store.get_or_create(session_id="tgcli_user_1_8c393341f536", workdir="/tmp", terminal_id="user_1_8c393341f536")
    store.process(
        SessionEvent(
            session_id=state.session_id,
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
                        "text": "\n第一条\n",
                        "source": "jsonl",
                        "is_complete": True,
                        "started_at": "2026-04-16T10:00:01+00:00",
                        "ended_at": "2026-04-16T10:00:01+00:00",
                    },
                ],
                "tool_calls": {},
                "last_reply": "第一条",
                "last_reply_role": "assistant",
                "last_offset": 12,
            },
        )
    )

    turn_id = store.latest_completed_assistant_turn_id(
        terminal_id="user_1_8c393341f536",
        workdir="/tmp",
        claude_session_id="claude-session-1",
        fallback_session_id="tgcli_user_1_8c393341f536",
    )

    assert turn_id == "a1"


def test_session_store_file_synced_records_claude_identity_and_history(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))

    state = store.process(
        SessionEvent(
            session_id="claude-session-1",
            type=SessionEventType.HISTORY_LOADED,
            payload={
                "cwd": "/tmp/project",
                "claude_session_id": "claude-session-1",
                "turns": [
                    {
                        "turn_id": "a1",
                        "role": "assistant",
                        "text": "\n恢复后的回复\n",
                        "source": "jsonl",
                        "is_complete": True,
                        "started_at": "2026-04-16T10:00:01+00:00",
                        "ended_at": "2026-04-16T10:00:01+00:00",
                    }
                ],
                "tool_calls": {},
                "last_reply": "恢复后的回复",
                "last_reply_role": "assistant",
                "last_offset": 12,
            },
        )
    )

    assert state.claude_session_id == "claude-session-1"
    assert state.workdir == "/tmp/project"
    assert state.history_loaded is True
    assert state.phase == SessionPhase.WAITING_FOR_INPUT


def test_session_store_clear_detected_resets_runtime_state(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.process(SessionEvent(session_id="s1", type=SessionEventType.TURN_STARTED, payload={"turn_id": "turn-1", "role": "assistant"}))
    store.process(
        SessionEvent(
            session_id="s1", type=SessionEventType.PARSER_UPDATED, payload={"turn_id": "turn-1", "text": "\n你好\n", "is_complete": True}
        )
    )

    state = store.process(SessionEvent(session_id="s1", type=SessionEventType.CLEAR_DETECTED))

    assert state.turns == []
    assert state.tool_calls == {}
    assert state.pending_permission is None
    assert state.clear_detected is True
    assert state.phase == SessionPhase.WAITING_FOR_INPUT


def test_session_store_interrupt_detected_marks_running_tool(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.process(
        SessionEvent(
            session_id="claude-session-1",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "claude-session-1",
                "cwd": "/tmp",
                "event": "PreToolUse",
                "status": "processing",
                "tool": "Bash",
                "tool_input": {"command": "pwd"},
                "tool_use_id": "tool-1",
            },
        )
    )

    state = store.process(SessionEvent(session_id="claude-session-1", type=SessionEventType.INTERRUPT_DETECTED))

    assert state.interrupted is True
    assert state.pending_permission is None
    assert state.tool_calls["tool-1"].status.value == "interrupted"
    assert state.phase == SessionPhase.WAITING_FOR_INPUT


def test_session_store_stop_failure_moves_to_waiting_for_input(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.process(
        SessionEvent(
            session_id="claude-session-1",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "claude-session-1",
                "cwd": "/tmp",
                "event": "PreToolUse",
                "status": "processing",
                "tool": "Bash",
                "tool_input": {"command": "pwd"},
                "tool_use_id": "tool-1",
            },
        )
    )

    state = store.process(
        SessionEvent(
            session_id="claude-session-1",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "claude-session-1",
                "cwd": "/tmp",
                "event": "StopFailure",
                "status": "failed",
            },
        )
    )

    assert state.phase == SessionPhase.WAITING_FOR_INPUT
    assert state.interrupted is True
    assert state.tool_calls["tool-1"].status.value == "interrupted"


def test_session_store_stop_without_running_tool_does_not_mark_interrupted(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    state = store.process(
        SessionEvent(
            session_id="claude-session-1",
            type=SessionEventType.HOOK_RECEIVED,
            payload={
                "session_id": "claude-session-1",
                "cwd": "/tmp",
                "event": "Stop",
                "status": "waiting_for_input",
            },
        )
    )

    assert state.phase == SessionPhase.WAITING_FOR_INPUT
    assert state.interrupted is False


@pytest.mark.asyncio
async def test_session_store_wait_for_publish_notifies_cursor(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="claude-session-1")
    cursor = store.get_cursor("claude-session-1")

    waiter = asyncio.create_task(store.wait_for_publish("claude-session-1", since_cursor=cursor, timeout_sec=0.2))
    await asyncio.sleep(0)
    store.process(SessionEvent(session_id="claude-session-1", type=SessionEventType.SESSION_STARTED))

    assert await waiter is True
    assert store.get_cursor("claude-session-1") > cursor


@pytest.mark.asyncio
async def test_session_store_ack_only_does_not_wake_wait_for_publish(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="claude-session-1")
    cursor = store.get_cursor("claude-session-1")

    waiter = asyncio.create_task(store.wait_for_publish("claude-session-1", since_cursor=cursor, timeout_sec=0.01))
    await asyncio.sleep(0)
    store.mark_structured_reply_emitted("claude-session-1", turn_id="turn-1")

    assert await waiter is False
    assert store.get_cursor("claude-session-1") == cursor


def test_session_store_ack_updates_do_not_advance_cursor(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="claude-session-1")
    cursor = store.get_cursor("claude-session-1")

    store.mark_structured_reply_emitted("claude-session-1", turn_id="turn-1")
    assert store.get_cursor("claude-session-1") == cursor

    store.mark_structured_permission_emitted("claude-session-1", permission_key="tool-1:Bash")
    assert store.get_cursor("claude-session-1") == cursor


def test_session_store_get_or_create_existing_state_does_not_advance_cursor(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="claude-session-1", workdir="/tmp/one")
    cursor = store.get_cursor("claude-session-1")

    store.get_or_create(session_id="claude-session-1", workdir="/tmp/two", terminal_id="term-1")

    assert store.get_cursor("claude-session-1") == cursor


def test_session_store_save_checkpoint_does_not_advance_cursor(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="claude-session-1")
    cursor = store.get_cursor("claude-session-1")

    store.save_checkpoint("claude-session-1", ParserCheckpoint(last_offset=7))

    assert store.get_cursor("claude-session-1") == cursor


def test_session_store_reload_does_not_advance_cursor(tmp_path) -> None:
    first = SessionStore(FileSessionStore(str(tmp_path)))
    first.get_or_create(session_id="claude-session-1", workdir="/tmp")
    cursor = first.get_cursor("claude-session-1")

    second = SessionStore(FileSessionStore(str(tmp_path)))
    second.get_or_create(session_id="claude-session-1", workdir="/tmp")

    assert second.get_cursor("claude-session-1") == cursor


def test_file_session_store_writes_checkpoint_atomically(tmp_path) -> None:
    storage = FileSessionStore(str(tmp_path))
    checkpoint = ParserCheckpoint(last_offset=7, pending_buffer="abc")

    storage.save_checkpoint("s1", checkpoint)

    path = storage.cursor_path("s1")
    assert json.loads(path.read_text(encoding="utf-8"))["last_offset"] == 7
    assert not list(path.parent.glob("tmp*"))


def test_file_session_store_writes_state_and_conversation_atomically(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    store.get_or_create(session_id="s1", user_id=1, workdir="/tmp", terminal_id="user_1_8c393341f536")
    store.process(SessionEvent(session_id="s1", type=SessionEventType.TURN_STARTED, payload={"turn_id": "turn-1", "role": "assistant"}))
    store.process(
        SessionEvent(
            session_id="s1", type=SessionEventType.PARSER_UPDATED, payload={"turn_id": "turn-1", "text": "\n你好\n", "is_complete": True}
        )
    )

    storage = FileSessionStore(str(tmp_path))
    state_path = storage.state_path("s1")
    conversation_path = storage.conversation_path("s1")

    assert json.loads(state_path.read_text(encoding="utf-8"))["session_id"] == "s1"
    assert json.loads(conversation_path.read_text(encoding="utf-8"))[0]["text"] == "\n你好\n"
    assert not list(state_path.parent.glob("tmp*"))


@pytest.mark.asyncio
async def test_session_service_switch_rebuilds_terminal_id_when_workdir_changes(tmp_path) -> None:
    from app.services.session_service import SessionService

    old_workdir = str(tmp_path / "one")
    new_workdir = str(tmp_path / "two")

    class _Store:
        def __init__(self):
            self.session = None

        async def get(self, user_id: int):
            return self.session

        async def save(self, session):
            self.session = session

    store = _Store()
    service = SessionService(store)

    first, _ = await service.switch(
        user_id=1,
        provider="claude_code",
        workdir=old_workdir,
        terminal_mode=True,
        claude_chat_active=True,
    )
    first_terminal_id = first.terminal_id
    second, _ = await service.switch(user_id=1, workdir=new_workdir)

    assert second.workdir == new_workdir
    assert first_terminal_id != second.terminal_id


@pytest.mark.asyncio
async def test_session_service_clear_terminal_group_resets_owner_and_attached_contexts(tmp_path) -> None:
    from app.adapters.storage.file_session_context_store import FileSessionContextStore
    from app.services.session_service import SessionService

    service = SessionService(FileSessionContextStore(FileSessionStore(str(tmp_path))))
    owner, _ = await service.switch(
        user_id=1,
        provider="claude_code",
        workdir="/proj",
        terminal_mode=True,
        claude_chat_active=True,
    )
    owner.terminal_id = "user_1_abc123"
    owner.claude_session_id = "claude-owner"
    owner.attached_user_ids = [2]
    await service.save_session_context(owner)

    attached, _ = await service.switch(
        user_id=2,
        provider="claude_code",
        workdir="/proj",
        terminal_mode=True,
        claude_chat_active=True,
    )
    attached.terminal_id = "user_1_abc123"
    attached.claude_session_id = "claude-attached"
    attached.is_owner = False
    await service.save_session_context(attached)

    other, _ = await service.switch(
        user_id=3,
        provider="claude_code",
        workdir="/other",
        terminal_mode=True,
        claude_chat_active=True,
    )
    other.terminal_id = "user_3_other"
    other.claude_session_id = "claude-other"
    other.attached_user_ids = [4]
    await service.save_session_context(other)

    assert await service.lookup_by_claude_session_id("claude-owner") is not None
    assert await service.lookup_by_claude_session_id("claude-attached") is not None

    affected = await service.clear_terminal_group("user_1_abc123")

    assert set(affected) == {1, 2}
    owner = await service.get(1)
    attached = await service.get(2)
    other = await service.get(3)
    assert owner is not None
    assert owner.terminal_mode is False
    assert owner.terminal_id is None
    assert owner.claude_chat_active is False
    assert owner.claude_session_id is None
    assert owner.attached_user_ids == []
    assert owner.is_owner is True
    assert attached is not None
    assert attached.terminal_mode is False
    assert attached.terminal_id is None
    assert attached.claude_chat_active is False
    assert attached.claude_session_id is None
    assert attached.attached_user_ids == []
    assert attached.is_owner is True
    assert other is not None
    assert other.terminal_mode is True
    assert other.terminal_id == "user_3_other"
    assert other.claude_chat_active is True
    assert other.claude_session_id == "claude-other"
    assert other.attached_user_ids == [4]
    assert await service.lookup_by_claude_session_id("claude-owner") is None
    assert await service.lookup_by_claude_session_id("claude-attached") is None
    assert await service.lookup_by_claude_session_id("claude-other") is not None


def test_session_store_get_or_create_does_not_clobber_valid_workdir_with_default_dot(tmp_path) -> None:
    store = SessionStore(FileSessionStore(str(tmp_path)))
    state = store.get_or_create(session_id="claude-session-1", workdir="/tmp/project")
    assert state.workdir == "/tmp/project"

    # 调用 get_or_create 传入缺省 "." 时，绝不能冲掉已有的真实目录
    updated = store.get_or_create(session_id="claude-session-1", workdir=".")
    assert updated.workdir == "/tmp/project"

    # 重新从磁盘加载 hydrates 时，也不被缺省 "." 冲掉
    reloaded = SessionStore(FileSessionStore(str(tmp_path)))
    state_reloaded = reloaded.get_or_create(session_id="claude-session-1", workdir=".")
    assert state_reloaded.workdir == "/tmp/project"
