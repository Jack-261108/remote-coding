"""External Ghostty dispatch in UserQuestionService (design §C / §9 / §10).

When an active external (Ghostty) pending question matches a managed
AskUserQuestion tool_call, the four top-level answer methods dispatch to the
external transport instead of the managed terminal, fail-closed (never falling
back to the managed terminal on REJECTED/INDETERMINATE), and orchestrate the
final ``allow`` exactly once on the last question.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.adapters.storage.file_session_store import FileSessionStore
from app.adapters.storage.memory import MemoryTaskStore
from app.domain.session_models import PendingPermission, SessionPhase, ToolCallRecord, ToolStatus
from app.domain.user_question_models import (
    ExternalQuestionActionResult,
    ExternalQuestionActionStatus,
    ExternalTerminalQuestionTarget,
    ExternalUserQuestionContext,
    ExternalUserQuestionPhase,
    UserQuestionOption,
    UserQuestionPrompt,
)
from app.services.external_user_question_state import (
    ExternalUserQuestionState,
    PendingExternalUserQuestion,
)
from app.services.session_store import SessionStore
from app.services.task_service import TaskService
from app.services.user_question_callback_registry import (
    UserQuestionCallbackNotFound,
    UserQuestionCallbackOrigin,
    UserQuestionCallbackRegistry,
)
from tests.fakes.cli import StubAdapter, StubFactory, make_file_backed_session_service, make_settings


def _pending(tool_use_id: str, session_id: str, prompts: tuple[UserQuestionPrompt, ...]) -> PendingExternalUserQuestion:
    return PendingExternalUserQuestion(
        tool_use_id=tool_use_id,
        session_id=session_id,
        user_id=1,
        prompts=prompts,
        target=ExternalTerminalQuestionTarget(
            binding_id="bind-1",
            terminal_id="term-1",
            paired_tty="/dev/ttys005",
            paired_at=datetime(2024, 1, 1, tzinfo=UTC),
            kind="ghostty",
        ),
    )


def _single_question_state(
    structured_store: SessionStore,
    session_id: str,
    workdir: str,
    tool_use_id: str = "tool-ask-1",
) -> None:
    state = structured_store.get_or_create(
        session_id=session_id,
        workdir=workdir,
        terminal_id="user_1_t",
        claude_session_id=session_id,
    )
    state.phase = SessionPhase.PROCESSING
    state.tool_calls[tool_use_id] = ToolCallRecord(
        tool_use_id=tool_use_id,
        name="AskUserQuestion",
        input={
            "questions": [
                {
                    "question": "Pick one",
                    "options": [{"label": "A"}, {"label": "B"}],
                    "multiSelect": False,
                }
            ]
        },
        status=ToolStatus.RUNNING,
    )
    state.pending_permission = PendingPermission(
        tool_use_id=tool_use_id,
        tool_name="AskUserQuestion",
        tool_input={"questions": [{"question": "Pick one", "options": [{"label": "A"}, {"label": "B"}]}]},
    )
    structured_store._persist(state)


def _two_question_state(
    structured_store: SessionStore,
    session_id: str,
    workdir: str,
    tool_use_id: str = "tool-ask-multi",
) -> None:
    state = structured_store.get_or_create(
        session_id=session_id,
        workdir=workdir,
        terminal_id="user_1_t",
        claude_session_id=session_id,
    )
    state.phase = SessionPhase.PROCESSING
    state.tool_calls[tool_use_id] = ToolCallRecord(
        tool_use_id=tool_use_id,
        name="AskUserQuestion",
        input={
            "questions": [
                {"question": "Q1", "options": [{"label": "A"}, {"label": "B"}], "multiSelect": False},
                {"question": "Q2", "options": [{"label": "C"}, {"label": "D"}], "multiSelect": False},
            ]
        },
        status=ToolStatus.RUNNING,
    )
    structured_store._persist(state)


class _RecordingTransport:
    """Minimal fake transport that records calls and returns APPLIED by default.

    Mimics the real input-service phase transitions so the UserQuestionService
    final-allow orchestration behaves against an ACTIVE→TERMINAL→COMPLETED
    lifecycle: ``select_option``/``answer_with_text`` advance to
    TERMINAL_ACTION_APPLIED (only meaningful on the final submit),
    ``question_completed`` advances to COMPLETED.
    """

    def __init__(
        self,
        state: ExternalUserQuestionState,
        *,
        result: ExternalQuestionActionStatus = ExternalQuestionActionStatus.APPLIED,
    ) -> None:
        self._state = state
        self._result = result
        self.select_calls: list[dict] = []
        self.text_calls: list[dict] = []
        self.completed: list[ExternalUserQuestionContext] = []
        self.indeterminate: list[tuple[ExternalUserQuestionContext, str]] = []

    async def select_option(self, *, context, question_index, option_count, option_index, submit_after) -> ExternalQuestionActionResult:  # noqa: ANN001
        self.select_calls.append(
            {
                "context": context,
                "question_index": question_index,
                "option_count": option_count,
                "option_index": option_index,
                "submit_after": submit_after,
            }
        )
        if submit_after:
            self._state.mark_terminal_action_applied(tool_use_id=context.tool_use_id, expected_target=context.target)
        return ExternalQuestionActionResult(self._result)

    async def answer_with_text(self, *, context, question_index, option_count, text, submit_after) -> ExternalQuestionActionResult:  # noqa: ANN001
        self.text_calls.append({"context": context, "text": text, "submit_after": submit_after})
        if submit_after:
            self._state.mark_terminal_action_applied(tool_use_id=context.tool_use_id, expected_target=context.target)
        return ExternalQuestionActionResult(self._result)

    async def advance_after_multi_select(self, *, context, question_index, option_count, final_question) -> ExternalQuestionActionResult:  # noqa: ANN001
        if final_question:
            self._state.mark_terminal_action_applied(tool_use_id=context.tool_use_id, expected_target=context.target)
        return ExternalQuestionActionResult(self._result)

    async def question_completed(self, *, context) -> None:  # noqa: ANN001
        self.completed.append(context)
        self._state.mark_completed(tool_use_id=context.tool_use_id, expected_target=context.target)

    async def question_indeterminate(self, *, context, reason) -> None:  # noqa: ANN001
        self.indeterminate.append((context, reason))
        self._state.mark_indeterminate(tool_use_id=context.tool_use_id, expected_target=context.target, reason=reason)


async def _build_service(tmp_path: Path) -> tuple[TaskService, StubFactory, SessionStore]:
    from unittest.mock import AsyncMock

    adapter = StubAdapter(events=[])
    factory = StubFactory(adapter)
    session_service = make_file_backed_session_service(tmp_path)
    structured_store = SessionStore(FileSessionStore(str(tmp_path)))
    service = TaskService(
        settings=make_settings(tmp_path, claude_tmux_mode=True),
        task_store=MemoryTaskStore(),
        session_service=session_service,
        cli_factory=factory,
        semaphore=asyncio.Semaphore(2),
        structured_session_store=structured_store,
    )
    # session_service.switch / bind use a tmux runner; the StubFactory's tmux runner
    # is updated by make_file_backed_session_service. Avoid network by stubbing the
    # interactive transport the managed path would touch — but for external dispatch
    # we only assert the managed path is NOT used, so a benign stub suffices.
    await session_service.switch(user_id=1, provider="claude_code", workdir=str(tmp_path), terminal_mode=True, claude_chat_active=True)
    await session_service.bind_claude_session(user_id=1, claude_session_id="claude-session-1", workdir=str(tmp_path))
    # Wire a no-op hook socket so _approve can call respond_to_permission_outcome.
    # Default to "wrote" (allow delivered) so the standard success path needs no
    # per-test setup; the not_pending/write_failed outcomes are set explicitly.
    hook_socket = AsyncMock()
    hook_socket.respond_to_permission_outcome.return_value = "wrote"
    service._user_question_service._hook_socket_server = hook_socket  # type: ignore[attr-defined]
    return service, factory, structured_store


@pytest.mark.asyncio
async def test_external_single_select_final_calls_transport_allow_and_completes(tmp_path: Path) -> None:
    service, factory, structured_store = await _build_service(tmp_path)
    session_id = "claude-session-1"
    _single_question_state(structured_store, session_id, str(tmp_path))
    prompts = (
        UserQuestionPrompt(
            tool_use_id="tool-ask-1",
            question_index=0,
            total_questions=1,
            question="Pick one",
            options=(UserQuestionOption(label="A"), UserQuestionOption(label="B")),
        ),
    )
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending("tool-ask-1", session_id, prompts))
    registry = UserQuestionCallbackRegistry(ttl_sec=60)
    await registry.register_question_tokens(
        owner_user_id=1,
        session_id=session_id,
        tool_use_id="tool-ask-1",
        question_index=0,
        option_count=2,
        multi_select=False,
        origin=UserQuestionCallbackOrigin.EXTERNAL_TERMINAL,
    )
    transport = _RecordingTransport(state)
    service.configure_external(external_uq_state=state, external_question_transport=transport, callback_registry=registry)

    ok, text, next_prompt = await service.answer_pending_user_question_option(
        user_id=1, tool_use_id="tool-ask-1", question_index=0, option_index=0
    )

    assert ok is True
    assert "Claude 继续执行中" in text
    assert next_prompt is None
    assert transport.select_calls and transport.select_calls[0]["submit_after"] is True
    service._user_question_service._hook_socket_server.respond_to_permission_outcome.assert_awaited_once_with(  # type: ignore[attr-defined,union-attr]
        tool_use_id="tool-ask-1", decision="allow"
    )
    assert transport.completed
    assert state.get_active("tool-ask-1") is None
    # Fail-closed: managed terminal never invoked.
    assert factory._interactive_inputs == []
    assert factory._user_question_option_actions == []


@pytest.mark.asyncio
async def test_external_rejected_does_not_allow_hook_or_complete(tmp_path: Path) -> None:
    service, factory, structured_store = await _build_service(tmp_path)
    session_id = "claude-session-1"
    _single_question_state(structured_store, session_id, str(tmp_path))
    prompts = (
        UserQuestionPrompt(
            tool_use_id="tool-ask-1",
            question_index=0,
            total_questions=1,
            question="Pick one",
            options=(UserQuestionOption(label="A"), UserQuestionOption(label="B")),
        ),
    )
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending("tool-ask-1", session_id, prompts))
    registry = UserQuestionCallbackRegistry(ttl_sec=60)
    transport = _RecordingTransport(state, result=ExternalQuestionActionStatus.REJECTED)
    service.configure_external(external_uq_state=state, external_question_transport=transport, callback_registry=registry)

    ok, text, next_prompt = await service.answer_pending_user_question_option(
        user_id=1, tool_use_id="tool-ask-1", question_index=0, option_index=0
    )

    assert ok is False
    assert next_prompt is None
    assert transport.completed == []
    service._user_question_service._hook_socket_server.respond_to_permission_outcome.assert_not_awaited()  # type: ignore[attr-defined,union-attr]
    assert factory._user_question_option_actions == []


@pytest.mark.asyncio
async def test_external_intermediate_question_does_not_allow_hook(tmp_path: Path) -> None:
    service, factory, structured_store = await _build_service(tmp_path)
    session_id = "claude-session-1"
    _two_question_state(structured_store, session_id, str(tmp_path))
    prompts = (
        UserQuestionPrompt(
            tool_use_id="tool-ask-multi",
            question_index=0,
            total_questions=2,
            question="Q1",
            options=(UserQuestionOption(label="A"), UserQuestionOption(label="B")),
        ),
        UserQuestionPrompt(
            tool_use_id="tool-ask-multi",
            question_index=1,
            total_questions=2,
            question="Q2",
            options=(UserQuestionOption(label="C"), UserQuestionOption(label="D")),
        ),
    )
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending("tool-ask-multi", session_id, prompts))
    registry = UserQuestionCallbackRegistry(ttl_sec=60)
    transport = _RecordingTransport(state)
    service.configure_external(external_uq_state=state, external_question_transport=transport, callback_registry=registry)

    ok, text, next_prompt = await service.answer_pending_user_question_option(
        user_id=1, tool_use_id="tool-ask-multi", question_index=0, option_index=0
    )

    assert ok is True
    assert next_prompt is not None
    assert next_prompt.question_index == 1
    assert transport.select_calls and transport.select_calls[0]["submit_after"] is False
    assert transport.completed == []
    service._user_question_service._hook_socket_server.respond_to_permission_outcome.assert_not_awaited()  # type: ignore[attr-defined,union-attr]


@pytest.mark.asyncio
async def test_external_single_select_without_managed_state_still_dispatches(tmp_path: Path) -> None:
    """Regression: a bound+paired Ghostty Claude session has NO managed
    structured tool_call state (external sessions don't go through the
    structured SessionStore). ``_resolve_active_user_question_context`` returns
    empty prompts, so the option button must fall back to the external
    pending's prompts instead of reporting “当前没有待处理的选择题”.
    Mirrors the real-world failure seen on bound Ghostty sessions.
    """
    service, factory, structured_store = await _build_service(tmp_path)
    session_id = "claude-session-1"
    # Intentionally NOT calling _single_question_state — a Ghostty binding does
    # not surface an AskUserQuestion tool_call through the structured store.
    prompts = (
        UserQuestionPrompt(
            tool_use_id="tool-ask-1",
            question_index=0,
            total_questions=1,
            question="Pick one",
            options=(UserQuestionOption(label="A"), UserQuestionOption(label="B")),
        ),
    )
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending("tool-ask-1", session_id, prompts))
    registry = UserQuestionCallbackRegistry(ttl_sec=60)
    transport = _RecordingTransport(state)
    service.configure_external(external_uq_state=state, external_question_transport=transport, callback_registry=registry)

    ok, text, next_prompt = await service.answer_pending_user_question_option(
        user_id=1, tool_use_id="tool-ask-1", question_index=0, option_index=0
    )

    assert ok is True, text
    assert "Claude 继续执行中" in text
    assert next_prompt is None
    assert transport.select_calls and transport.select_calls[0]["submit_after"] is True
    service._user_question_service._hook_socket_server.respond_to_permission_outcome.assert_awaited_once_with(  # type: ignore[attr-defined,union-attr]
        tool_use_id="tool-ask-1", decision="allow"
    )
    assert transport.completed
    assert state.get_active("tool-ask-1") is None
    # Fail-closed: managed terminal never invoked.
    assert factory._interactive_inputs == []
    assert factory._user_question_option_actions == []


@pytest.mark.asyncio
async def test_external_final_allows_when_hook_pending_already_closed_by_terminal(tmp_path: Path) -> None:
    """Regression race: the transport delivers the TUI answer, Claude reads it,
    resumes and emits PostToolUse, whose terminal-resolution path closes the
    pending permission connection *before* ``_approve_external_pending_permission``
    runs. ``respond_to_permission_outcome`` therefore returns ``"not_pending"``
    (pop found nothing). The answer reached Claude and Claude continued, so this
    is a successful approval in fact — we must NOT report “待处理权限请求已失效”
    and must still complete the question and invalidate the pending.
    """
    service, factory, structured_store = await _build_service(tmp_path)
    session_id = "claude-session-1"
    prompts = (
        UserQuestionPrompt(
            tool_use_id="tool-ask-1",
            question_index=0,
            total_questions=1,
            question="Pick one",
            options=(UserQuestionOption(label="A"), UserQuestionOption(label="B")),
        ),
    )
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending("tool-ask-1", session_id, prompts))
    registry = UserQuestionCallbackRegistry(ttl_sec=60)
    transport = _RecordingTransport(state)
    service.configure_external(external_uq_state=state, external_question_transport=transport, callback_registry=registry)
    # Simulate the PostToolUse having already closed the pending permission.
    service._user_question_service._hook_socket_server.respond_to_permission_outcome.return_value = "not_pending"  # type: ignore[attr-defined,union-attr]

    ok, text, next_prompt = await service.answer_pending_user_question_option(
        user_id=1, tool_use_id="tool-ask-1", question_index=0, option_index=0
    )

    assert ok is True, text
    assert "Claude 继续执行中" in text
    assert "失效" not in text
    assert next_prompt is None
    # Hook allow was still attempted exactly once.
    service._user_question_service._hook_socket_server.respond_to_permission_outcome.assert_awaited_once_with(  # type: ignore[attr-defined,union-attr]
        tool_use_id="tool-ask-1", decision="allow"
    )
    # Completion path still runs despite the closed connection.
    assert transport.completed
    assert state.get_active("tool-ask-1") is None
    assert factory._user_question_option_actions == []


@pytest.mark.asyncio
async def test_external_final_indeterminate_when_hook_write_fails(tmp_path: Path) -> None:
    """Distinct from the PostToolUse race: here the pending connection EXISTS when
    we try to allow it, but the ``allow`` response write/drain raises. Claude is
    still blocked on this ``allow`` — so ``respond_to_permission_outcome`` returns
    ``"write_failed"`` and we must NOT claim success: question stays alive-rejected
    (``question_indeterminate``, not ``question_completed``), tokens are cleared,
    Hook allow is attempted exactly once, the managed terminal is never invoked.
    """
    service, factory, structured_store = await _build_service(tmp_path)
    session_id = "claude-session-1"
    prompts = (
        UserQuestionPrompt(
            tool_use_id="tool-ask-1",
            question_index=0,
            total_questions=1,
            question="Pick one",
            options=(UserQuestionOption(label="A"), UserQuestionOption(label="B")),
        ),
    )
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending("tool-ask-1", session_id, prompts))
    registry = UserQuestionCallbackRegistry(ttl_sec=60)
    transport = _RecordingTransport(state)
    service.configure_external(external_uq_state=state, external_question_transport=transport, callback_registry=registry)
    service._user_question_service._hook_socket_server.respond_to_permission_outcome.return_value = "write_failed"  # type: ignore[attr-defined,union-attr]

    ok, text, next_prompt = await service.answer_pending_user_question_option(
        user_id=1, tool_use_id="tool-ask-1", question_index=0, option_index=0
    )

    assert ok is False, text
    assert next_prompt is None
    assert transport.completed == []
    assert transport.indeterminate, "expected indeterminate, not silent failure"
    assert transport.indeterminate[0][1]  # non-empty reason
    assert state.get("tool-ask-1") is not None
    assert state.get("tool-ask-1").phase is ExternalUserQuestionPhase.INDETERMINATE  # type: ignore[union-attr]
    service._user_question_service._hook_socket_server.respond_to_permission_outcome.assert_awaited_once_with(  # type: ignore[attr-defined,union-attr]
        tool_use_id="tool-ask-1", decision="allow"
    )
    # Fail-closed: managed terminal never invoked.
    assert factory._interactive_inputs == []
    assert factory._user_question_option_actions == []


@pytest.mark.asyncio
async def test_free_text_prefers_managed_pending_over_external_ghostty(tmp_path: Path) -> None:
    """Regression (#5): when the same user has BOTH an active managed
    AskUserQuestion (tool-ask-managed) AND an active external Ghostty question
    (tool-ask-external), a typed free-text answer must resolve to the MANAGED
    pending — never hijacked to the external Ghostty terminal. Previously
    ``answer_pending_user_question_text`` probed the external target first (by
    owner only, ignoring the user's intent), so the text landed in Ghostty and
    the managed Claude session's permission request was never answered.
    """
    service, factory, structured_store = await _build_service(tmp_path)
    session_id = "claude-session-1"
    # At least one managed AskUserQuestion pending via the structured store.
    _single_question_state(structured_store, session_id, str(tmp_path), tool_use_id="tool-ask-managed")

    # A simultaneously-active external Ghostty question for the same user.
    external_prompts = (
        UserQuestionPrompt(
            tool_use_id="tool-ask-external",
            question_index=0,
            total_questions=1,
            question="External pick",
            options=(UserQuestionOption(label="X"), UserQuestionOption(label="Y")),
        ),
    )
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending("tool-ask-external", session_id, external_prompts))
    registry = UserQuestionCallbackRegistry(ttl_sec=60)
    transport = _RecordingTransport(state)
    service.configure_external(external_uq_state=state, external_question_transport=transport, callback_registry=registry)

    ok, text, next_prompt = await service.answer_pending_user_question_text(user_id=1, text="my-other-answer")

    assert ok is True, text
    # The critical #5 invariant: the external Ghostty transport is NOT touched.
    # The free-text answer went to the managed pending, not the external terminal.
    assert transport.text_calls == []
    assert transport.select_calls == []
    assert transport.completed == []
    # The external Ghostty pending remains ACTIVE (untouched) so the user can
    # still answer it later via buttons — no hijack, no premature completion.
    external_pending = state.get_active("tool-ask-external")
    assert external_pending is not None
    assert external_pending.phase is ExternalUserQuestionPhase.ACTIVE
    # The managed pending permission allow fired for the MANAGED tool_use_id
    # (``respond_to_permission``, not the external ``..._outcome``), confirming
    # the managed session's Claude is unblocked — not the external one.
    service._user_question_service._hook_socket_server.respond_to_permission.assert_awaited_once_with(  # type: ignore[attr-defined,union-attr]
        tool_use_id="tool-ask-managed", decision="allow"
    )


@pytest.mark.asyncio
async def test_external_multi_select_toggle_and_intermediate_advance(tmp_path: Path) -> None:
    """Multi-select external workflow: a toggle on question 0 selects an option
    and the intermediate submit advances to question 1 without completing or
    allowing the Hook (design §9 intermediate-question semantics).
    """
    service, factory, structured_store = await _build_service(tmp_path)
    session_id = "claude-session-1"
    prompts = (
        UserQuestionPrompt(
            tool_use_id="tool-ask-multi2",
            question_index=0,
            total_questions=2,
            question="Q1 many",
            options=(UserQuestionOption(label="A"), UserQuestionOption(label="B")),
            multi_select=True,
        ),
        UserQuestionPrompt(
            tool_use_id="tool-ask-multi2",
            question_index=1,
            total_questions=2,
            question="Q2 many",
            options=(UserQuestionOption(label="C"), UserQuestionOption(label="D")),
            multi_select=True,
        ),
    )
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending("tool-ask-multi2", session_id, prompts))
    registry = UserQuestionCallbackRegistry(ttl_sec=60)
    transport = _RecordingTransport(state)
    service.configure_external(external_uq_state=state, external_question_transport=transport, callback_registry=registry)

    # Toggle Q1 option A.
    ok, text, next_prompt, selected = await service.toggle_pending_user_question_multi_select_option(
        user_id=1, tool_use_id="tool-ask-multi2", question_index=0, option_index=0
    )
    assert ok is True, text
    assert selected == frozenset({0})
    # Toggle Q1 option B (now two selected).
    ok, text, next_prompt, selected = await service.toggle_pending_user_question_multi_select_option(
        user_id=1, tool_use_id="tool-ask-multi2", question_index=0, option_index=1
    )
    assert ok is True, text
    assert selected == frozenset({0, 1})
    assert transport.completed == []

    # Submit Q1 → intermediate: advances to Q2, no Hook allow yet.
    ok, text, next_prompt = await service.submit_pending_user_question_multi_select(
        user_id=1, tool_use_id="tool-ask-multi2", question_index=0
    )
    assert ok is True, text
    assert next_prompt is not None and next_prompt.question_index == 1
    service._user_question_service._hook_socket_server.respond_to_permission_outcome.assert_not_awaited()  # type: ignore[attr-defined,union-attr]
    # Managed terminal never invoked.
    assert factory._interactive_inputs == []
    assert factory._user_question_option_actions == []


@pytest.mark.asyncio
async def test_external_multi_select_final_submit_allows_and_completes(tmp_path: Path) -> None:
    """Second (final) multi-select question submit must send the allow to the
    Hook exactly once and complete the question, advancing the external state.
    """
    service, factory, structured_store = await _build_service(tmp_path)
    session_id = "claude-session-1"
    prompts = (
        UserQuestionPrompt(
            tool_use_id="tool-ask-multi3",
            question_index=0,
            total_questions=1,
            question="Final many",
            options=(UserQuestionOption(label="A"), UserQuestionOption(label="B")),
            multi_select=True,
        ),
    )
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending("tool-ask-multi3", session_id, prompts))
    registry = UserQuestionCallbackRegistry(ttl_sec=60)
    transport = _RecordingTransport(state)
    service.configure_external(external_uq_state=state, external_question_transport=transport, callback_registry=registry)

    await service.toggle_pending_user_question_multi_select_option(
        user_id=1, tool_use_id="tool-ask-multi3", question_index=0, option_index=0
    )
    ok, text, next_prompt = await service.submit_pending_user_question_multi_select(
        user_id=1, tool_use_id="tool-ask-multi3", question_index=0
    )

    assert ok is True, text
    assert next_prompt is None
    service._user_question_service._hook_socket_server.respond_to_permission_outcome.assert_awaited_once_with(  # type: ignore[attr-defined,union-attr]
        tool_use_id="tool-ask-multi3", decision="allow"
    )
    assert transport.completed
    assert state.get_active("tool-ask-multi3") is None
    assert factory._user_question_option_actions == []


@pytest.mark.asyncio
async def test_external_multi_select_toggle_when_not_configured(tmp_path: Path) -> None:
    """External multi-select toggle when ``configure_external`` wired no transport
    must fail closed with “外部 Ghostty 问答功能未启用”.
    """
    service, factory, structured_store = await _build_service(tmp_path)
    session_id = "claude-session-1"
    prompts = (
        UserQuestionPrompt(
            tool_use_id="tool-ask-ms-off",
            question_index=0,
            total_questions=1,
            question="Pick many",
            options=(UserQuestionOption(label="A"), UserQuestionOption(label="B")),
            multi_select=True,
        ),
    )
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending("tool-ask-ms-off", session_id, prompts))
    service.configure_external(external_uq_state=state, external_question_transport=None, callback_registry=None)

    ok, text, next_prompt, selected = await service.toggle_pending_user_question_multi_select_option(
        user_id=1, tool_use_id="tool-ask-ms-off", question_index=0, option_index=0
    )

    assert ok is False, text
    assert next_prompt is None
    assert selected is None


@pytest.mark.asyncio
async def test_external_final_indeterminate_transport_invalidates_tokens(tmp_path: Path) -> None:
    """INDETERMINATE transport result must not complete or allow; it must clear
    the callback registry tokens so the stale buttons stop working.

    Distinguishes the transport-action uncertainty from REJECTED: the question is
    left alive (not COMPLETED) and the Hook allow is never attempted.
    """
    service, factory, structured_store = await _build_service(tmp_path)
    session_id = "claude-session-1"
    prompts = (
        UserQuestionPrompt(
            tool_use_id="tool-ask-1",
            question_index=0,
            total_questions=1,
            question="Pick one",
            options=(UserQuestionOption(label="A"), UserQuestionOption(label="B")),
        ),
    )
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending("tool-ask-1", session_id, prompts))
    registry = UserQuestionCallbackRegistry(ttl_sec=60)
    tokens = await registry.register_question_tokens(
        owner_user_id=1,
        session_id=session_id,
        tool_use_id="tool-ask-1",
        question_index=0,
        option_count=2,
        multi_select=False,
        origin=UserQuestionCallbackOrigin.EXTERNAL_TERMINAL,
    )
    transport = _RecordingTransport(state, result=ExternalQuestionActionStatus.INDETERMINATE)
    service.configure_external(external_uq_state=state, external_question_transport=transport, callback_registry=registry)

    ok, text, next_prompt = await service.answer_pending_user_question_option(
        user_id=1, tool_use_id="tool-ask-1", question_index=0, option_index=0
    )

    assert ok is False, text
    assert next_prompt is None
    assert transport.completed == []
    # INDETERMINATE → tokens invalidated (buttons dead): resolve reports NotFound.
    resolved = await registry.resolve(tokens.select_tokens[0], user_id=1)
    assert isinstance(resolved, UserQuestionCallbackNotFound)
    assert factory._interactive_inputs == []
    assert factory._user_question_option_actions == []


@pytest.mark.asyncio
async def test_external_approve_hookless_is_approved(tmp_path: Path) -> None:
    """A hookless wiring (``_hook_socket_server`` is None) treats the external
    allow as approved, so a final select completes without hanging.
    """
    service, factory, structured_store = await _build_service(tmp_path)
    session_id = "claude-session-1"
    prompts = (
        UserQuestionPrompt(
            tool_use_id="tool-ask-6",
            question_index=0,
            total_questions=1,
            question="Pick one",
            options=(UserQuestionOption(label="A"), UserQuestionOption(label="B")),
        ),
    )
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending("tool-ask-6", session_id, prompts))
    registry = UserQuestionCallbackRegistry(ttl_sec=60)
    transport = _RecordingTransport(state)
    service.configure_external(external_uq_state=state, external_question_transport=transport, callback_registry=registry)
    # Remove the hook socket: the approve helper must treat it as approved.
    service._user_question_service._hook_socket_server = None  # type: ignore[attr-defined]

    ok, text, next_prompt = await service.answer_pending_user_question_option(
        user_id=1, tool_use_id="tool-ask-6", question_index=0, option_index=0
    )

    assert ok is True, text
    assert "Claude 继续执行中" in text
    assert next_prompt is None
    assert transport.completed
    assert state.get_active("tool-ask-6") is None
