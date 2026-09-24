import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.types import InaccessibleMessage

from app.adapters.process.tmux_runner import TmuxRunner
from app.adapters.storage.file_session_context_store import FileSessionContextStore
from app.adapters.storage.file_session_store import FileSessionStore
from app.adapters.storage.memory import MemoryTaskStore
from app.bot.handlers.command_permission import register_permission_handlers
from app.bot.handlers.command_session import register_session_handler
from app.bot.handlers.command_status import register_status_handler
from app.bot.handlers.command_user_question import _is_accessible_message, register_user_question_handlers
from app.bot.handlers.external_permission import register_external_permission_handler
from app.bot.router import create_router
from app.domain.models import TaskRecord, TaskStatus
from app.domain.session_models import (
    ConversationTurn,
    SessionEvent,
    SessionEventType,
    SessionPhase,
    ToolCallRecord,
    ToolStatus,
)
from app.domain.user_question_models import ExternalTmuxQuestionTarget, UserQuestionOption, UserQuestionPrompt
from app.services.background_task_registry import BackgroundTaskRegistry
from app.services.external_user_question_state import ExternalUserQuestionState, PendingExternalUserQuestion
from app.services.session_service import SessionService
from app.services.session_store import SessionStore
from app.services.task_service import TaskService
from app.services.user_question_callback_registry import (
    UserQuestionCallbackOrigin,
    UserQuestionCallbackRegistry,
)
from tests.fakes.cli import StubAdapter, StubFactory, make_settings
from tests.fakes.telegram import DummyCallbackQuery, DummyMessage


def _find_handler_by_name(router, name: str):
    """Search main router and sub-routers for a handler by function name."""
    for h in router.message.handlers:
        if h.callback.__name__ == name:
            return h.callback
    for sub in router.sub_routers:
        result = _find_handler_by_name(sub, name)
        if result is not None:
            return result
    return None


def _find_callback_handler_by_name(router, name: str):
    """Search main router and sub-routers for a callback handler by function name."""
    for handler in router.callback_query.handlers:
        if handler.callback.__name__ == name:
            return handler.callback
    for sub in router.sub_routers:
        result = _find_callback_handler_by_name(sub, name)
        if result is not None:
            return result
    return None


class _DummyEventObserver:
    def __init__(self, handlers: list) -> None:
        self._handlers = handlers
        self.middlewares = []

    def __call__(self, *args, **kwargs):
        def decorator(fn):
            self._handlers.append(fn)
            return fn

        return decorator

    def middleware(self, middleware) -> None:
        self.middlewares.append(middleware)


class DummyRouter:
    def __init__(self) -> None:
        self.handlers = []
        self.callback_handlers = []
        self.message = _DummyEventObserver(self.handlers)
        self.callback_query = _DummyEventObserver(self.callback_handlers)


async def _call_callback(router: DummyRouter, index: int, callback) -> None:
    """Dispatch a callback to a DummyRouter handler, injecting callback_parts when needed."""
    handler = router.callback_handlers[index]
    kwargs = {}
    if callback.data:
        sig = inspect.signature(handler)
        if "callback_parts" in sig.parameters:
            kwargs["callback_parts"] = tuple(callback.data.split(":"))
    await handler(callback, **kwargs)


def _tmux_runner_with_session_store(tmp_path) -> TmuxRunner:
    file_store = FileSessionStore(str(tmp_path))
    return TmuxRunner(data_dir=str(tmp_path), file_store=file_store, session_store=SessionStore(file_store))


def _session_service(tmp_path) -> SessionService:
    return SessionService(FileSessionContextStore(FileSessionStore(str(tmp_path))))


@pytest.fixture
def stream_background_tasks() -> BackgroundTaskRegistry:
    return BackgroundTaskRegistry(label="stream")


def test_is_accessible_message_rejects_inaccessible_message() -> None:
    inaccessible = InaccessibleMessage.model_construct(chat=None, message_id=1, date=0)

    assert _is_accessible_message(inaccessible) is False
    assert _is_accessible_message(DummyMessage("Question")) is True


@pytest.mark.asyncio
async def test_session_handler_renders_structured_snapshot(tmp_path) -> None:
    tmux_runner = _tmux_runner_with_session_store(tmp_path)
    factory = StubFactory(StubAdapter(events=[]))
    factory._tmux_runner = tmux_runner
    factory._claude_tmux_enabled = True
    factory.get_claude_session_state = lambda session_id: tmux_runner.get_session_state(session_id)
    service = TaskService(
        settings=make_settings(tmp_path, claude_tmux_mode=True),
        task_store=MemoryTaskStore(),
        session_service=_session_service(tmp_path),
        cli_factory=factory,
        semaphore=asyncio.Semaphore(1),
        structured_session_store=tmux_runner._session_store,
    )
    session_service = service._session_service
    session, _ = await session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await service.bind_claude_session(user_id=1, claude_session_id="claude-session-1", workdir=str(tmp_path))
    state = tmux_runner._session_store.get_or_create(
        session_id="claude-session-1",
        workdir=str(tmp_path),
        terminal_id=session.terminal_id,
    )
    tmux_runner._session_store.process(SessionEvent(session_id="claude-session-1", type=SessionEventType.SESSION_STARTED))
    tmux_runner._session_store.process(
        SessionEvent(session_id="claude-session-1", type=SessionEventType.TURN_STARTED, payload={"turn_id": "turn-1", "role": "assistant"})
    )
    state.turns[-1].text = "\n你好\n"
    state.turns[-1].is_complete = True
    state = tmux_runner._session_store.process(
        SessionEvent(session_id="claude-session-1", type=SessionEventType.TURN_COMPLETED, payload={"turn_id": "turn-1"})
    )
    state.phase = SessionPhase.WAITING_FOR_INPUT
    tmux_runner._session_store._persist(state)

    router = DummyRouter()
    register_session_handler(router, task_service=service, session_service=session_service)
    handler = router.handlers[0]
    message = DummyMessage("/session")

    await handler(message)

    assert message.answers
    assert "structured_session:" in message.answers[0]
    assert "phase: waiting_for_input" in message.answers[0]
    assert "last_reply: 你好" in message.answers[0]
    assert "terminal_mode" not in message.answers[0]
    assert "terminal_id" not in message.answers[0]


@pytest.mark.asyncio
async def test_session_handler_cleans_orphaned_terminal_resources(tmp_path) -> None:
    factory = StubFactory(StubAdapter(events=[]))
    service = TaskService(
        settings=make_settings(tmp_path),
        task_store=MemoryTaskStore(),
        session_service=_session_service(tmp_path),
        cli_factory=factory,
        semaphore=asyncio.Semaphore(1),
    )
    session_service = service._session_service
    old_workdir = tmp_path / "old"
    new_workdir = tmp_path / "new"
    old_workdir.mkdir()
    new_workdir.mkdir()
    original, _ = await session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(old_workdir),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await session_service.bind_claude_session(
        user_id=1,
        claude_session_id="claude-session-old",
    )
    assert original.terminal_id is not None
    old_terminal_id = original.terminal_id
    cleanup = AsyncMock()
    service.cleanup_orphaned_terminal = cleanup

    router = DummyRouter()
    register_session_handler(router, task_service=service, session_service=session_service)
    handler = router.handlers[0]
    message = DummyMessage(f"/session claude_code {new_workdir}")

    await handler(message)

    cleanup.assert_awaited_once_with(
        old_terminal_id,
        claude_session_id="claude-session-old",
        user_id=1,
    )
    assert message.answers
    assert "session 已更新" in message.answers[0]


@pytest.mark.asyncio
async def test_session_handler_rejects_missing_workdir(tmp_path) -> None:
    factory = StubFactory(StubAdapter(events=[]))
    service = TaskService(
        settings=make_settings(tmp_path),
        task_store=MemoryTaskStore(),
        session_service=_session_service(tmp_path),
        cli_factory=factory,
        semaphore=asyncio.Semaphore(1),
    )
    session_service = service._session_service
    missing_workdir = tmp_path / "missing"

    router = DummyRouter()
    register_session_handler(router, task_service=service, session_service=session_service)
    handler = router.handlers[0]
    message = DummyMessage(f"/session claude_code {missing_workdir}")

    await handler(message)

    assert message.answers == [f"workdir 不存在或不是目录: {missing_workdir.resolve()}"]
    assert await session_service.get(1) is None


@pytest.mark.asyncio
async def test_status_handler_renders_structured_snapshot(tmp_path) -> None:
    tmux_runner = _tmux_runner_with_session_store(tmp_path)
    factory = StubFactory(StubAdapter(events=[]))
    factory._tmux_runner = tmux_runner
    factory._claude_tmux_enabled = True
    factory.get_claude_session_state = lambda session_id: tmux_runner.get_session_state(session_id)
    service = TaskService(
        settings=make_settings(tmp_path, claude_tmux_mode=True),
        task_store=MemoryTaskStore(),
        session_service=_session_service(tmp_path),
        cli_factory=factory,
        semaphore=asyncio.Semaphore(1),
        structured_session_store=tmux_runner._session_store,
    )
    await service._session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await service.bind_claude_session(user_id=1, claude_session_id="claude-session-1", workdir=str(tmp_path))
    state = tmux_runner._session_store.get_or_create(
        session_id="claude-session-1", workdir=str(tmp_path), terminal_id="user_1_36d00faeb25f"
    )
    state.phase = SessionPhase.WAITING_FOR_INPUT
    state.turns.append(ConversationTurn(turn_id="turn-1", role="assistant", text="\n你好\n", is_complete=True))
    tmux_runner._session_store._persist(state)

    record = TaskRecord(
        task_id="task-1",
        session_id="session-1",
        user_id=1,
        provider="claude_code",
        prompt="hi",
        workdir=str(tmp_path),
        timeout_sec=10,
        claude_session_id="claude-session-1",
        status=TaskStatus.SUCCEEDED,
    )
    await service._task_store.add(record)

    router = DummyRouter()
    register_status_handler(router, task_service=service)
    handler = router.handlers[0]
    message = DummyMessage("/status task-1")
    command = SimpleNamespace(args="task-1")

    await handler(message, command)

    assert message.answers
    assert "structured_session:" in message.answers[0]
    assert "phase: waiting_for_input" in message.answers[0]
    assert "terminal_mode" not in message.answers[0]
    assert "terminal_id" not in message.answers[0]


@pytest.mark.asyncio
async def test_permission_handlers_delegate_text_commands_to_gateway() -> None:
    class FakeGateway:
        def __init__(self) -> None:
            self.approve_user_ids: list[int] = []
            self.deny_calls: list[tuple[int, str | None]] = []

        async def handle_approve_command(self, *, user_id: int) -> str:
            self.approve_user_ids.append(user_id)
            return "approved via gateway"

        async def handle_deny_command(self, *, user_id: int, reason: str | None = None) -> str:
            self.deny_calls.append((user_id, reason))
            return "denied via gateway"

    gateway = FakeGateway()
    router = DummyRouter()
    register_permission_handlers(router, permission_gateway=gateway)

    approve_message = DummyMessage("/approve", user_id=11)
    await router.handlers[0](approve_message)
    deny_message = DummyMessage("/deny no", user_id=22)
    await router.handlers[1](deny_message, SimpleNamespace(args="no"))

    assert gateway.approve_user_ids == [11]
    assert gateway.deny_calls == [(22, "no")]
    assert approve_message.answers == ["approved via gateway"]
    assert deny_message.answers == ["denied via gateway"]


@pytest.mark.asyncio
async def test_permission_callback_handler_delegates_to_gateway_response() -> None:
    class FakeGateway:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        async def handle_callback(self, *, data: str, user_id: int):  # noqa: ANN202
            self.calls.append((data, user_id))
            return SimpleNamespace(
                alert_text="gateway alert",
                show_alert=True,
                edit_message_text="",
                clear_keyboard=True,
            )

    gateway = FakeGateway()
    router = DummyRouter()
    register_permission_handlers(router, permission_gateway=gateway)
    message = DummyMessage("权限请求")
    callback = DummyCallbackQuery("perm:tok12345:allow", user_id=7, message=message)

    await router.callback_handlers[0](callback)

    assert gateway.calls == [("perm:tok12345:allow", 7)]
    assert message.edited_reply_markups == [None]
    assert callback.answers == [("gateway alert", True)]


@pytest.mark.asyncio
async def test_external_permission_callback_delegates_reformatted_payload_to_gateway() -> None:
    class FakeGateway:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        async def handle_callback(self, *, data: str, user_id: int):  # noqa: ANN202
            self.calls.append((data, user_id))
            return SimpleNamespace(alert_text="已批准", show_alert=False, edit_message_text="", clear_keyboard=True)

    gateway = FakeGateway()
    router = DummyRouter()
    register_external_permission_handler(
        router,
        hook_socket_server=SimpleNamespace(respond_to_permission=AsyncMock(return_value=True)),
        unbound_permission_handler=SimpleNamespace(),
        permission_gateway=gateway,
    )

    message = DummyMessage("Permission request")
    callback = DummyCallbackQuery("ext_perm:tok12345:allow", user_id=321, message=message)

    await _call_callback(router, 0, callback)

    assert gateway.calls == [("perm:tok12345:allow", 321)]
    assert callback.answers == [("已批准", False)]
    assert message.edited_reply_markups == [None]


@pytest.mark.asyncio
async def test_external_user_question_callback_still_uses_existing_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_inject(pane_id: str, *, option_index: int, submit_after: bool, tmux_bin: str = "tmux") -> tuple[bool, str]:
        assert pane_id == "%1"
        assert option_index == 0
        assert submit_after is True
        return True, ""

    monkeypatch.setattr("app.adapters.process.pty_injector.inject_option_selection", fake_inject)

    hook_socket_server = SimpleNamespace(respond_to_permission=AsyncMock(return_value=True))
    external_uq_state = ExternalUserQuestionState()
    external_uq_state.store(
        PendingExternalUserQuestion(
            tool_use_id="tool-question",
            session_id="external-session",
            user_id=1,
            target=ExternalTmuxQuestionTarget(pane_id="%1"),
            prompts=(
                UserQuestionPrompt(
                    tool_use_id="tool-question",
                    question_index=0,
                    total_questions=1,
                    question="Pick one",
                    options=(UserQuestionOption(label="A"),),
                ),
            ),
        )
    )
    registry = UserQuestionCallbackRegistry(ttl_sec=300)
    tokens = await registry.register_question_tokens(
        owner_user_id=1,
        session_id="external-session",
        tool_use_id="tool-question",
        question_index=0,
        option_count=1,
        multi_select=False,
        origin=UserQuestionCallbackOrigin.EXTERNAL_TMUX,
    )
    token = tokens.select_tokens[0]

    router = DummyRouter()
    register_external_permission_handler(
        router,
        hook_socket_server=hook_socket_server,
        unbound_permission_handler=SimpleNamespace(),
        permission_gateway=SimpleNamespace(handle_callback=AsyncMock()),
        external_uq_state=external_uq_state,
        user_question_callback_registry=registry,
    )

    message = DummyMessage("Question")
    callback = DummyCallbackQuery(f"ext_uq:{token}", user_id=1, message=message)

    await _call_callback(router, 1, callback)

    hook_socket_server.respond_to_permission.assert_awaited_once_with(
        tool_use_id="tool-question",
        decision="allow",
        reason="AskUserQuestion answered via Telegram by user 1",
    )
    assert callback.answers == [("✅ Selected: A", False)]
    assert external_uq_state.get("tool-question") is None


@pytest.mark.asyncio
async def test_external_tmux_user_question_concurrent_callbacks_are_serialized_and_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inject_calls: list[int] = []

    async def fake_inject(pane_id: str, *, option_index: int, submit_after: bool, tmux_bin: str = "tmux") -> tuple[bool, str]:
        inject_calls.append(option_index)
        await asyncio.sleep(0.05)
        return True, ""

    monkeypatch.setattr("app.adapters.process.pty_injector.inject_option_selection", fake_inject)

    hook_socket_server = SimpleNamespace(respond_to_permission=AsyncMock(return_value=True))
    external_uq_state = ExternalUserQuestionState()
    registry = UserQuestionCallbackRegistry(ttl_sec=300)

    prompt = UserQuestionPrompt(
        tool_use_id="tool-concurrent",
        question_index=0,
        total_questions=1,
        question="Choice",
        header="H",
        options=(UserQuestionOption(label="Opt0"), UserQuestionOption(label="Opt1")),
        multi_select=False,
    )
    external_uq_state.store(
        PendingExternalUserQuestion(
            tool_use_id="tool-concurrent",
            session_id="external-session",
            user_id=1,
            target=ExternalTmuxQuestionTarget(pane_id="%1"),
            prompts=(prompt,),
        )
    )
    tokens = await registry.register_question_tokens(
        owner_user_id=1,
        session_id="external-session",
        tool_use_id="tool-concurrent",
        question_index=0,
        option_count=2,
        multi_select=False,
        origin=UserQuestionCallbackOrigin.EXTERNAL_TMUX,
    )
    token0 = tokens.select_tokens[0]
    token1 = tokens.select_tokens[1]

    router = DummyRouter()
    register_external_permission_handler(
        router,
        hook_socket_server=hook_socket_server,
        unbound_permission_handler=SimpleNamespace(),
        permission_gateway=SimpleNamespace(handle_callback=AsyncMock()),
        external_uq_state=external_uq_state,
        user_question_callback_registry=registry,
    )

    cb0 = DummyCallbackQuery(f"ext_uq:{token0}", user_id=1, message=DummyMessage("Q"))
    cb1 = DummyCallbackQuery(f"ext_uq:{token1}", user_id=1, message=DummyMessage("Q"))

    await asyncio.gather(
        _call_callback(router, 1, cb0),
        _call_callback(router, 1, cb1),
    )

    assert len(inject_calls) == 1
    answers = [cb0.answers[0][0], cb1.answers[0][0]]
    assert any("Selected" in a for a in answers)
    assert any("already answered" in a for a in answers)
    hook_socket_server.respond_to_permission.assert_awaited_once()


async def _store_multi_prompt_external_question(
    *,
    registry: UserQuestionCallbackRegistry,
    external_uq_state: ExternalUserQuestionState,
    multi_select_prompt_one_options: int = 2,
    prompt_one_multi_select: bool = False,
) -> tuple[str, str, str, str]:
    """Store a 2-prompt AskUserQuestion and register EXTERNAL_TMUX tokens for both.

    Returns ``(tool_use_id, q0_option0_token, q1_option0_token, q1_option1_token)``.
    Mirrors what ``notify_user_question`` would emit: one single-choice card per
    prompt, each backed by opaque ``ext_uq:`` tokens (tmux renders multi-select
    prompts as single-choice per the injector's capability).
    """
    tool_use_id = "tool-multi"
    external_uq_state.store(
        PendingExternalUserQuestion(
            tool_use_id=tool_use_id,
            session_id="external-session",
            user_id=1,
            target=ExternalTmuxQuestionTarget(pane_id="%1"),
            prompts=(
                UserQuestionPrompt(
                    tool_use_id=tool_use_id,
                    question_index=0,
                    total_questions=2,
                    question="First pick",
                    options=(UserQuestionOption(label="A0"), UserQuestionOption(label="B0")),
                    multi_select=prompt_one_multi_select,
                ),
                UserQuestionPrompt(
                    tool_use_id=tool_use_id,
                    question_index=1,
                    total_questions=2,
                    question="Second pick",
                    options=(UserQuestionOption(label="A1"), UserQuestionOption(label="B1")),
                    multi_select=False,
                ),
            ),
        )
    )
    q0_tokens = await registry.register_question_tokens(
        owner_user_id=1,
        session_id="external-session",
        tool_use_id=tool_use_id,
        question_index=0,
        option_count=multi_select_prompt_one_options,
        multi_select=False,
        origin=UserQuestionCallbackOrigin.EXTERNAL_TMUX,
    )
    q1_tokens = await registry.register_question_tokens(
        owner_user_id=1,
        session_id="external-session",
        tool_use_id=tool_use_id,
        question_index=1,
        option_count=2,
        multi_select=False,
        origin=UserQuestionCallbackOrigin.EXTERNAL_TMUX,
    )
    return (
        tool_use_id,
        q0_tokens.select_tokens[0],
        q1_tokens.select_tokens[0],
        q1_tokens.select_tokens[1],
    )


@pytest.mark.asyncio
async def test_external_multi_prompt_callback_advances_without_submitting_or_allowing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answering an intermediate prompt (N>1) must NOT submit, NOT allow the
    Hook, NOT remove the pending record, and must push a fresh button card for
    the next prompt. This is the regression fixed after the prompts[0]/len==1
    bug that left multi-prompt external tmux sessions blocked on Claude's side.
    """
    submit_after_seen: list[bool] = []

    async def fake_inject(pane_id: str, *, option_index: int, submit_after: bool, tmux_bin: str = "tmux") -> tuple[bool, str]:
        assert pane_id == "%1"
        submit_after_seen.append(submit_after)
        return True, ""

    monkeypatch.setattr("app.adapters.process.pty_injector.inject_option_selection", fake_inject)

    hook_socket_server = SimpleNamespace(respond_to_permission=AsyncMock(return_value=True))
    external_uq_state = ExternalUserQuestionState()
    registry = UserQuestionCallbackRegistry(ttl_sec=300)
    tool_use_id, q0_token, _, _ = await _store_multi_prompt_external_question(
        registry=registry,
        external_uq_state=external_uq_state,
    )

    router = DummyRouter()
    register_external_permission_handler(
        router,
        hook_socket_server=hook_socket_server,
        unbound_permission_handler=SimpleNamespace(),
        permission_gateway=SimpleNamespace(handle_callback=AsyncMock()),
        external_uq_state=external_uq_state,
        user_question_callback_registry=registry,
    )

    message = DummyMessage("First pick")
    callback = DummyCallbackQuery(f"ext_uq:{q0_token}", user_id=1, message=message)

    await _call_callback(router, 1, callback)

    # Intermediate prompt: keystroke selected option 0 but did NOT submit.
    assert submit_after_seen == [False]
    # Hook allow must never fire for an intermediate question.
    hook_socket_server.respond_to_permission.assert_not_awaited()
    # The pending record survives — remaining prompts still need it.
    assert external_uq_state.get(tool_use_id) is not None
    # User is acknowledged for this prompt.
    assert callback.answers == [("✅ Selected: A0", False)]
    # A fresh button card for the next question was pushed with a non-empty
    # keyboard. ``edits`` also land in ``sent_messages``, so the answer card is
    # the last entry; ``answers`` tracks only fresh ``answer()`` calls.
    expected_next = (
        "❓ [external] 用户选择\n问题: Second pick\n选项:\n  1. A1\n  2. B1\n\n👇 点击按钮选择；可直接回复文字作为 Other/自由文本"
    )
    assert expected_next in message.answers
    answer_cards = [m for m in message.sent_messages if m.reply_markup is not None]
    assert len(answer_cards) == 1
    next_card = answer_cards[-1]
    assert next_card.text == expected_next
    next_buttons = [btn for row in next_card.reply_markup.inline_keyboard for btn in row]
    assert {btn.text for btn in next_buttons} == {"1. A1", "2. B1"}
    # The original card's keyboard was cleared.
    assert message.edited_reply_markups == [None]


@pytest.mark.asyncio
async def test_external_multi_prompt_callback_final_prompt_submits_allows_and_removes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answering the LAST prompt of a multi-prompt batch must submit the answer,
    allow the Hook permission, and remove the pending record — so Claude
    actually resumes instead of staying blocked.
    """
    submit_after_seen: list[bool] = []

    async def fake_inject(pane_id: str, *, option_index: int, submit_after: bool, tmux_bin: str = "tmux") -> tuple[bool, str]:
        submit_after_seen.append(submit_after)
        return True, ""

    monkeypatch.setattr("app.adapters.process.pty_injector.inject_option_selection", fake_inject)

    hook_socket_server = SimpleNamespace(respond_to_permission=AsyncMock(return_value=True))
    external_uq_state = ExternalUserQuestionState()
    registry = UserQuestionCallbackRegistry(ttl_sec=300)
    tool_use_id, _, q1_token_a, q1_token_b = await _store_multi_prompt_external_question(
        registry=registry,
        external_uq_state=external_uq_state,
    )

    router = DummyRouter()
    register_external_permission_handler(
        router,
        hook_socket_server=hook_socket_server,
        unbound_permission_handler=SimpleNamespace(),
        permission_gateway=SimpleNamespace(handle_callback=AsyncMock()),
        external_uq_state=external_uq_state,
        user_question_callback_registry=registry,
    )

    message = DummyMessage("Second pick")
    callback = DummyCallbackQuery(f"ext_uq:{q1_token_b}", user_id=1, message=message)

    await _call_callback(router, 1, callback)

    # Final prompt: option 1 (B1) selected AND submitted.
    assert submit_after_seen == [True]
    hook_socket_server.respond_to_permission.assert_awaited_once_with(
        tool_use_id=tool_use_id,
        decision="allow",
        reason="AskUserQuestion answered via Telegram by user 1",
    )
    assert callback.answers == [("✅ Selected: B1", False)]
    # Pending record removed after the final answer.
    assert external_uq_state.get(tool_use_id) is None
    # No next-question card pushed for the final prompt: only the annotation
    # edit was applied; no fresh ``answer`` carrying the next prompt text.
    assert message.answers == []
    assert message.edited_reply_markups == [None]
    # The unused q1 token-b is consumed; ensure no stale behavior
    _ = q1_token_a  # registered but not exercised in this scenario


@pytest.mark.asyncio
async def test_external_user_question_callback_rejects_invalidated_session(monkeypatch: pytest.MonkeyPatch) -> None:
    injected: list[str] = []

    async def fake_inject(pane_id: str, *, option_index: int, submit_after: bool, tmux_bin: str = "tmux") -> tuple[bool, str]:
        injected.append(pane_id)
        return True, ""

    monkeypatch.setattr("app.adapters.process.pty_injector.inject_option_selection", fake_inject)

    hook_socket_server = SimpleNamespace(respond_to_permission=AsyncMock(return_value=True))
    external_uq_state = ExternalUserQuestionState()
    external_uq_state.store(
        PendingExternalUserQuestion(
            tool_use_id="tool-question",
            session_id="external-session",
            user_id=1,
            target=ExternalTmuxQuestionTarget(pane_id="%1"),
            prompts=(
                UserQuestionPrompt(
                    tool_use_id="tool-question",
                    question_index=0,
                    total_questions=1,
                    question="Pick one",
                    options=(UserQuestionOption(label="A"),),
                ),
            ),
        )
    )
    assert external_uq_state.invalidate_session("external-session") == 1
    registry = UserQuestionCallbackRegistry(ttl_sec=300)
    tokens = await registry.register_question_tokens(
        owner_user_id=1,
        session_id="external-session",
        tool_use_id="tool-question",
        question_index=0,
        option_count=1,
        multi_select=False,
        origin=UserQuestionCallbackOrigin.EXTERNAL_TMUX,
    )
    token = tokens.select_tokens[0]

    router = DummyRouter()
    register_external_permission_handler(
        router,
        hook_socket_server=hook_socket_server,
        unbound_permission_handler=SimpleNamespace(),
        permission_gateway=SimpleNamespace(handle_callback=AsyncMock()),
        external_uq_state=external_uq_state,
        user_question_callback_registry=registry,
    )

    callback = DummyCallbackQuery(f"ext_uq:{token}", user_id=1, message=DummyMessage("Question"))

    await _call_callback(router, 1, callback)

    assert callback.answers == [("Question expired or already answered", True)]
    assert injected == []
    hook_socket_server.respond_to_permission.assert_not_awaited()


@pytest.mark.asyncio
async def test_user_question_callback_handler_records_choice_and_prompts_next_question(tmp_path) -> None:
    tmux_runner = _tmux_runner_with_session_store(tmp_path)
    factory = StubFactory(StubAdapter(events=[]))
    factory._tmux_runner = tmux_runner
    factory._claude_tmux_enabled = True
    factory.get_claude_session_state = lambda session_id: tmux_runner.get_session_state(session_id)
    service = TaskService(
        settings=make_settings(tmp_path, claude_tmux_mode=True),
        task_store=MemoryTaskStore(),
        session_service=_session_service(tmp_path),
        cli_factory=factory,
        semaphore=asyncio.Semaphore(1),
        structured_session_store=tmux_runner._session_store,
    )
    await service._session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await service.bind_claude_session(user_id=1, claude_session_id="claude-session-1", workdir=str(tmp_path))
    state = tmux_runner._session_store.get_or_create(
        session_id="claude-session-1", workdir=str(tmp_path), terminal_id="user_1_36d00faeb25f"
    )
    state.phase = SessionPhase.PROCESSING
    state.tool_calls["tool-ask-1"] = ToolCallRecord(
        tool_use_id="tool-ask-1",
        name="AskUserQuestion",
        input={
            "questions": [
                {
                    "header": "处理范围",
                    "question": "你说的范围我理解为这三块之一，具体按哪种处理？",
                    "options": [
                        {"label": "当前相关改动(推荐)", "description": "只处理相关已改动文件"},
                        {"label": "三个目录全部", "description": "范围非常大"},
                    ],
                    "multiSelect": False,
                },
                {
                    "header": "提交前置",
                    "question": "按你的 CLAUDE.md，要修改代码前先提交现有改动。现在是否允许我先做这一步？",
                    "options": [
                        {"label": "允许先提交(推荐)", "description": "先提交后继续"},
                        {"label": "暂不允许", "description": "先不改代码"},
                    ],
                    "multiSelect": False,
                },
            ]
        },
        status=ToolStatus.RUNNING,
    )
    tmux_runner._session_store._persist(state)

    router = DummyRouter()
    register_user_question_handlers(router, task_service=service, callback_registry=None)
    message = DummyMessage("需要你选择")
    callback = DummyCallbackQuery("ask:tool-ask-1:0:0", message=message)

    await _call_callback(router, 0, callback)

    assert factory._interactive_inputs == []
    assert factory._user_question_option_actions == [("user_1_36d00faeb25f", str(tmp_path), 0, False)]
    assert message.answers[0] == "已记录选择: 当前相关改动(推荐)"
    assert "问题: 2/2" in message.answers[1]
    assert callback.answers == [("已记录选择: 当前相关改动(推荐)", False)]
    assert tmux_runner._session_store.get("claude-session-1").structured_user_question_key == "tool-ask-1:1"


@pytest.mark.asyncio
async def test_user_question_callback_handler_rejects_cross_user_button(tmp_path) -> None:
    tmux_runner = _tmux_runner_with_session_store(tmp_path)
    factory = StubFactory(StubAdapter(events=[]))
    factory._tmux_runner = tmux_runner
    factory._claude_tmux_enabled = True
    factory.get_claude_session_state = lambda session_id: tmux_runner.get_session_state(session_id)
    service = TaskService(
        settings=make_settings(tmp_path, claude_tmux_mode=True),
        task_store=MemoryTaskStore(),
        session_service=_session_service(tmp_path),
        cli_factory=factory,
        semaphore=asyncio.Semaphore(1),
        structured_session_store=tmux_runner._session_store,
    )
    await service._session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await service.bind_claude_session(user_id=1, claude_session_id="claude-session-1", workdir=str(tmp_path))
    await service._session_service.switch(
        user_id=2,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await service.bind_claude_session(user_id=2, claude_session_id="claude-session-2", workdir=str(tmp_path))
    state = tmux_runner._session_store.get_or_create(
        session_id="claude-session-1",
        user_id=1,
        workdir=str(tmp_path),
        terminal_id="user_1_36d00faeb25f",
    )
    state.phase = SessionPhase.PROCESSING
    state.tool_calls["tool-ask-1"] = ToolCallRecord(
        tool_use_id="tool-ask-1",
        name="AskUserQuestion",
        input={
            "questions": [
                {
                    "header": "处理范围",
                    "question": "你说的范围我理解为这三块之一，具体按哪种处理？",
                    "options": [
                        {"label": "当前相关改动(推荐)", "description": "只处理相关已改动文件"},
                        {"label": "三个目录全部", "description": "范围非常大"},
                    ],
                    "multiSelect": False,
                }
            ]
        },
        status=ToolStatus.RUNNING,
    )
    tmux_runner._session_store._persist(state)

    router = DummyRouter()
    register_user_question_handlers(router, task_service=service, callback_registry=None)
    message = DummyMessage("需要你选择", user_id=2)
    callback = DummyCallbackQuery("ask:tool-ask-1:0:0", user_id=2, message=message)

    await _call_callback(router, 0, callback)

    assert factory._interactive_inputs == []
    assert factory._user_question_option_actions == []
    assert message.answers == ["选择失败: 当前没有待处理的选择题"]
    assert message.edited_reply_markups == []
    assert callback.answers == [("当前没有待处理的选择题", True)]
    assert tmux_runner._session_store.get("claude-session-1").structured_user_question_key is None


@pytest.mark.asyncio
async def test_user_question_callback_handler_toggles_multi_select_and_submits(tmp_path) -> None:
    tmux_runner = _tmux_runner_with_session_store(tmp_path)
    factory = StubFactory(StubAdapter(events=[]))
    factory._tmux_runner = tmux_runner
    factory._claude_tmux_enabled = True
    factory.get_claude_session_state = lambda session_id: tmux_runner.get_session_state(session_id)
    service = TaskService(
        settings=make_settings(tmp_path, claude_tmux_mode=True),
        task_store=MemoryTaskStore(),
        session_service=_session_service(tmp_path),
        cli_factory=factory,
        semaphore=asyncio.Semaphore(1),
        structured_session_store=tmux_runner._session_store,
    )
    session, _ = await service._session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await service.bind_claude_session(user_id=1, claude_session_id="claude-session-1", workdir=str(tmp_path))
    state = tmux_runner._session_store.get_or_create(
        session_id="claude-session-1",
        workdir=str(tmp_path),
        terminal_id=session.terminal_id,
    )
    state.phase = SessionPhase.PROCESSING
    state.tool_calls["tool-ask-multi"] = ToolCallRecord(
        tool_use_id="tool-ask-multi",
        name="AskUserQuestion",
        input={
            "questions": [
                {
                    "header": "处理方式",
                    "question": "这次要保留哪些动作？",
                    "options": [
                        {"label": "保留日志", "description": "继续输出调试日志"},
                        {"label": "保留测试", "description": "继续保留回归测试"},
                    ],
                    "multiSelect": True,
                }
            ]
        },
        status=ToolStatus.RUNNING,
    )
    tmux_runner._session_store._persist(state)

    router = DummyRouter()
    register_user_question_handlers(router, task_service=service, callback_registry=None)
    message = DummyMessage("需要你选择")

    toggle_callback = DummyCallbackQuery("ask:toggle:tool-ask-multi:0:0", message=message)
    await _call_callback(router, 0, toggle_callback)

    assert message.answers == []
    assert len(message.edited_reply_markups) == 1
    toggle_markup = message.edited_reply_markups[0]
    assert toggle_markup is not None
    assert [button.text for row in toggle_markup.inline_keyboard for button in row] == [
        "☑ 保留日志",
        "☐ 保留测试",
        "提交选择",
    ]
    assert toggle_callback.answers == [("已选择: 保留日志", False)]

    submit_callback = DummyCallbackQuery("ask:submit:tool-ask-multi:0", message=message)
    await _call_callback(router, 0, submit_callback)

    assert factory._interactive_inputs == []
    assert factory._user_question_option_actions == [(session.terminal_id, str(tmp_path), 0, False)]
    assert factory._user_question_multi_select_advances == [(session.terminal_id, str(tmp_path), True)]
    assert message.answers == ["已提交你的选择，Claude 继续执行中"]
    assert message.edited_reply_markups[-1] is None
    assert submit_callback.answers == [("已提交你的选择，Claude 继续执行中", False)]


@pytest.mark.asyncio
async def test_router_text_chat_answers_pending_user_question_instead_of_creating_new_task(
    tmp_path, monkeypatch: pytest.MonkeyPatch, stream_background_tasks: BackgroundTaskRegistry
) -> None:
    tmux_runner = _tmux_runner_with_session_store(tmp_path)
    factory = StubFactory(StubAdapter(events=[]))
    factory._tmux_runner = tmux_runner
    factory._claude_tmux_enabled = True
    factory.get_claude_session_state = lambda session_id: tmux_runner.get_session_state(session_id)
    session_service = _session_service(tmp_path)
    service = TaskService(
        settings=make_settings(tmp_path, claude_tmux_mode=True),
        task_store=MemoryTaskStore(),
        session_service=session_service,
        cli_factory=factory,
        semaphore=asyncio.Semaphore(1),
        structured_session_store=tmux_runner._session_store,
    )
    session, _ = await session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await service.bind_claude_session(user_id=1, claude_session_id="claude-session-1", workdir=str(tmp_path))
    state = tmux_runner._session_store.get_or_create(
        session_id="claude-session-1",
        workdir=str(tmp_path),
        terminal_id=session.terminal_id,
    )
    state.phase = SessionPhase.PROCESSING
    state.tool_calls["tool-ask-2"] = ToolCallRecord(
        tool_use_id="tool-ask-2",
        name="AskUserQuestion",
        input={
            "questions": [
                {
                    "header": "处理方式",
                    "question": "这两条误写到项目级的记忆，你要我怎么处理？",
                    "options": [
                        {"label": "迁到全局(推荐)", "description": "保留记忆内容并迁移"},
                        {"label": "直接删除", "description": "删除项目级这两条记忆"},
                    ],
                    "multiSelect": False,
                }
            ]
        },
        status=ToolStatus.RUNNING,
    )
    tmux_runner._session_store._persist(state)

    run_mock = AsyncMock()
    monkeypatch.setattr("app.bot.router.run_prompt_and_stream", run_mock)

    router = create_router(
        settings=make_settings(tmp_path, claude_tmux_mode=True),
        task_service=service,
        session_service=session_service,
        stream_background_tasks=stream_background_tasks,
    )
    text_handler = _find_handler_by_name(router, "command_claude_chat_text")
    assert text_handler is not None, "command_claude_chat_text handler not found"
    message = DummyMessage("直接删除")

    await text_handler(message, session=session)

    run_mock.assert_not_awaited()
    assert factory._interactive_inputs == []
    assert factory._user_question_option_actions == [(session.terminal_id, str(tmp_path), 1, True)]
    assert message.answers == ["已提交你的选择，Claude 继续执行中"]


@pytest.mark.asyncio
async def test_router_text_chat_awaits_background_stream_task(
    tmp_path, monkeypatch: pytest.MonkeyPatch, stream_background_tasks: BackgroundTaskRegistry
) -> None:
    tmux_runner = _tmux_runner_with_session_store(tmp_path)
    factory = StubFactory(StubAdapter(events=[]))
    factory._tmux_runner = tmux_runner
    factory._claude_tmux_enabled = True
    factory.get_claude_session_state = lambda session_id: tmux_runner.get_session_state(session_id)
    session_service = _session_service(tmp_path)
    service = TaskService(
        settings=make_settings(tmp_path, claude_tmux_mode=True),
        task_store=MemoryTaskStore(),
        session_service=session_service,
        cli_factory=factory,
        semaphore=asyncio.Semaphore(1),
        structured_session_store=tmux_runner._session_store,
    )
    session, _ = await session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )

    task = asyncio.create_task(asyncio.sleep(0))
    await task
    run_mock = AsyncMock(return_value=task)
    monkeypatch.setattr("app.bot.router.run_prompt_and_stream", run_mock)

    settings = make_settings(
        tmp_path,
        claude_tmux_mode=True,
        STRUCTURED_REPLY_PUMP_INTERVAL_SEC=0.77,
        SPINNER_INITIAL_DELAY_SEC=0.88,
        SPINNER_INTERVAL_SEC=0.99,
    )
    status_display = object()
    router = create_router(
        settings=settings,
        task_service=service,
        session_service=session_service,
        status_display=status_display,
        stream_background_tasks=stream_background_tasks,
    )
    text_handler = _find_handler_by_name(router, "command_claude_chat_text")
    assert text_handler is not None, "command_claude_chat_text handler not found"
    message = DummyMessage("你好")

    await text_handler(message, session=session)

    run_mock.assert_awaited_once()
    assert run_mock.await_args.kwargs["status_display"] is status_display
    assert run_mock.await_args.kwargs["structured_reply_pump_interval_sec"] == 0.77
    assert run_mock.await_args.kwargs["spinner_initial_delay_sec"] == 0.88
    assert run_mock.await_args.kwargs["spinner_interval_sec"] == 0.99


@pytest.mark.asyncio
async def test_router_clcmd_callback_passes_permission_gateway_to_stream(
    tmp_path, monkeypatch: pytest.MonkeyPatch, stream_background_tasks: BackgroundTaskRegistry
) -> None:
    session_service = _session_service(tmp_path)
    service = SimpleNamespace(available_providers=lambda: ["claude_code"])
    session, _ = await session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    permission_gateway = object()
    run_mock = AsyncMock()
    monkeypatch.setattr("app.bot.handlers.command_run.run_prompt_and_stream", run_mock)

    router = create_router(
        settings=make_settings(tmp_path),
        task_service=service,
        session_service=session_service,
        permission_gateway=permission_gateway,
        stream_background_tasks=stream_background_tasks,
    )
    callback_handler = _find_callback_handler_by_name(router, "handle_cmd_callback")
    assert callback_handler is not None, "handle_cmd_callback handler not found"
    callback = DummyCallbackQuery("clcmd:/verify", message=DummyMessage("/cmds"))

    await callback_handler(callback, session=session)

    run_mock.assert_awaited_once()
    assert run_mock.await_args.kwargs["prompt"] == "/verify"
    assert run_mock.await_args.kwargs["provider"] == "claude_code"
    assert run_mock.await_args.kwargs["workdir"] == str(tmp_path)
    assert run_mock.await_args.kwargs["permission_gateway"] is permission_gateway
