"""External Ghostty free-text router filter (design §G).

``ExternalQuestionActiveFilter`` decides whether a plain-text message routes
to the external-question answer handler (instead of falling through to the
managed chat or external text routers). It matches only when:

* there is a non-empty, non-slash text message;
* the user has NO managed pending question (so managed tmux questions are
  never hijacked by this earlier router — the chief regression risk);
* there is at least one active Ghostty external question for the user — an
  ambiguous state also passes so the handler can tell the user to answer via
  buttons; returning False would let the text fall through and be injected
  into a terminal waiting on a question.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Chat, Message, User

from app.bot.router import ExternalQuestionActiveFilter, answer_external_user_question_text
from app.domain.user_question_models import (
    ExternalTerminalQuestionTarget,
    UserQuestionOption,
    UserQuestionPrompt,
)
from app.services.external_user_question_state import (
    ExternalUserQuestionState,
    PendingExternalUserQuestion,
)
from app.services.user_question_callback_registry import QuestionCallbackTokens


def _message(text: str | None, *, user_id: int = 1) -> Message:
    return Message(
        message_id=1,
        date=datetime(2024, 1, 1, tzinfo=UTC),
        chat=Chat(id=1, type="private"),
        from_user=User(id=user_id, is_bot=False, first_name="t"),
        text=text,
    )


class _FakeTaskService:
    def __init__(self, pending: list | None = None) -> None:
        self._pending = pending if pending is not None else []
        self.answer_text_calls: list[dict] = []
        self.answer_text_result: tuple[bool, str, object] = (True, "ok", None)
        self.acknowledge_calls: list[str] = []

    async def get_pending_user_questions(self, user_id: int) -> list:  # noqa: ARG002
        return list(self._pending)

    async def answer_pending_user_question_text(  # noqa: PLR6301
        self, *, user_id: int, text: str
    ) -> tuple[bool, str, object]:
        self.answer_text_calls.append({"user_id": user_id, "text": text})
        return self.answer_text_result  # type: ignore[return-value]

    async def register_question_callback_tokens(  # noqa: PLR6301
        self,
        *,
        user_id: int,
        prompt: UserQuestionPrompt,  # noqa: ARG002
    ) -> QuestionCallbackTokens:
        # No registry wired in the router-level test: a non-tokenised bundle makes
        # ``build_user_question_keyboard`` fall back to legacy inline callback_data,
        # which is exactly the path we need to assert (buttons ARE rendered).
        return QuestionCallbackTokens()

    async def acknowledge_structured_user_question(  # noqa: PLR6301
        self,
        user_id: int,
        *,
        question_key: str | None = None,  # noqa: ARG002
    ) -> None:
        if question_key is not None:
            self.acknowledge_calls.append(question_key)


class _ScopedTaskService:
    """TaskService whose pending set can flip after routing.

    Simulates a managed AskUserQuestion appearing in the filter→handler window:
    the filter sees an empty managed set (so routing proceeds), then the handler
    re-checks and must observe the now-non-empty set and fail closed.
    """

    def __init__(self) -> None:
        self._pending: list = []
        self.answer_text_calls: list[dict] = []

    async def get_pending_user_questions(self, user_id: int) -> list:  # noqa: ARG002
        return list(self._pending)

    async def answer_pending_user_question_text(  # noqa: PLR6301
        self, *, user_id: int, text: str
    ) -> tuple[bool, str, object]:
        self.answer_text_calls.append({"user_id": user_id, "text": text})
        return (True, "ok", None)


def _pending(tool_use_id: str = "tuid-1", user_id: int = 1) -> PendingExternalUserQuestion:
    return PendingExternalUserQuestion(
        tool_use_id=tool_use_id,
        session_id="sess-1",
        user_id=user_id,
        prompts=(
            UserQuestionPrompt(
                tool_use_id=tool_use_id,
                question_index=0,
                total_questions=1,
                question="Pick?",
                options=(UserQuestionOption(label="A"),),
            ),
        ),
        target=ExternalTerminalQuestionTarget(
            binding_id="bind-1",
            terminal_id="term-1",
            paired_tty="/dev/ttys005",
            paired_at=datetime(2024, 1, 1, tzinfo=UTC),
            kind="ghostty",
        ),
    )


@pytest.mark.asyncio
async def test_match_when_unique_ghostty_question_and_no_managed_pending() -> None:
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending(user_id=1))
    task_service = _FakeTaskService(pending=[])
    filt = ExternalQuestionActiveFilter(state, task_service)  # type: ignore[arg-type]

    assert await filt(_message("some answer", user_id=1)) is True


@pytest.mark.asyncio
async def test_no_match_when_managed_pending_present() -> None:
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending(user_id=1))
    task_service = _FakeTaskService(pending=[object()])  # managed tmux question active
    filt = ExternalQuestionActiveFilter(state, task_service)  # type: ignore[arg-type]

    # Even with a unique Ghostty question, a managed pending blocks this router
    # so the managed question is answered instead.
    assert await filt(_message("some answer", user_id=1)) is False


@pytest.mark.asyncio
async def test_match_when_ambiguous_multiple_ghostty_questions_routes_to_handler() -> None:
    """Ambiguity must route INTO the handler, not fall through.

    The handler re-resolves and answers "存在多个待处理的外部问题" (see
    ``test_handler_rejects_when_multiple_ghostty_questions``). If the filter
    returned False here, the text would fall through to the external-terminal
    router and be injected into a terminal that is waiting on a question.
    """
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending(tool_use_id="tuid-a", user_id=1))
    state.store(_pending(tool_use_id="tuid-b", user_id=1))
    task_service = _FakeTaskService(pending=[])
    filt = ExternalQuestionActiveFilter(state, task_service)  # type: ignore[arg-type]

    assert await filt(_message("some answer", user_id=1)) is True


@pytest.mark.asyncio
async def test_no_match_when_no_active_question() -> None:
    state = ExternalUserQuestionState(ttl_sec=60)
    task_service = _FakeTaskService(pending=[])
    filt = ExternalQuestionActiveFilter(state, task_service)  # type: ignore[arg-type]

    assert await filt(_message("some answer", user_id=1)) is False


@pytest.mark.asyncio
async def test_no_match_for_slash_command() -> None:
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending(user_id=1))
    task_service = _FakeTaskService(pending=[])
    filt = ExternalQuestionActiveFilter(state, task_service)  # type: ignore[arg-type]

    # A registered slash is handled by an earlier command router; an unregistered
    # slash falls through rather than being consumed as a free-text answer here.
    assert await filt(_message("/compact", user_id=1)) is False


@pytest.mark.asyncio
async def test_no_match_when_no_text() -> None:
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending(user_id=1))
    task_service = _FakeTaskService(pending=[])
    filt = ExternalQuestionActiveFilter(state, task_service)  # type: ignore[arg-type]

    assert await filt(_message(None, user_id=1)) is False


@pytest.mark.asyncio
async def test_no_match_when_state_is_none() -> None:
    task_service = _FakeTaskService(pending=[])
    filt = ExternalQuestionActiveFilter(None, task_service)

    assert await filt(_message("some answer", user_id=1)) is False


@pytest.mark.asyncio
async def test_no_match_for_other_user() -> None:
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending(user_id=1))
    task_service = _FakeTaskService(pending=[])
    filt = ExternalQuestionActiveFilter(state, task_service)  # type: ignore[arg-type]

    # The Ghostty question belongs to user 1; user 2 must not have their text consumed.
    assert await filt(_message("some answer", user_id=2)) is False


# --- handler integration (design §F/§G) ---
# The free-text handler mirrors ExternalQuestionActiveFilter. These tests drive
# ``answer_external_user_question_text`` directly (router closure calls it) so the
# managed-pending re-check and stale/ambiguous guards are unit-verifiable without
# standing up a full Aiogram dispatcher.


def _patched_message(text: str, *, user_id: int = 1) -> Message:
    msg = _message(text, user_id=user_id)
    # Aiogram Message is a pydantic frozen model, so assign via __dict__ to stub answer.
    object.__setattr__(msg, "answer", AsyncMock())
    return msg


@pytest.mark.asyncio
async def test_handler_consumes_free_text_when_no_managed_pending() -> None:
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending(user_id=1))
    task_service = _FakeTaskService(pending=[])
    task_service.answer_text_result = (True, "Claude 继续执行中", None)

    msg = _patched_message("my answer", user_id=1)
    await answer_external_user_question_text(msg, external_uq_state=state, task_service=task_service)  # type: ignore[arg-type]

    assert task_service.answer_text_calls == [{"user_id": 1, "text": "my answer"}]
    msg.answer.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_handler_rechecks_managed_pending_and_fails_closed() -> None:
    """A managed AskUserQuestion appearing in the filter→handler window must win.

    The filter would have routed the message (empty managed set at filter time),
    but the handler re-checks ``get_pending_user_questions`` and reports a managed
    card instead of consuming the text as a Ghostty Other answer.
    """
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending(user_id=1))
    task_service = _ScopedTaskService()
    # Simulate the managed question appearing after filtering.
    task_service._pending.append(object())

    msg = _patched_message("my answer", user_id=1)
    await answer_external_user_question_text(msg, external_uq_state=state, task_service=task_service)  # type: ignore[arg-type]

    assert task_service.answer_text_calls == []
    msg.answer.assert_awaited_once()  # type: ignore[attr-defined]
    sent = msg.answer.call_args.args[0]  # type: ignore[attr-defined]
    assert "选择题" in sent


@pytest.mark.asyncio
async def test_handler_rejects_when_pending_question_vanished() -> None:
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending(user_id=1))
    # Prune the only pending question before the handler resolves it.
    state.invalidate_tool("tuid-1")
    task_service = _FakeTaskService(pending=[])

    msg = _patched_message("my answer", user_id=1)
    await answer_external_user_question_text(msg, external_uq_state=state, task_service=task_service)  # type: ignore[arg-type]

    assert task_service.answer_text_calls == []
    sent = msg.answer.call_args.args[0]  # type: ignore[attr-defined]
    assert "过期" in sent


@pytest.mark.asyncio
async def test_handler_rejects_when_multiple_ghostty_questions() -> None:
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending(tool_use_id="tuid-a", user_id=1))
    state.store(_pending(tool_use_id="tuid-b", user_id=1))
    task_service = _FakeTaskService(pending=[])

    msg = _patched_message("my answer", user_id=1)
    await answer_external_user_question_text(msg, external_uq_state=state, task_service=task_service)  # type: ignore[arg-type]

    assert task_service.answer_text_calls == []
    sent = msg.answer.call_args.args[0]  # type: ignore[attr-defined]
    assert "多个" in sent


def _second_prompt() -> UserQuestionPrompt:
    return UserQuestionPrompt(
        tool_use_id="tuid-1",
        question_index=1,
        total_questions=2,
        question="Second pick?",
        options=(UserQuestionOption(label="X"), UserQuestionOption(label="Y")),
    )


@pytest.mark.asyncio
async def test_handler_renders_button_keyboard_for_next_prompt() -> None:
    """A successful free-text answer that advances to a next prompt must render
    that next prompt as an inline-button card (tokens + keyboard), not a bare
    "下一题: ... 可直接回复文字作答" plain-text line. Regression: the router
    previously emitted only text, so option/multi-select intermediate prompts
    were answerable only via the "Other" free-text fallback.
    """
    state = ExternalUserQuestionState(ttl_sec=60)
    state.store(_pending(user_id=1))
    next_prompt = _second_prompt()
    task_service = _FakeTaskService(pending=[])
    task_service.answer_text_result = (True, "已记录选择: my answer", next_prompt)

    msg = _patched_message("my answer", user_id=1)
    await answer_external_user_question_text(msg, external_uq_state=state, task_service=task_service)  # type: ignore[arg-type]

    # The ack sentence is sent verbatim...
    assert msg.answer.call_args_list[0].args[0] == "已记录选择: my answer"  # type: ignore[attr-defined]
    # ...then a second answer carries the next prompt WITH an inline keyboard.
    assert msg.answer.await_count == 2  # type: ignore[attr-defined]
    second_call = msg.answer.call_args_list[1]  # type: ignore[attr-defined]
    assert "Second pick" in second_call.args[0]
    reply_markup = second_call.kwargs.get("reply_markup")
    assert reply_markup is not None
    rows = reply_markup.inline_keyboard
    labels = {btn.text for row in rows for btn in row}
    assert {"X", "Y"} == labels
    # Next prompt acknowledged via the structured cursor.
    assert task_service.acknowledge_calls == [next_prompt.key]
