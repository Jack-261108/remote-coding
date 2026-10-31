"""Tests for ExternalSessionPushNotifier.

Covers: notify_assistant_reply, notify_phase_change, notify_session_end,
notify_user_question, notify_info, _send_with_retry, error branches.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.session_models import SessionPhase
from app.domain.user_question_models import UserQuestionOption, UserQuestionPrompt
from app.services.external_session_push_notifier import (
    ExternalSessionPushNotifier,
    format_external_tmux_question_text,
)
from app.services.user_question_callback_registry import (
    UserQuestionCallbackOrigin,
    UserQuestionCallbackRegistry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_notifier(
    *,
    retry_count: int = 1,
    send_side_effects: list | None = None,
    registry: UserQuestionCallbackRegistry | None = None,
) -> tuple[ExternalSessionPushNotifier, MagicMock]:
    sender = MagicMock()
    if send_side_effects:
        sender.send_message = AsyncMock(side_effect=send_side_effects)
    else:
        sender.send_message = AsyncMock(return_value=123)
    notifier = ExternalSessionPushNotifier(
        message_sender=sender,
        binding_store=MagicMock(),
        retry_count=retry_count,
        user_question_callback_registry=registry,
    )
    return notifier, sender


# ---------------------------------------------------------------------------
# notify_assistant_reply
# ---------------------------------------------------------------------------


class TestNotifyAssistantReply:
    @pytest.mark.asyncio
    async def test_sends_formatted_reply(self):
        notifier, sender = _make_notifier()

        result = await notifier.notify_assistant_reply(
            user_id=42,
            session_id="sess-123456",
            text="**完成**\n\n`pytest -q` 已通过。",
            title="修复测试",
        )

        assert result is True
        sender.send_message.assert_awaited_once()
        kwargs = sender.send_message.call_args.kwargs
        assert kwargs["chat_id"] == 42
        assert kwargs["parse_mode"] == "HTML"
        assert "[sess-123]" in kwargs["text"]
        assert "修复测试" in kwargs["text"]
        assert "<b>完成</b>" in kwargs["text"]
        assert "<code>pytest -q</code>" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_splits_long_reply_at_telegram_limit(self):
        notifier, sender = _make_notifier()

        result = await notifier.notify_assistant_reply(
            user_id=42,
            session_id="sess-1",
            text="line\n" * 2000,
        )

        assert result is True
        assert sender.send_message.await_count > 1
        for call in sender.send_message.await_args_list:
            assert len(call.kwargs["text"]) <= 4096
            assert call.kwargs["parse_mode"] == "HTML"

    @pytest.mark.asyncio
    async def test_long_html_tag_falls_back_to_plain_text_chunks(self):
        notifier, sender = _make_notifier()
        long_url = f"https://example.com/{'a' * 5000}"

        result = await notifier.notify_assistant_reply(
            user_id=42,
            session_id="sess-1",
            text=f"[link]({long_url})",
        )

        assert result is True
        assert sender.send_message.await_count > 1
        for call in sender.send_message.await_args_list:
            assert len(call.kwargs["text"]) <= 4096
            assert call.kwargs["parse_mode"] is None

    @pytest.mark.asyncio
    async def test_retry_resumes_from_first_unsent_chunk(self):
        notifier, sender = _make_notifier(retry_count=0)
        send_count = 0

        async def send_message(**kwargs) -> int:
            nonlocal send_count
            send_count += 1
            if send_count == 2:
                raise RuntimeError("second chunk failed")
            return send_count

        sender.send_message.side_effect = send_message

        first_result = await notifier.notify_assistant_reply(
            user_id=42,
            session_id="sess-1",
            turn_id="turn-1",
            text="x" * 5000,
        )
        second_result = await notifier.notify_assistant_reply(
            user_id=42,
            session_id="sess-1",
            turn_id="turn-1",
            text="x" * 5000,
        )

        assert first_result is False
        assert second_result is True
        assert sender.send_message.await_count >= 3
        calls = sender.send_message.await_args_list
        assert calls[0].kwargs["text"] != calls[1].kwargs["text"]
        assert calls[1].kwargs["text"] == calls[2].kwargs["text"]
        assert all(call.kwargs["text"] != calls[0].kwargs["text"] for call in calls[2:])

    @pytest.mark.asyncio
    async def test_returns_false_for_empty_reply(self):
        notifier, sender = _make_notifier()

        result = await notifier.notify_assistant_reply(
            user_id=42,
            session_id="sess-1",
            text="   ",
        )

        assert result is False
        sender.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_false_when_delivery_fails(self):
        notifier, sender = _make_notifier(
            retry_count=1,
            send_side_effects=[RuntimeError("fail"), RuntimeError("fail")],
        )

        result = await notifier.notify_assistant_reply(
            user_id=42,
            session_id="sess-1",
            text="reply",
        )

        assert result is False
        assert sender.send_message.await_count == 2


# ---------------------------------------------------------------------------
# notify_phase_change
# ---------------------------------------------------------------------------


class TestNotifyPhaseChange:
    @pytest.mark.asyncio
    async def test_sends_phase_change_message(self):
        notifier, sender = _make_notifier()
        result = await notifier.notify_phase_change(
            user_id=42,
            session_id="sess-1",
            old_phase=SessionPhase.WAITING_FOR_INPUT,
            new_phase=SessionPhase.PROCESSING,
            cwd="/tmp/project",
        )
        assert result is True
        sender.send_message.assert_awaited_once()
        text = sender.send_message.call_args.kwargs["text"]
        assert "waiting_for_input" in text
        assert "processing" in text


# ---------------------------------------------------------------------------
# notify_session_end
# ---------------------------------------------------------------------------


class TestNotifySessionEnd:
    @pytest.mark.asyncio
    async def test_sends_session_end_message(self):
        notifier, sender = _make_notifier()
        result = await notifier.notify_session_end(
            user_id=42,
            session_id="sess-1",
            cwd="/tmp/project",
        )
        assert result is True
        text = sender.send_message.call_args.kwargs["text"]
        assert "会话已结束" in text


# ---------------------------------------------------------------------------
# notify_user_question
# ---------------------------------------------------------------------------


class TestNotifyUserQuestion:
    @pytest.mark.asyncio
    async def test_returns_false_for_empty_prompts(self):
        notifier, sender = _make_notifier()
        result = await notifier.notify_user_question(
            user_id=42,
            session_id="sess-1",
            prompts=(),
        )
        assert result is False
        sender.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sends_info_only_when_not_interactive(self):
        notifier, sender = _make_notifier()
        prompts = (
            UserQuestionPrompt(
                tool_use_id="tuid-1",
                question_index=0,
                total_questions=1,
                question="Continue?",
            ),
        )
        result = await notifier.notify_user_question(
            user_id=42,
            session_id="sess-1",
            prompts=prompts,
            interactive=False,
        )
        assert result is True
        text = sender.send_message.call_args.kwargs["text"]
        assert "请在终端中选择" in text

    @pytest.mark.asyncio
    async def test_sends_interactive_with_options(self):
        registry = UserQuestionCallbackRegistry(ttl_sec=300)
        notifier, sender = _make_notifier(registry=registry)
        prompts = (
            UserQuestionPrompt(
                tool_use_id="tuid-1",
                question_index=0,
                total_questions=1,
                question="Choose?",
                options=(
                    UserQuestionOption(label="Yes", description="confirm"),
                    UserQuestionOption(label="No"),
                ),
            ),
        )
        result = await notifier.notify_user_question(
            user_id=42,
            session_id="sess-1",
            prompts=prompts,
            interactive=True,
        )
        assert result is True
        text = sender.send_message.call_args.kwargs["text"]
        assert "可直接回复文字作为 Other/自由文本" in text
        assert "请勿在本地终端操作" in text
        keyboard = sender.send_message.call_args.kwargs["keyboard"]
        assert keyboard is not None
        rows = keyboard.rows
        assert len(rows) == 2
        for row in rows:
            cb_data = row[0].callback_data
            assert cb_data.startswith("ask:")
            assert len(cb_data.encode()) <= 64
            assert cb_data.count(":") == 1  # colon-free token, two segments
            token = cb_data.split(":", 1)[1]
            resolved = await registry.resolve(token, user_id=42)
            assert resolved.__class__.__name__ == "UserQuestionCallbackResolved"

    @pytest.mark.asyncio
    async def test_sends_interactive_multi_select_with_toggle_and_submit(self):
        registry = UserQuestionCallbackRegistry(ttl_sec=300)
        notifier, sender = _make_notifier(registry=registry)
        prompts = (
            UserQuestionPrompt(
                tool_use_id="tuid-m",
                question_index=0,
                total_questions=1,
                question="Pick:",
                options=(
                    UserQuestionOption(label="A"),
                    UserQuestionOption(label="B"),
                ),
                multi_select=True,
            ),
        )
        result = await notifier.notify_user_question(
            user_id=42,
            session_id="sess-1",
            prompts=prompts,
            interactive=True,
        )
        assert result is True
        keyboard = sender.send_message.call_args.kwargs["keyboard"]
        assert keyboard is not None
        rows = keyboard.rows
        assert len(rows) == 3  # two toggles + submit
        assert rows[-1][0].text == "提交选择"
        for row in rows:
            cb_data = row[0].callback_data
            assert cb_data.startswith("ask:")
            assert len(cb_data.encode()) <= 64
            assert cb_data.count(":") == 1

    @pytest.mark.asyncio
    async def test_tmux_origin_renders_multi_select_as_single_choice(self):
        """The legacy tmux injector has no multi-select toggle/submit, so a
        multi_select prompt under EXTERNAL_TMUX must render one single-choice
        button per option (``ext_uq:`` tokens, no submit row) — otherwise the
        ext_uq handler receives toggle/submit actions it cannot satisfy."""
        registry = UserQuestionCallbackRegistry(ttl_sec=300)
        notifier, sender = _make_notifier(registry=registry)
        prompts = (
            UserQuestionPrompt(
                tool_use_id="tuid-tm",
                question_index=0,
                total_questions=1,
                question="Pick:",
                options=(
                    UserQuestionOption(label="A"),
                    UserQuestionOption(label="B"),
                ),
                multi_select=True,
            ),
        )
        result = await notifier.notify_user_question(
            user_id=42,
            session_id="sess-1",
            prompts=prompts,
            interactive=True,
            origin=UserQuestionCallbackOrigin.EXTERNAL_TMUX,
        )
        assert result is True
        keyboard = sender.send_message.call_args.kwargs["keyboard"]
        assert keyboard is not None
        rows = keyboard.rows
        # Two single-choice option rows only — NO submit button (tmux one-shot).
        assert len(rows) == 2
        for row in rows:
            cb_data = row[0].callback_data
            assert cb_data.startswith("ext_uq:")
            assert len(cb_data.encode()) <= 64
            assert cb_data.count(":") == 1
        assert rows[-1][0].text != "提交选择"

    @pytest.mark.asyncio
    async def test_sends_info_when_registry_missing(self):
        notifier, sender = _make_notifier()
        prompts = (
            UserQuestionPrompt(
                tool_use_id="tuid-1",
                question_index=0,
                total_questions=1,
                question="Choose?",
                options=(UserQuestionOption(label="Yes"),),
            ),
        )
        result = await notifier.notify_user_question(
            user_id=42,
            session_id="sess-1",
            prompts=prompts,
            interactive=True,
        )
        assert result is True
        # No registry ⇒ informational-only, no buttons.
        assert "请在终端中选择" in sender.send_message.call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_sends_info_when_interactive_but_no_options(self):
        notifier, sender = _make_notifier()
        prompts = (
            UserQuestionPrompt(
                tool_use_id="tuid-1",
                question_index=0,
                total_questions=1,
                question="Type something:",
            ),
        )
        result = await notifier.notify_user_question(
            user_id=42,
            session_id="sess-1",
            prompts=prompts,
            interactive=True,
        )
        assert result is True
        text = sender.send_message.call_args.kwargs["text"]
        assert "请在终端中选择" in text


# ---------------------------------------------------------------------------
# notify_info
# ---------------------------------------------------------------------------


class TestNotifyInfo:
    @pytest.mark.asyncio
    async def test_sends_info_message(self):
        notifier, sender = _make_notifier()
        result = await notifier.notify_info(user_id=42, text="Hello!")
        assert result is True
        sender.send_message.assert_awaited_once_with(chat_id=42, text="Hello!", keyboard=None, parse_mode=None)


# ---------------------------------------------------------------------------
# _send_with_retry
# ---------------------------------------------------------------------------


class TestSendWithRetry:
    @pytest.mark.asyncio
    async def test_returns_message_id_on_success(self):
        notifier, sender = _make_notifier()
        result = await notifier._send_with_retry(chat_id=1, text="hi")
        assert result == 123

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        notifier, sender = _make_notifier(
            retry_count=2,
            send_side_effects=[RuntimeError("fail"), RuntimeError("fail"), 456],
        )
        result = await notifier._send_with_retry(chat_id=1, text="hi")
        assert result == 456
        assert sender.send_message.await_count == 3

    @pytest.mark.asyncio
    async def test_returns_none_after_all_retries_fail(self):
        notifier, sender = _make_notifier(
            retry_count=1,
            send_side_effects=[RuntimeError("fail"), RuntimeError("fail")],
        )
        result = await notifier._send_with_retry(chat_id=1, text="hi")
        assert result is None
        assert sender.send_message.await_count == 2


# ---------------------------------------------------------------------------
# notify_permission_resolved_in_terminal
# ---------------------------------------------------------------------------


class TestNotifyPermissionResolved:
    @pytest.mark.asyncio
    async def test_sends_approved_notification(self):
        notifier, sender = _make_notifier()
        result = await notifier.notify_permission_resolved_in_terminal(
            user_id=42,
            session_id="sess-1",
            tool_name="Bash",
            tool_use_id="tuid-1",
            reason="terminal_approved",
        )
        assert result is True
        text = sender.send_message.call_args.kwargs["text"]
        assert "已批准" in text
        assert "Bash" in text

    @pytest.mark.asyncio
    async def test_sends_other_reason_notification(self):
        notifier, sender = _make_notifier()
        result = await notifier.notify_permission_resolved_in_terminal(
            user_id=42,
            session_id="sess-1",
            tool_name="Write",
            tool_use_id="tuid-2",
            reason="denied",
        )
        assert result is True
        text = sender.send_message.call_args.kwargs["text"]
        assert "denied" in text


def test_format_external_tmux_question_text_matches_initial_card_layout() -> None:
    """The mid-batch next-prompt push must reuse the initial card's ❓/sid/选项 layout.

    Regression guard for #13: the external_permission handler used to send the
    next prompt as a bare question string. Both initial card and follow-up must
    show the ❓ user-select header, the question, and the numbered options list.
    """
    prompt = UserQuestionPrompt(
        tool_use_id="tool-1",
        question_index=1,
        total_questions=2,
        question="pick a color",
        options=(
            UserQuestionOption(label="red", description="warm"),
            UserQuestionOption(label="blue", description="cool"),
        ),
        multi_select=False,
    )
    text = format_external_tmux_question_text(prompt, session_id="session-uuid-1")

    lines = text.splitlines()
    # short_id("session-uuid-1") == "session-" (first 8 chars).
    assert lines[0] == "❓ [session-] 用户选择"
    assert "问题: pick a color" in lines
    assert "选项:" in lines
    assert "  1. red — warm" in lines
    assert "  2. blue — cool" in lines


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
