from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.domain.permission_models import PermissionPromptInput
from app.domain.user_question_models import UserQuestionPrompt
from app.infra.text_formatting import render_markdownish_to_telegram_html, render_markdownish_with_html_fallback, short_id
from app.infra.user_question_callbacks import build_user_question_callback_data
from app.services.message_sender import Button, Keyboard, MessageSender
from app.services.permission_callback_registry import SessionOrigin
from app.services.permission_gateway import RegisterForButtonConflict, RegisterForButtonOk
from app.services.user_question_callback_registry import (
    QuestionCallbackTokens,
    UserQuestionCallbackOrigin,
)

if TYPE_CHECKING:
    from app.domain.session_models import SessionPhase
    from app.services.external_binding_store import ExternalBindingStore
    from app.services.external_user_question_state import ExternalUserQuestionState
    from app.services.permission_gateway import PermissionGateway
    from app.services.user_question_callback_registry import UserQuestionCallbackRegistry

logger = logging.getLogger(__name__)


def format_external_tmux_question_text(prompt: UserQuestionPrompt, session_id: str) -> str:
    """Render the shared ❓/{sid}/问题/选项 block for a tmux external question card.

    Used both by the initial ``notify_user_question`` push and the mid-batch
    next-prompt push in the external_permission handler, so the two cards stay
    visually consistent instead of the follow-up drifting to a bare question
    string. Callers append their own CTA tail (button hint etc) when needed.
    """
    sid = short_id(session_id)
    lines: list[str] = [f"❓ [{sid}] 用户选择"]
    lines.append(f"问题: {prompt.question}")
    if prompt.options:
        lines.append("选项:")
        for i, option in enumerate(prompt.options, start=1):
            label = option.label
            if option.description:
                label += f" — {option.description}"
            lines.append(f"  {i}. {label}")
    return "\n".join(lines)


@dataclass
class _PendingReplyChunks:
    chunks: tuple[str, ...]
    parse_mode: str | None
    next_index: int = 0


class ExternalSessionPushNotifier:
    """Sends Telegram push notifications for bound external session events."""

    def __init__(
        self,
        *,
        message_sender: MessageSender,
        binding_store: ExternalBindingStore,
        retry_count: int = 1,
        permission_gateway: PermissionGateway | None = None,
        external_uq_state: ExternalUserQuestionState | None = None,
        user_question_callback_registry: UserQuestionCallbackRegistry | None = None,
    ) -> None:
        self._message_sender = message_sender
        self._binding_store = binding_store
        self._retry_count = retry_count
        self._permission_gateway = permission_gateway
        self._external_uq_state = external_uq_state
        self._user_question_callback_registry = user_question_callback_registry
        self._pending_reply_chunks: dict[tuple[int, str, str], _PendingReplyChunks] = {}

    async def notify_permission_request(
        self,
        *,
        user_id: int,
        session_id: str,
        tool_name: str,
        tool_input: dict | None,
        tool_use_id: str,
        cwd: str,
        title: str | None = None,
    ) -> bool:
        """Send permission request notification to bound user. Returns True if delivered."""
        gateway = self._permission_gateway
        if gateway is None:
            raise RuntimeError("permission gateway is not configured")

        result = await gateway.register_for_button(
            tool_use_id=tool_use_id,
            session_id=session_id,
            origin=SessionOrigin.EXTERNAL_BOUND,
            candidate_user_id=user_id,
        )
        if isinstance(result, RegisterForButtonConflict):
            logger.warning(
                "bound permission registration conflict",
                extra={"tool_use_id": tool_use_id, "session_id": session_id, "user_id": user_id},
            )
            return await self._send_with_retry(chat_id=user_id, text=result.advisory_text, keyboard=result.keyboard) is not None
        if not isinstance(result, RegisterForButtonOk):
            raise RuntimeError("unexpected permission gateway registration result")

        prompt = PermissionPromptInput(
            tool_name=tool_name or "unknown tool",
            tool_input=tool_input,
            cwd=cwd,
            session_id=session_id,
            session_title=title,
        )
        text = render_markdownish_to_telegram_html(gateway.message_builder.build_permission_prompt(prompt))
        message_id = await self._send_with_retry(chat_id=user_id, text=text, keyboard=result.keyboard, parse_mode="HTML")
        if message_id is not None:
            # Store the message ID and text in the token record for later editing
            await gateway.registry.update_telegram_message(
                token=result.token,
                chat_id=user_id,
                message_id=message_id,
                message_text=text,
            )
        return message_id is not None

    async def notify_permission_resolved_in_terminal(
        self,
        *,
        user_id: int,
        session_id: str,
        tool_name: str,
        tool_use_id: str,
        reason: str,
    ) -> bool:
        """Notify user that a permission was resolved in the terminal."""
        sid = short_id(session_id)
        reason_text = "已批准" if reason == "terminal_approved" else reason
        text = f"✅ [{sid}] 权限已在终端{reason_text}\n工具: {tool_name}"
        return await self._send_with_retry(chat_id=user_id, text=text) is not None

    async def notify_assistant_reply(
        self,
        *,
        user_id: int,
        session_id: str,
        text: str,
        title: str | None = None,
        turn_id: str | None = None,
    ) -> bool:
        """Send one completed assistant reply to the bound user."""
        reply = text.strip()
        if not reply:
            return False

        delivery_key = (user_id, session_id, turn_id) if turn_id is not None else None
        pending = self._pending_reply_chunks.get(delivery_key) if delivery_key is not None else None
        if pending is None:
            sid = short_id(session_id)
            heading = f"💬 [{sid}] Claude 回复"
            if title:
                heading = f"{heading}\n会话: {title.strip()}"
            message = f"{heading}\n\n{reply}"
            chunks, parse_mode = render_markdownish_with_html_fallback(message, 4096)
            if not chunks:
                return False
            pending = _PendingReplyChunks(chunks=tuple(chunks), parse_mode=parse_mode)
            if delivery_key is not None:
                self._pending_reply_chunks[delivery_key] = pending

        for index in range(pending.next_index, len(pending.chunks)):
            if (
                await self._send_with_retry(
                    chat_id=user_id,
                    text=pending.chunks[index],
                    parse_mode=pending.parse_mode,
                )
                is None
            ):
                return False
            pending.next_index = index + 1
        if delivery_key is not None:
            self._pending_reply_chunks.pop(delivery_key, None)
        return True

    def discard_assistant_reply_progress(self, session_id: str) -> None:
        stale_keys = [key for key in self._pending_reply_chunks if key[1] == session_id]
        for key in stale_keys:
            self._pending_reply_chunks.pop(key, None)

    async def notify_phase_change(
        self,
        *,
        user_id: int,
        session_id: str,
        old_phase: SessionPhase,
        new_phase: SessionPhase,
        cwd: str,
    ) -> bool:
        """Send phase change notification. Returns True if delivered."""
        sid = short_id(session_id)
        text = f"📊 [{sid}] {old_phase.value} → {new_phase.value}\n路径: {cwd}"
        return await self._send_with_retry(chat_id=user_id, text=text) is not None

    async def notify_session_end(
        self,
        *,
        user_id: int,
        session_id: str,
        cwd: str,
    ) -> bool:
        """Send session ended notification. Returns True if delivered."""
        sid = short_id(session_id)
        text = f"🔚 [{sid}] 会话已结束\n路径: {cwd}"
        return await self._send_with_retry(chat_id=user_id, text=text) is not None

    async def notify_user_question(
        self,
        *,
        user_id: int,
        session_id: str,
        prompts: tuple[UserQuestionPrompt, ...],
        interactive: bool = False,
        origin: UserQuestionCallbackOrigin = UserQuestionCallbackOrigin.EXTERNAL_TERMINAL,
    ) -> bool:
        """Send notification showing AskUserQuestion options.

        When *interactive* is True, the first prompt's options are shown as
        clickable buttons backed by opaque registry tokens (``ask:`` for
        Ghostty, ``ext_uq:`` for the legacy tmux path). Multiple-select prompts
        get toggles plus a submit button. Otherwise this is informational only
        (the user answers in the terminal). Returns True if delivered.
        """
        if not prompts:
            return False
        # For interactive mode, we only show the first unanswered prompt with buttons
        prompt = prompts[0]
        prefix = "ask" if origin is UserQuestionCallbackOrigin.EXTERNAL_TERMINAL else "ext_uq"
        lines = format_external_tmux_question_text(prompt, session_id).splitlines()

        if not interactive or not prompt.options:
            lines.append("请在终端中选择")
            lines.append("")
            text = "\n".join(lines).rstrip()
            return await self._send_with_retry(chat_id=user_id, text=text) is not None

        tokens = await self._register_question_buttons(
            user_id=user_id,
            session_id=session_id,
            prompt=prompt,
            origin=origin,
        )
        if tokens is None:
            # No registry wired in: show informational-only card (no buttons).
            lines.append("请在终端中选择")
            lines.append("")
            text = "\n".join(lines).rstrip()
            return await self._send_with_retry(chat_id=user_id, text=text) is not None

        lines.append("")
        lines.append("👇 点击按钮选择；可直接回复文字作为 Other/自由文本")
        if origin is UserQuestionCallbackOrigin.EXTERNAL_TERMINAL:
            lines.append("作答期间请勿在本地终端操作")
        text = "\n".join(lines).rstrip()
        buttons: list[list[Button]] = []
        # The legacy tmux PTY injector (``inject_option_selection``) only models a
        # single-choice "move cursor down N then Enter" sequence — it has no
        # multi-select toggle/submit semantics. Render multi-select prompts as
        # one-shot single-choice buttons for tmux so every emitted button carries
        # a resolveable SELECT token; Ghostty keeps its native toggle + submit UX.
        supports_multi_select = origin is UserQuestionCallbackOrigin.EXTERNAL_TERMINAL
        if prompt.multi_select and supports_multi_select:
            for index, option in enumerate(prompt.options):
                token = tokens.toggle_tokens[index] if index < len(tokens.toggle_tokens) else None
                buttons.append([Button(text=f"{index + 1}. {option.label}"[:40], callback_data=self._button_data(prefix, token))])
            submit_data = self._button_data(prefix, tokens.submit_token)
            buttons.append([Button(text="提交选择", callback_data=submit_data)])
        else:
            for index, option in enumerate(prompt.options):
                token = tokens.select_tokens[index] if index < len(tokens.select_tokens) else None
                buttons.append([Button(text=f"{index + 1}. {option.label}"[:40], callback_data=self._button_data(prefix, token))])
        keyboard = Keyboard(rows=buttons)
        return await self._send_with_retry(chat_id=user_id, text=text, keyboard=keyboard) is not None

    async def _register_question_buttons(
        self,
        *,
        user_id: int,
        session_id: str,
        prompt: UserQuestionPrompt,
        origin: UserQuestionCallbackOrigin,
    ) -> QuestionCallbackTokens | None:
        registry = self._user_question_callback_registry
        if registry is None or not prompt.options:
            return None
        # The legacy tmux injector only models single-choice keystrokes; render
        # even a multi_select prompt as single-choice (registering select_tokens,
        # not toggle/submit). Ghostty keeps the native multi-select registration.
        supports_multi_select = origin is UserQuestionCallbackOrigin.EXTERNAL_TERMINAL
        return await registry.register_question_tokens(
            owner_user_id=user_id,
            session_id=session_id,
            tool_use_id=prompt.tool_use_id,
            question_index=prompt.question_index,
            option_count=len(prompt.options),
            multi_select=prompt.multi_select and supports_multi_select,
            origin=origin,
        )

    @staticmethod
    def _button_data(prefix: str, token: str | None) -> str:
        if token is None:
            # Degenerate: registration returned fewer tokens than options.
            # Forbidden by contract, but guard so we never emit empty callback_data.
            raise RuntimeError("user-question callback token missing")
        return build_user_question_callback_data(prefix=prefix, token=token)

    async def notify_info(
        self,
        *,
        user_id: int,
        text: str,
    ) -> bool:
        """Send an informational notification (no action buttons). Returns True if delivered."""
        return await self._send_with_retry(chat_id=user_id, text=text) is not None

    async def _send_with_retry(
        self, *, chat_id: int, text: str, keyboard: Keyboard | None = None, parse_mode: str | None = None
    ) -> int | None:
        """Send message with retry on failure. Returns message_id on success."""
        for attempt in range(1 + self._retry_count):
            try:
                message_id = await self._message_sender.send_message(chat_id=chat_id, text=text, keyboard=keyboard, parse_mode=parse_mode)
                return message_id
            except Exception:
                if attempt < self._retry_count:
                    logger.warning(
                        "Push notification delivery failed (attempt %d), retrying...",
                        attempt + 1,
                    )
                else:
                    logger.error(
                        "Push notification delivery failed after %d attempts, giving up. chat_id=%d",
                        attempt + 1,
                        chat_id,
                    )
        return None
