from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiogram import F, Router
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from app.bot.handlers.callback_utils import apply_callback_response, safe_edit_keyboard
from app.bot.handlers.command_user_question import _is_accessible_message
from app.bot.handlers.user_utils import extract_user_id
from app.domain.user_question_models import ExternalTmuxQuestionTarget, UserQuestionPrompt
from app.infra.lock_registry import RefCountedLockRegistry
from app.infra.user_question_callbacks import (
    build_user_question_callback_data as build_opaque_callback_data,
)
from app.infra.user_question_callbacks import (
    parse_user_question_callback_token,
)
from app.services.external_session_push_notifier import format_external_tmux_question_text
from app.services.user_question_callback_registry import (
    UserQuestionCallbackOrigin,
    UserQuestionCallbackRegistry,
    UserQuestionCallbackResolved,
    UserQuestionCallbackUnauthorized,
)

if TYPE_CHECKING:
    from app.adapters.claude.hook_socket_server import HookSocketServer
    from app.services.external_user_question_state import ExternalUserQuestionState
    from app.services.permission_gateway import PermissionGateway
    from app.services.unbound_permission_handler import UnboundPermissionHandler

logger = logging.getLogger(__name__)

_tmux_uq_locks = RefCountedLockRegistry(ttl_sec=300, cleanup_interval_sec=60, cleanup_batch_size=50)


async def _build_next_prompt_keyboard(
    *,
    user_id: int,
    pending_session_id: str,
    prompt: UserQuestionPrompt,
    registry: UserQuestionCallbackRegistry,
) -> InlineKeyboardMarkup | None:
    """Register EXTERNAL_TMUX select tokens for *prompt* and build an inline keyboard.

    The multi-prompt tmux path renders one question at a time; after the user
    answers an intermediate prompt we push a fresh card whose buttons resolve
    to the same ``ext_uq:`` handler. Multi-select prompts are rendered as
    single-choice here (one button per option), matching the initial card — the
    legacy tmux injector has no multi-select toggle/submit.
    """
    if not prompt.options:
        return None
    tokens = await registry.register_question_tokens(
        owner_user_id=user_id,
        session_id=pending_session_id,
        tool_use_id=prompt.tool_use_id,
        question_index=prompt.question_index,
        option_count=len(prompt.options),
        multi_select=False,
        origin=UserQuestionCallbackOrigin.EXTERNAL_TMUX,
    )
    rows: list[list[InlineKeyboardButton]] = []
    for index, option in enumerate(prompt.options):
        token = tokens.select_tokens[index] if index < len(tokens.select_tokens) else None
        if token is None:
            # Should not happen for a tokenised registration; guard defensively.
            return None
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{index + 1}. {option.label}"[:40],
                    callback_data=build_opaque_callback_data(prefix="ext_uq", token=token),
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def register_external_permission_handler(
    permission_router: Router,
    uq_router: Router | None = None,
    *,
    hook_socket_server: HookSocketServer,
    unbound_permission_handler: UnboundPermissionHandler,
    external_uq_state: ExternalUserQuestionState | None = None,
    permission_gateway: PermissionGateway,
    user_question_callback_registry: UserQuestionCallbackRegistry | None = None,
) -> None:
    external_uq_target = uq_router if uq_router is not None else permission_router

    @permission_router.callback_query(F.data.startswith("ext_perm:"))
    async def handle_external_permission_callback(callback: CallbackQuery, callback_parts: tuple[str, ...]) -> None:
        _, token, decision = callback_parts
        if decision not in ("allow", "deny", "auto_approve"):
            await callback.answer("Invalid decision", show_alert=True)
            return

        user_id = extract_user_id(callback)
        response = await permission_gateway.handle_callback(data=f"perm:{token}:{decision}", user_id=user_id)
        await apply_callback_response(
            callback,
            edit_text=response.edit_message_text,
            clear_keyboard=response.clear_keyboard,
            alert_text=response.alert_text,
            show_alert=response.show_alert,
            log_prefix="external permission",
        )

    @external_uq_target.callback_query(F.data.startswith("ext_uq:"))
    async def handle_external_user_question_callback(callback: CallbackQuery, callback_parts: tuple[str, ...]) -> None:
        """Handle an AskUserQuestion option button for external (tmux) sessions.

        The button carries an opaque registry token (``ext_uq:{token}``); only
        tokens registered with the ``EXTERNAL_TMUX`` origin route to the legacy
        tmux PTY injection path. Ghostty questions use the ``ask:`` prefix and
        are handled by the managed user-question router.
        """
        if user_question_callback_registry is None or external_uq_state is None:
            await callback.answer("Feature not available", show_alert=True)
            return

        token = parse_user_question_callback_token(callback_parts, prefix="ext_uq")
        if token is None:
            await callback.answer("无效的选择操作", show_alert=True)
            return

        user_id = extract_user_id(callback)
        resolved = await user_question_callback_registry.resolve(token, user_id=user_id)
        if isinstance(resolved, UserQuestionCallbackUnauthorized):
            await callback.answer("Question does not belong to you", show_alert=True)
            return
        if not isinstance(resolved, UserQuestionCallbackResolved):
            await callback.answer("Question expired or already answered", show_alert=True)
            return

        snapshot = resolved.snapshot
        if snapshot.origin is not UserQuestionCallbackOrigin.EXTERNAL_TMUX:
            # Ghostty/managed questions never travel via ext_uq:; fail closed.
            await callback.answer("该问题不支持在此处作答", show_alert=True)
            return

        from app.adapters.process.pty_injector import inject_option_selection

        pending = external_uq_state.get(snapshot.tool_use_id)
        if pending is None or pending.user_id != user_id:
            await callback.answer("Question expired or already answered", show_alert=True)
            return
        target = pending.target
        if not isinstance(target, ExternalTmuxQuestionTarget):
            await callback.answer("Cannot inject: question is not a tmux target", show_alert=True)
            return

        next_prompt: UserQuestionPrompt | None = None
        pending_session_id = pending.session_id
        async with _tmux_uq_locks.lock(target.pane_id):
            fresh_resolved = await user_question_callback_registry.resolve(token, user_id=user_id)
            if not isinstance(fresh_resolved, UserQuestionCallbackResolved):
                await callback.answer("Question expired or already answered", show_alert=True)
                return

            pending = external_uq_state.get(snapshot.tool_use_id)
            if pending is None or pending.user_id != user_id:
                await callback.answer("Question expired or already answered", show_alert=True)
                return

            prompt = next((item for item in pending.prompts if item.question_index == snapshot.question_index), None)
            if prompt is None:
                await callback.answer("该问题已变化，请等待最新卡片", show_alert=True)
                return
            option_index = snapshot.option_index if snapshot.option_index is not None else -1
            if option_index < 0 or option_index >= len(prompt.options):
                await callback.answer("Invalid option", show_alert=True)
                return

            selected_label = prompt.options[option_index].label
            is_final = snapshot.question_index == pending.prompts[-1].question_index
            pending_session_id = pending.session_id

            ok, err = await inject_option_selection(
                target.pane_id,
                option_index=option_index,
                submit_after=is_final,
                tmux_bin=target.tmux_bin,
            )
            if not ok:
                logger.warning(
                    "pty injection failed for external user question",
                    extra={"tool_use_id": snapshot.tool_use_id, "pane_id": target.pane_id, "error": err},
                )
                await callback.answer(f"Injection failed: {err}", show_alert=True)
                return

            # Drop tokens before releasing the pane lock. Telegram I/O can raise
            # (expired callback query); a waiter must not inject a second time.
            if is_final:
                external_uq_state.remove(snapshot.tool_use_id)
                await user_question_callback_registry.invalidate_tool(
                    session_id=pending_session_id,
                    tool_use_id=snapshot.tool_use_id,
                )
            else:
                await user_question_callback_registry.invalidate_prompt(
                    session_id=pending_session_id,
                    tool_use_id=snapshot.tool_use_id,
                    question_index=snapshot.question_index,
                )
                next_prompt = next(
                    (item for item in pending.prompts if item.question_index > snapshot.question_index),
                    None,
                )

        try:
            await callback.answer(f"✅ Selected: {selected_label}")
        except Exception:
            logger.warning("failed to answer external user question callback", exc_info=True)

        callback_message = callback.message if _is_accessible_message(callback.message) else None
        if callback_message is not None:
            await safe_edit_keyboard(callback_message, None, "clear ext_uq inline keyboard")
            original_text = callback_message.text or ""
            try:
                await callback_message.edit_text(f"{original_text}\n\n✅ Selected: {selected_label}")
            except Exception:
                logger.debug("failed to annotate ext_uq message", exc_info=True)

        if is_final:
            await hook_socket_server.respond_to_permission(
                tool_use_id=snapshot.tool_use_id,
                decision="allow",
                reason=f"AskUserQuestion answered via Telegram by user {user_id}",
            )
            logger.info(
                "external user question answered via Telegram",
                extra={
                    "tool_use_id": snapshot.tool_use_id,
                    "question_index": snapshot.question_index,
                    "option_index": option_index,
                    "selected_label": selected_label,
                    "user_id": user_id,
                    "pane_id": target.pane_id,
                },
            )
            return

        logger.info(
            "external user question advanced to next prompt",
            extra={
                "tool_use_id": snapshot.tool_use_id,
                "question_index": snapshot.question_index,
                "option_index": option_index,
                "user_id": user_id,
                "pane_id": target.pane_id,
            },
        )
        if next_prompt is not None and callback_message is not None:
            keyboard = await _build_next_prompt_keyboard(
                user_id=user_id,
                pending_session_id=pending_session_id,
                prompt=next_prompt,
                registry=user_question_callback_registry,
            )
            body = format_external_tmux_question_text(next_prompt, pending_session_id)
            text = f"{body}\n\n👇 点击按钮选择；可直接回复文字作为 Other/自由文本"
            await callback_message.answer(text, reply_markup=keyboard)
