from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import BaseFilter
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.bot.handlers.user_utils import extract_user_id
from app.infra.text_formatting import format_external_session_action_outcome, short_cwd, short_id, truncate_text
from app.services.external_session_binder import ExternalSessionBinder
from app.services.external_session_discovery import ExternalSessionDiscoveryService
from app.services.external_session_input_service import ExternalSessionInputService, PairOutcome, SendOutcome
from app.services.session_action_validator import validate_external_session_select
from app.services.session_id_resolver import (
    BindResult,
    UnbindResult,
    external_session_select_token,
    resolve_and_bind,
    resolve_and_unbind,
    resolve_unique_prefix,
)
from app.services.session_registry import SessionRegistryService

if TYPE_CHECKING:
    from app.services.session_service import SessionService

logger = logging.getLogger(__name__)

# 进入/退出外部输入模式的消息标记：leave 依赖它在原文上做替换，
# 与 handle_session_select 的 ACTIVATED 分支必须保持同一常量。
# external_session 命令入口同样复用 LEFT_MARK，保证多入口文案一致。
EXTERNAL_INPUT_ACTIVE_MARK = "✅ 已进入外部输入模式"
EXTERNAL_INPUT_LEFT_MARK = "✅ 已退出外部输入模式"


def _pair_outcome_text(outcome: PairOutcome) -> str:
    return {
        PairOutcome.ACTIVATED: "已进入外部输入模式，普通文本将注入已配对的终端。",
        PairOutcome.NEEDS_PAIRING: "需要配对终端。",
        PairOutcome.NOT_OWNER: "你不是该会话的绑定者。",
        PairOutcome.BINDING_STALE: "绑定已变更，请重新操作。",
        PairOutcome.SESSION_ENDED: "会话已结束。",
        PairOutcome.PROCESS_INVALID: "无法验证前台 Claude 进程。",
        PairOutcome.TERMINAL_INVALID: "配对的终端已失效。",
        PairOutcome.ADAPTER_UNAVAILABLE: "终端适配器不可用（请检查终端远程控制配置）。",
        PairOutcome.NO_TERMINALS: "未发现可用的终端。",
        PairOutcome.TOKEN_INVALID: "配对令牌无效或已过期。",
        PairOutcome.TOKEN_UNAUTHORIZED: "配对令牌不属于你。",
        PairOutcome.PAIRING_NOT_ENABLED: "外部输入功能未启用。",
        PairOutcome.PAIRED: "配对完成。",
    }.get(outcome, f"操作失败：{outcome.value}")


def _pair_terminal_label(
    *,
    terminal_id: str,
    name: str | None,
    cwd: str | None,
    pid_match: bool = False,
) -> str:
    """Build a compact but distinguishable explicit-pairing label.

    ``pid_match`` marks a candidate whose window hosts the bound Claude PID.
    It is a hint that helps the owner pick correctly; it never selects.
    """
    name_label = truncate_text((name or "未命名").strip() or "未命名", 22)
    cwd_label = truncate_text(short_cwd(cwd or "", fallback="cwd 未知"), 22)
    suffix = " · PID✓" if pid_match else ""
    return f"配对: {name_label} · {cwd_label} · {terminal_id[-8:]}{suffix}"


async def _resolve_terminal_id_prefix(
    terminal_id_prefix: str,
    registry_service: SessionRegistryService,
) -> tuple[str | None, str | None]:
    candidates = [session.terminal_id for session in await registry_service.list_active_sessions() if session.is_alive]
    return resolve_unique_prefix(terminal_id_prefix, candidates)


def _clear_keyboard() -> InlineKeyboardMarkup:
    """Telegram 对省略 reply_markup 的语义是「保持不变」，须传空键盘才能移除旧按钮。"""
    return InlineKeyboardMarkup(inline_keyboard=[])


def _unbind_button(callback_token: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text="取消绑定", callback_data=f"sess:unbind:{callback_token}")


async def _edit_or_answer(
    callback: CallbackQuery,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    """编辑触发 callback 的消息；确定性不可编辑时另发一条，暂时性错误向上冒泡。

    把「绑定 → 配对 → 进入输入模式」的连续步骤合并到同一条消息上，而不是每步
    都另发新消息。消息超过 48h 后 Telegram 会返回不可编辑的 InaccessibleMessage，
    此时另发新消息；edit 因内容未变被拒（message is not modified）时视为无事
    发生，避免回退成重复消息。未指定新键盘时必须显式传空键盘，否则旧 inline
    按钮会原样残留。

    仅 TelegramBadRequest 中的确定性错误（消息不可编辑/已删除/文本超长等）回落
    answer；TelegramRetryAfter/TelegramNetworkError/TelegramServerError 等暂时性
    错误不回落——旧消息其实仍可编辑，回落会制造重复内容，交由 router 错误层处理。
    """
    markup = reply_markup if reply_markup is not None else _clear_keyboard()
    message = callback.message
    if isinstance(message, Message):
        try:
            await message.edit_text(text, reply_markup=markup)
            return
        except TelegramBadRequest as exc:
            if "message is not modified" in str(exc).lower():
                return
            # 确定性 BAD REQUEST（消息不可编辑/已删除/文本超长等）：旧消息已不可用，回落为新消息。
            logger.warning("编辑消息失败，回退为发送新消息: %s", exc)
        # TelegramRetryAfter / TelegramNetworkError / TelegramServerError 等暂时性错误不回落
        # answer：旧消息其实仍可编辑，回落会制造重复内容；交由 router 错误层处理。
    if message is not None:
        await message.answer(text, reply_markup=reply_markup)


def register_session_action_handlers(
    router: Router,
    *,
    discovery: ExternalSessionDiscoveryService,
    binder: ExternalSessionBinder,
    registry_service: SessionRegistryService | None = None,
    external_session_input_service: ExternalSessionInputService | None = None,
) -> None:
    @router.callback_query(F.data.startswith("sess:select:") | F.data.startswith("sess:open:"))
    async def handle_session_select(callback: CallbackQuery, callback_parts: tuple[str, ...]) -> None:
        user_id = extract_user_id(callback)
        opens_from_list = callback_parts[1] == "open"
        session_id_prefix = callback_parts[2]

        validation = validate_external_session_select(
            session_id_prefix,
            user_id=user_id,
            discovery=discovery,
            binder=binder,
        )
        if validation.denial_message or not validation.session_id or not validation.action:
            await callback.answer(validation.denial_message or "Session not found")
            return

        session_id = validation.session_id
        callback_token = validation.callback_token
        detail_text = f"📂 Session: {short_id(session_id, 12)}...\n  cwd: {validation.cwd}"

        async def _show_selection_result(
            text: str,
            reply_markup: InlineKeyboardMarkup | None = None,
        ) -> None:
            if opens_from_list:
                message = callback.message
                if message is not None:
                    await message.answer(text, reply_markup=reply_markup)
                return
            await _edit_or_answer(callback, text, reply_markup=reply_markup)

        async def _show_pair_failure(outcome: PairOutcome) -> None:
            await callback.answer()
            keyboard = InlineKeyboardMarkup(inline_keyboard=[[_unbind_button(callback_token)]])
            await _show_selection_result(detail_text + f"\n\n⚠️ {_pair_outcome_text(outcome)}", reply_markup=keyboard)

        if validation.action == "unbind" and external_session_input_service is not None:
            # Bound + owner: offer external input activation/pairing in addition to unbind.
            outcome = await external_session_input_service.activate_select(user_id=user_id, session_id=session_id)
            if outcome == PairOutcome.ACTIVATED:
                await callback.answer()
                keyboard = InlineKeyboardMarkup(
                    inline_keyboard=[
                        [InlineKeyboardButton(text="退出输入模式", callback_data=f"sess:leave:{callback_token}")],
                        [_unbind_button(callback_token)],
                    ]
                )
                await _show_selection_result(detail_text + "\n\n" + EXTERNAL_INPUT_ACTIVE_MARK, reply_markup=keyboard)
                return
            if outcome == PairOutcome.NEEDS_PAIRING:
                pair_outcome, candidates = await external_session_input_service.pair_candidates(user_id=user_id, session_id=session_id)
                if pair_outcome == PairOutcome.NEEDS_PAIRING and candidates is not None:
                    buttons: list[list[InlineKeyboardButton]] = []
                    for terminal in candidates.terminals:
                        token = await external_session_input_service.register_pair_token(
                            user_id=user_id,
                            session_id=session_id,
                            expected_binding_id=candidates.binding_id,
                            terminal_kind=terminal.kind,
                            terminal_id=terminal.terminal_id,
                        )
                        if token is None:
                            continue
                        label = _pair_terminal_label(
                            terminal_id=terminal.terminal_id,
                            name=terminal.name,
                            cwd=terminal.cwd,
                            pid_match=candidates.binding_pid is not None
                            and candidates.binding_pid > 0
                            and (candidates.binding_pid == terminal.pid or candidates.binding_pid in terminal.foreground_pids),
                        )
                        buttons.append([InlineKeyboardButton(text=label, callback_data=f"tpair:{token}")])
                    buttons.append([_unbind_button(callback_token)])
                    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
                    await callback.answer()
                    await _show_selection_result(detail_text + "\n\n🔌 选择要配对的终端：", reply_markup=keyboard)
                    return
                await _show_pair_failure(pair_outcome)
                return
            await _show_pair_failure(outcome)
            return

        if validation.action == "unbind":
            buttons = [[_unbind_button(callback_token)]]
        else:
            buttons = [[InlineKeyboardButton(text="绑定", callback_data=f"sess:bind:{callback_token}")]]

        keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)

        await callback.answer()
        await _show_selection_result(detail_text, reply_markup=keyboard)

    async def _handle_bind_unbind_action(callback: CallbackQuery, callback_parts: tuple[str, ...], action_type: str) -> None:
        user_id = extract_user_id(callback)
        session_id_prefix = callback_parts[2]

        result: BindResult | UnbindResult
        if action_type == "bind":
            result = await resolve_and_bind(session_id_prefix, user_id=user_id, discovery=discovery, binder=binder)
        else:
            result = await resolve_and_unbind(session_id_prefix, user_id=user_id, discovery=discovery, binder=binder)

        if result.success:
            success_text = "绑定成功" if action_type == "bind" else "取消绑定成功"
            await callback.answer(success_text)
            if action_type == "bind":
                assert result.session_id is not None  # success ⇒ resolved session_id
                token = external_session_select_token(result.session_id, discovery=discovery, binder=binder)
                keyboard = InlineKeyboardMarkup(
                    inline_keyboard=[[InlineKeyboardButton(text="进入终端输入", callback_data=f"sess:select:{token}")]]
                )
                await _edit_or_answer(
                    callback,
                    format_external_session_action_outcome(action_type, True, session_id=result.session_id, message=result.message),
                    reply_markup=keyboard,
                )
            else:
                await _edit_or_answer(
                    callback,
                    format_external_session_action_outcome(action_type, True, session_id=result.session_id, message=result.message),
                )
        else:
            await callback.answer(f"❌ {result.message}")

    @router.callback_query(F.data.startswith("sess:bind:"))
    async def handle_session_bind(callback: CallbackQuery, callback_parts: tuple[str, ...]) -> None:
        await _handle_bind_unbind_action(callback, callback_parts, "bind")

    @router.callback_query(F.data.startswith("sess:unbind:"))
    async def handle_session_unbind(callback: CallbackQuery, callback_parts: tuple[str, ...]) -> None:
        await _handle_bind_unbind_action(callback, callback_parts, "unbind")

    # ── tmux session actions ─────────────────────────────────────────────────

    @router.callback_query(F.data.startswith("sess:attach:"))
    async def handle_session_attach(callback: CallbackQuery, callback_parts: tuple[str, ...]) -> None:
        if registry_service is None:
            await callback.answer("功能不可用")
            return
        user_id = extract_user_id(callback)
        terminal_id_prefix = callback_parts[2]
        terminal_id, error = await _resolve_terminal_id_prefix(terminal_id_prefix, registry_service)
        if error or not terminal_id:
            await callback.answer(error or "Session not found")
            return
        ok, text = await registry_service.attach_user(user_id=user_id, terminal_id=terminal_id)
        await callback.answer(text if ok else f"❌ {text}")
        if callback.message:
            await callback.message.answer(text)

    @router.callback_query(F.data.startswith("sess:close:"))
    async def handle_session_close(callback: CallbackQuery, callback_parts: tuple[str, ...]) -> None:
        if registry_service is None:
            await callback.answer("功能不可用")
            return
        terminal_id_prefix = callback_parts[2]
        terminal_id, error = await _resolve_terminal_id_prefix(terminal_id_prefix, registry_service)
        if error or not terminal_id:
            await callback.answer(error or "Session not found")
            return
        ok = await registry_service.close_session(terminal_id)
        await callback.answer("会话已关闭" if ok else "关闭失败")
        if callback.message:
            await callback.message.answer(f"{'✅' if ok else '❌'} 会话 `{terminal_id}` {'已关闭' if ok else '关闭失败'}")

    @router.callback_query(F.data.startswith("sess:leave:"))
    async def handle_session_leave(
        callback: CallbackQuery,
        callback_parts: tuple[str, ...] | None = None,
    ) -> None:
        if external_session_input_service is None:
            await callback.answer("功能不可用")
            return
        user_id = extract_user_id(callback)
        left = await external_session_input_service.leave(user_id=user_id)
        await callback.answer("已退出外部输入模式" if left else "当前不在外部输入模式")
        message = callback.message
        if not left or not isinstance(message, Message):
            return

        parts = callback_parts or tuple((callback.data or "").split(":"))
        callback_token = parts[2] if len(parts) > 2 else None
        current_text = message.text
        if isinstance(current_text, str) and EXTERNAL_INPUT_ACTIVE_MARK in current_text:
            updated_text = current_text.replace(EXTERNAL_INPUT_ACTIVE_MARK, EXTERNAL_INPUT_LEFT_MARK)
        else:
            updated_text = EXTERNAL_INPUT_LEFT_MARK

        keyboard = None
        if callback_token:
            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="重新进入输入模式", callback_data=f"sess:select:{callback_token}")],
                    [_unbind_button(callback_token)],
                ]
            )
        await _edit_or_answer(callback, updated_text, reply_markup=keyboard)


def register_pair_consume_handler(
    router: Router,
    *,
    input_service: ExternalSessionInputService,
    session_service: SessionService | None = None,
) -> None:
    """Consume a ``tpair:<token>`` callback and finalise terminal pairing."""

    @router.callback_query(F.data.startswith("tpair:"))
    async def handle_pair_consume(callback: CallbackQuery, callback_parts: tuple[str, ...]) -> None:
        if len(callback_parts) < 2 or not callback_parts[1]:
            await callback.answer("无效的配对回调")
            return
        token = callback_parts[1]
        user_id = extract_user_id(callback)
        outcome = await input_service.consume_pair_token(token=token, user_id=user_id)
        await callback.answer()
        if outcome == PairOutcome.PAIRED:
            text = "✅ 配对成功，已进入外部输入模式。"
            # The external text router yields to an active managed chat
            # (see ExternalInputTargetActiveFilter), so tell the user up
            # front instead of letting texts silently go to the chat.
            session = await session_service.get(user_id) if session_service is not None else None
            if session is not None and session.claude_chat_active:
                text += "\n⚠️ 当前还有活跃的 managed 会话：普通文本将发给该会话，未注册的斜杠命令仍会注入外部终端。"
                if session.terminal_mode:
                    # /exit closes the managed persistent terminal (and its
                    # Claude session) in terminal_mode — irreversible, so
                    # warn instead of recommending it as a light switch.
                    text += "注意：/exit 会关闭 managed 终端及其中运行的 Claude 会话（不可逆），请确认后再使用。"
                else:
                    text += "如需让普通文本也走外部终端，请先 /exit 退出聊天模式。"
            await _edit_or_answer(callback, text)
        elif callback.message:
            await callback.message.answer(f"❌ {_pair_outcome_text(outcome)}")


def register_external_text_handlers(
    router: Router,
    *,
    input_service: ExternalSessionInputService,
    target_filter: BaseFilter,
) -> None:
    """Inject plain Telegram text into the active external terminal session."""

    @router.message(F.text, target_filter)
    async def handle_external_text(message: Message) -> None:
        text = (message.text or "").strip()
        if not text:
            return
        user_id = message.from_user.id if message.from_user else 0
        if not user_id:
            return
        outcome = await input_service.send_text(user_id=user_id, text=text)
        if outcome == SendOutcome.SENT:
            return
        if outcome == SendOutcome.QUEUED:
            await message.answer("📥 已排队，Claude 就绪后将自动发送。")
            return
        if outcome == SendOutcome.NO_TARGET:
            return
        text_map = {
            SendOutcome.NOT_PAIRED: "❌ 当前会话未配对终端。",
            SendOutcome.NOT_OWNER: "❌ 你不是该会话的绑定者。",
            SendOutcome.BINDING_STALE: "❌ 绑定已变更，请重新选择。",
            SendOutcome.SESSION_ENDED: "❌ 会话已结束。",
            SendOutcome.PROCESS_INVALID: "❌ 无法验证前台 Claude 进程。",
            SendOutcome.TERMINAL_INVALID: "❌ 配对的终端已失效。",
            SendOutcome.ADAPTER_UNAVAILABLE: "❌ 终端适配器不可用。",
            SendOutcome.QUEUE_FULL: "❌ 输入队列已满。",
            SendOutcome.INJECTION_FAILED: "❌ 注入失败。",
            SendOutcome.INJECTION_INDETERMINATE: "⚠️ 注入结果不确定。",
        }
        await message.answer(text_map.get(outcome, f"❌ 发送失败：{outcome.value}"))
