"""Handler-layer smoke tests for the external Ghostty input feature.

Covers the Telegram handlers registered in ``app/bot/handlers/session_actions.py``
and ``app/bot/handlers/external_session.py`` and the router wiring in
``app/bot/router.py``: ``tpair:`` pairing consume, ``sess:leave`` exit, the
external text router (active-target filter match / fall-through and the
SendOutcome -> reply mapping), and ``/external leave``.

These build a real ``ExternalSessionInputService`` over fake adapter/probe (no
Ghostty/TCC/PTYs required) and drive aiogram routers directly, so the behaviour
verified is the same code path the dispatcher hits in production.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram import Router
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message, User

from app.adapters.storage.file_session_store import FileSessionStore
from app.bot.handlers.external_session import register_external_session_handler
from app.bot.handlers.session_actions import (
    _pair_terminal_label,
    register_external_text_handlers,
    register_pair_consume_handler,
    register_session_action_handlers,
)
from app.bot.router import ExternalInputTargetActiveFilter
from app.domain.external_session_models import ExternalBinding
from app.domain.session_models import SessionPhase
from app.services.external_binding_store import ExternalBindingStore
from app.services.external_session_binder import ExternalSessionBinder
from app.services.external_session_discovery import ExternalSessionDiscoveryService
from app.services.external_session_input_service import (
    ExternalSessionInputService,
    PairOutcome,
)
from app.services.session_store import SessionStore
from tests.fakes.external_input_harness import build_input_harness
from tests.fakes.ghostty import FakeGhosttyTerminalAdapter

if TYPE_CHECKING:
    from app.services.session_service import SessionService


_make_service_serial = itertools.count(1)


def _make_service(
    tmp_path: Path,
    *,
    session_id: str = "session-1",
    user_id: int = 42,
    paired: bool = True,
    enabled: bool = True,
    adapter: FakeGhosttyTerminalAdapter | None = None,
) -> tuple[ExternalSessionInputService, ExternalBindingStore, SessionStore, ExternalBinding]:
    harness = build_input_harness(
        tmp_path,
        name=f"h-{next(_make_service_serial)}",
        session_id=session_id,
        user_id=user_id,
        paired=paired,
        enabled=enabled,
        adapter=adapter,
    )
    return harness.service, harness.binding_store, harness.session_store, harness.binding


def _callback(data: str, *, user_id: int = 42, message: Message | None = None) -> CallbackQuery:
    cb = MagicMock(spec=CallbackQuery)
    cb.data = data
    cb.from_user = MagicMock(spec=User)
    cb.from_user.id = user_id
    cb.answer = AsyncMock()
    cb.message = message
    return cb


class _StubSessionService:
    """Minimal SessionService stand-in for the router filters/handlers."""

    def __init__(self, *, chat_active: bool = False, terminal_mode: bool = False) -> None:
        self._session = SimpleNamespace(claude_chat_active=chat_active, terminal_mode=terminal_mode)

    async def get(self, user_id: int):
        return self._session


def _session_service_stub(*, chat_active: bool = False, terminal_mode: bool = False) -> SessionService:
    return cast("SessionService", _StubSessionService(chat_active=chat_active, terminal_mode=terminal_mode))


def _message(text: str, *, user_id: int = 42) -> Message:
    msg = MagicMock(spec=Message)
    msg.text = text
    msg.from_user = MagicMock(spec=User)
    msg.from_user.id = user_id
    msg.answer = AsyncMock()
    msg.edit_text = AsyncMock()
    return msg


async def _dispatch_cb(router: Router, index: int, callback: CallbackQuery) -> None:
    handler = router.callback_query.handlers[index]
    parts = tuple(callback.data.split(":"))
    await handler.callback(callback, callback_parts=parts)


async def _dispatch_msg(router: Router, index: int, message: Message) -> object:
    handler = router.message.handlers[index]
    return await handler.callback(message)


def test_pair_terminal_label_disambiguates_name_cwd_and_uuid() -> None:
    assert (
        _pair_terminal_label(
            terminal_id="53362216-AE12-4EC7-8D3D-F6BF6270B6FE",
            name="✳ Claude Code",
            cwd="/Users/jack/project/bitchat",
        )
        == "配对: ✳ Claude Code · project/bitchat · 6270B6FE"
    )


# ── tpair consume ──────────────────────────────────────────────────────────


class TestPairConsumeHandler:
    @pytest.mark.asyncio
    async def test_paired_replies_success(self, tmp_path: Path) -> None:
        service, binding_store, _, binding = _make_service(tmp_path)
        # Register a token bound to this terminal through the real registry.
        token = await service.register_pair_token(
            user_id=42,
            session_id=binding.session_id,
            expected_binding_id=binding.binding_id,
            terminal_kind="ghostty",
            terminal_id="term-1",
        )
        assert token is not None

        router = Router()
        register_pair_consume_handler(router, input_service=service)
        msg = _message("")
        cb = _callback(f"tpair:{token}", message=msg)
        await _dispatch_cb(router, 0, cb)
        # 配对成功后必须清掉消息上残留的配对/取消绑定按钮（空键盘而非 None）
        msg.edit_text.assert_awaited_once_with("✅ 配对成功，已进入外部输入模式。", reply_markup=InlineKeyboardMarkup(inline_keyboard=[]))

    @pytest.mark.asyncio
    async def test_paired_with_active_managed_chat_appends_hint(self, tmp_path: Path) -> None:
        """Pairing succeeds while a managed chat is active (no terminal): the
        reply must warn that plain text keeps going to the managed session and
        recommend the light /exit."""
        service, binding_store, _, binding = _make_service(tmp_path)
        token = await service.register_pair_token(
            user_id=42,
            session_id=binding.session_id,
            expected_binding_id=binding.binding_id,
            terminal_kind="ghostty",
            terminal_id="term-1",
        )
        assert token is not None

        router = Router()
        register_pair_consume_handler(router, input_service=service, session_service=_session_service_stub(chat_active=True))
        msg = _message("")
        cb = _callback(f"tpair:{token}", message=msg)
        await _dispatch_cb(router, 0, cb)
        assert msg.edit_text.await_count == 1
        reply = msg.edit_text.await_args.args[0]
        assert "配对成功" in reply
        assert "managed 会话" in reply
        assert "斜杠命令仍会注入外部终端" in reply
        assert "请先 /exit 退出聊天模式" in reply
        assert "不可逆" not in reply

    @pytest.mark.asyncio
    async def test_paired_with_active_managed_terminal_warns_about_exit(self, tmp_path: Path) -> None:
        """In terminal_mode /exit closes the managed terminal and its Claude
        session (irreversible): the hint must warn, not recommend it as a
        light route switch."""
        service, binding_store, _, binding = _make_service(tmp_path)
        token = await service.register_pair_token(
            user_id=42,
            session_id=binding.session_id,
            expected_binding_id=binding.binding_id,
            terminal_kind="ghostty",
            terminal_id="term-1",
        )
        assert token is not None

        router = Router()
        register_pair_consume_handler(
            router, input_service=service, session_service=_session_service_stub(chat_active=True, terminal_mode=True)
        )
        msg = _message("")
        cb = _callback(f"tpair:{token}", message=msg)
        await _dispatch_cb(router, 0, cb)
        assert msg.edit_text.await_count == 1
        reply = msg.edit_text.await_args.args[0]
        assert "配对成功" in reply
        assert "斜杠命令仍会注入外部终端" in reply
        assert "/exit 会关闭 managed 终端" in reply
        assert "不可逆" in reply

    @pytest.mark.asyncio
    async def test_invalid_token_replies_error_without_editing_candidates(self, tmp_path: Path) -> None:
        """配对失败时不能编辑候选消息：其余候选终端与「取消绑定」按钮需保留，
        用户才能换终端重试。错误另发新消息。"""
        service, *_ = _make_service(tmp_path)
        router = Router()
        register_pair_consume_handler(router, input_service=service)
        msg = _message("")
        cb = _callback("tpair:does-not-exist", message=msg)
        await _dispatch_cb(router, 0, cb)
        msg.edit_text.assert_not_awaited()
        msg.answer.assert_awaited_once()
        assert msg.answer.await_args.args[0].startswith("❌")


# ── sess:leave ─────────────────────────────────────────────────────────────


class TestSessionLeaveHandler:
    @pytest.mark.asyncio
    async def test_leave_when_input_service_none(self, tmp_path: Path) -> None:
        discovery = ExternalSessionDiscoveryService()
        binding_store = ExternalBindingStore(tmp_path / "b")
        binder = ExternalSessionBinder(
            discovery=discovery,
            binding_store=binding_store,
            projects_dir=tmp_path / "projects",
        )
        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder)
        # Handler order in register_session_action_handlers:
        # 0=select, then bind/unbind helpers attach → so we locate leave robustly.
        awaited = False
        for handler in router.callback_query.handlers:
            if _is_leave_handler(handler):
                cb = _callback("sess:leave:abc")
                await handler.callback(cb)
                cb.answer.assert_awaited_once_with("功能不可用")
                awaited = True
                break
        assert awaited, "sess:leave handler not registered"

    @pytest.mark.asyncio
    async def test_leave_exits_input_mode(self, tmp_path: Path) -> None:
        service, *_ = _make_service(tmp_path)
        # Put the user into input mode via activate_select.
        outcome = await service.activate_select(user_id=42, session_id="session-1")
        assert outcome == PairOutcome.ACTIVATED
        assert await service.has_target(42) is True

        discovery = ExternalSessionDiscoveryService()
        binding_store = ExternalBindingStore(tmp_path / "b")
        binder = ExternalSessionBinder(
            discovery=discovery,
            binding_store=binding_store,
            projects_dir=tmp_path / "projects",
        )
        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder, external_session_input_service=service)
        message = _message("")
        message.text = "📂 Session: session-1...\n  cwd: /project\n\n✅ 已进入外部输入模式"
        message.edit_text = AsyncMock()
        for handler in router.callback_query.handlers:
            if _is_leave_handler(handler):
                cb = _callback("sess:leave:abc", message=message)
                await handler.callback(cb)
                cb.answer.assert_awaited_once_with("已退出外部输入模式")
                message.edit_text.assert_awaited_once()
                assert message.edit_text.await_args is not None
                updated_text = message.edit_text.await_args.args[0]
                reply_markup = message.edit_text.await_args.kwargs["reply_markup"]
                assert "✅ 已退出外部输入模式" in updated_text
                assert "✅ 已进入外部输入模式" not in updated_text
                assert [row[0].text for row in reply_markup.inline_keyboard] == ["重新进入输入模式", "取消绑定"]
                break
        assert await service.has_target(42) is False


def _is_leave_handler(handler) -> bool:
    """Return True if this registered callback handler is the ``sess:leave`` one.

    Detected by dispatching a ``sess:leave:`` callback and observing the handler
    runs without raising (bind/unbind/attach/close propagate AttributeError/KeyError
    on a leave-shaped payload because their callback_parts[2] semantics differ);
    instead we inspect handler source name via the callback closure.
    """
    cb_obj = handler.callback
    name = getattr(cb_obj, "__name__", "")
    return name == "handle_session_leave"


# ── external text router ────────────────────────────────────────────────────


class TestExternalTextRouter:
    @pytest.mark.asyncio
    async def test_filter_no_target_falls_through(self, tmp_path: Path) -> None:
        service, *_ = _make_service(tmp_path)
        target_filter = ExternalInputTargetActiveFilter(service, _session_service_stub())
        assert await target_filter(_message("hi")) is False

    @pytest.mark.asyncio
    async def test_filter_with_target_matches(self, tmp_path: Path) -> None:
        service, *_ = _make_service(tmp_path)
        await service.activate_select(user_id=42, session_id="session-1")
        target_filter = ExternalInputTargetActiveFilter(service, _session_service_stub())
        assert await target_filter(_message("hi")) is True

    @pytest.mark.asyncio
    async def test_filter_yields_to_active_managed_chat(self, tmp_path: Path) -> None:
        """Coexistence: an active managed ``claude_chat_active`` session owns plain
        text; the external router must fall through instead of hijacking it."""
        service, *_ = _make_service(tmp_path)
        await service.activate_select(user_id=42, session_id="session-1")
        target_filter = ExternalInputTargetActiveFilter(service, _session_service_stub(chat_active=True))
        assert await target_filter(_message("hi")) is False

    @pytest.mark.asyncio
    async def test_filter_coexisting_chat_still_takes_slash_commands(self, tmp_path: Path) -> None:
        """``chat_text_router`` excludes slash text, so a slash that also yielded
        on coexistence would be silently dropped — it must still match."""
        service, *_ = _make_service(tmp_path)
        await service.activate_select(user_id=42, session_id="session-1")
        target_filter = ExternalInputTargetActiveFilter(service, _session_service_stub(chat_active=True))
        assert await target_filter(_message("/compact")) is True

    @pytest.mark.asyncio
    async def test_sent_is_silent(self, tmp_path: Path) -> None:
        service, *_ = _make_service(tmp_path)
        await service.activate_select(user_id=42, session_id="session-1")
        router = Router()
        register_external_text_handlers(
            router, input_service=service, target_filter=ExternalInputTargetActiveFilter(service, _session_service_stub())
        )
        msg = _message("hello world")
        await _dispatch_msg(router, 0, msg)
        msg.answer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_coexisting_chat_slash_command_routes_to_terminal(self, tmp_path: Path) -> None:
        """An unregistered slash command during coexistence must reach the
        external terminal instead of vanishing between the two routers
        (``chat_text_router`` excludes slash text)."""
        adapter = FakeGhosttyTerminalAdapter()
        service, *_ = _make_service(tmp_path, adapter=adapter)
        await service.activate_select(user_id=42, session_id="session-1")
        router = Router()
        register_external_text_handlers(
            router, input_service=service, target_filter=ExternalInputTargetActiveFilter(service, _session_service_stub(chat_active=True))
        )
        await _dispatch_msg(router, 0, _message("/compact"))
        assert [text for _term, text in adapter.inject_calls] == ["/compact"]

    @pytest.mark.asyncio
    async def test_queued_replies_notice(self, tmp_path: Path) -> None:
        adapter = FakeGhosttyTerminalAdapter()
        # Force a non-sendable phase so send_text enqueues.
        service, binding_store, session_store, binding = _make_service(tmp_path, adapter=adapter)
        state = session_store.get("session-1")
        state.phase = SessionPhase.PROCESSING
        session_store.save(state)
        await service.activate_select(user_id=42, session_id="session-1")
        router = Router()
        register_external_text_handlers(
            router, input_service=service, target_filter=ExternalInputTargetActiveFilter(service, _session_service_stub())
        )
        msg = _message("queue me")
        await _dispatch_msg(router, 0, msg)
        msg.answer.assert_awaited_once()
        assert "已排队" in msg.answer.await_args.args[0]
        await service.shutdown()


# ── /external leave ────────────────────────────────────────────────────────


class TestExternalLeaveCommand:
    @pytest.mark.asyncio
    async def test_external_leave_exits_input_mode(self, tmp_path: Path) -> None:
        service, *_ = _make_service(tmp_path)
        await service.activate_select(user_id=42, session_id="session-1")

        discovery = ExternalSessionDiscoveryService()
        binding_store = ExternalBindingStore(tmp_path / "b")
        binder = ExternalSessionBinder(
            discovery=discovery,
            binding_store=binding_store,
            projects_dir=tmp_path / "projects",
        )
        session_store = SessionStore(FileSessionStore(str(tmp_path / "s")))
        router = Router()
        register_external_session_handler(router, discovery=discovery, binder=binder, session_store=session_store, input_service=service)
        msg = _message("/external leave")
        # The Command filter consumes only the text body; emulate dispatcher.
        handler = router.message.handlers[0]
        await handler.callback(msg)
        msg.answer.assert_awaited_once_with("✅ 已退出外部输入模式。")
        assert await service.has_target(42) is False

    @pytest.mark.asyncio
    async def test_external_leave_disabled_when_no_service(self, tmp_path: Path) -> None:
        discovery = ExternalSessionDiscoveryService()
        binding_store = ExternalBindingStore(tmp_path / "b")
        binder = ExternalSessionBinder(
            discovery=discovery,
            binding_store=binding_store,
            projects_dir=tmp_path / "projects",
        )
        session_store = SessionStore(FileSessionStore(str(tmp_path / "s")))
        router = Router()
        register_external_session_handler(
            router,
            discovery=discovery,
            binder=binder,
            session_store=session_store,  # input_service defaults None
        )
        msg = _message("/external leave")
        handler = router.message.handlers[0]
        await handler.callback(msg)
        msg.answer.assert_awaited_once_with("外部输入功能未启用。")

    @pytest.mark.asyncio
    async def test_external_leave_not_in_mode(self, tmp_path: Path) -> None:
        service, *_ = _make_service(tmp_path)
        discovery = ExternalSessionDiscoveryService()
        binding_store = ExternalBindingStore(tmp_path / "b")
        binder = ExternalSessionBinder(
            discovery=discovery,
            binding_store=binding_store,
            projects_dir=tmp_path / "projects",
        )
        session_store = SessionStore(FileSessionStore(str(tmp_path / "s")))
        router = Router()
        register_external_session_handler(router, discovery=discovery, binder=binder, session_store=session_store, input_service=service)
        msg = _message("/external leave")
        handler = router.message.handlers[0]
        await handler.callback(msg)
        msg.answer.assert_awaited_once_with("当前不在外部输入模式。")
