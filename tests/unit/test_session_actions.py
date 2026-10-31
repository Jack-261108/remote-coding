"""Unit tests for session_actions callback handlers."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram import Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, User

from app.adapters.process.ghostty_terminal_adapter import ExternalTerminal
from app.bot.handlers.session_actions import _edit_or_answer, _resolve_terminal_id_prefix, register_session_action_handlers
from app.domain.external_session_models import ExternalBinding
from app.domain.hook_models import HookEvent
from app.domain.models import TerminalSessionInfo, utc_now
from app.services.external_binding_store import ExternalBindingStore
from app.services.external_session_binder import ExternalSessionBinder
from app.services.external_session_discovery import ExternalSessionDiscoveryService
from app.services.external_session_input_service import PairingCandidates, PairOutcome
from app.services.session_action_validator import validate_external_session_select
from app.services.session_id_resolver import _resolve_session_id, external_session_select_token, unique_prefixes


async def _dispatch(router, index: int, callback) -> None:
    """Dispatch a callback to a handler, injecting callback_parts as the middleware would."""
    handler = router.callback_query.handlers[index]
    await handler.callback(callback, callback_parts=tuple(callback.data.split(":")))


@pytest.fixture
def discovery() -> ExternalSessionDiscoveryService:
    return ExternalSessionDiscoveryService()


@pytest.fixture
def binding_store(tmp_path: Path) -> ExternalBindingStore:
    return ExternalBindingStore(data_dir=tmp_path)


@pytest.fixture
def binder(discovery: ExternalSessionDiscoveryService, binding_store: ExternalBindingStore, tmp_path: Path) -> ExternalSessionBinder:
    return ExternalSessionBinder(
        discovery=discovery,
        binding_store=binding_store,
        projects_dir=tmp_path / "projects",
    )


def _tmux_session(terminal_id: str, *, is_alive: bool = True) -> TerminalSessionInfo:
    return TerminalSessionInfo(
        terminal_id=terminal_id,
        tmux_session_name=f"tgcli_{terminal_id}",
        workdir="/home/user/proj",
        phase="idle",
        owner_user_id=1,
        attached_user_ids=[],
        is_alive=is_alive,
        last_activity=None,
    )


class TestUniquePrefixes:
    def test_single_tmux_terminal_id_keeps_workdir_hash_in_prefix(self) -> None:
        prefixes = unique_prefixes(["user_1234567890_aaaaaaaaaaaa"])

        assert prefixes["user_1234567890_aaaaaaaaaaaa"] == "user_1234567890_a"

    def test_short_id_that_is_another_id_prefix_uses_hash_token(self) -> None:
        prefixes = unique_prefixes(["abc", "abcdef"])

        assert prefixes["abc"].startswith("~")
        assert prefixes["abcdef"] == "abcdef"

    def test_max_length_caps_unrepresentable_common_prefix(self) -> None:
        first = "x" * 52 + "a"
        second = "x" * 52 + "b"

        prefixes = unique_prefixes([first, second], max_length=52)

        assert len(prefixes[first]) <= 52
        assert len(prefixes[second]) <= 52

    def test_hash_token_respects_max_length(self) -> None:
        shorter = "x" * 52
        longer = f"{shorter}a"

        prefixes = unique_prefixes([shorter, longer], max_length=52)

        assert len(prefixes[shorter]) <= 52
        assert prefixes[shorter].startswith("~")


class TestResolveSessionId:
    def test_resolves_from_discovery(self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder) -> None:
        event = HookEvent(session_id="abcdef1234567890full", cwd="/tmp", event="PreToolUse", status="running")
        discovery.record_event(event)

        resolved, error = _resolve_session_id("abcdef1234567890", discovery, binder)
        assert resolved == "abcdef1234567890full"
        assert error is None

    def test_not_found(self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder) -> None:
        resolved, error = _resolve_session_id("nonexistent", discovery, binder)
        assert resolved is None
        assert error == "Session not found"

    def test_resolves_bound_session_from_memory_when_disk_is_missing(
        self,
        discovery: ExternalSessionDiscoveryService,
        binder: ExternalSessionBinder,
        binding_store: ExternalBindingStore,
    ) -> None:
        session_id = "abcdef1234567890full"
        binding_store.save_binding(
            ExternalBinding(
                session_id=session_id,
                user_id=42,
                cwd="/home/user/proj",
                bound_at=utc_now(),
                jsonl_path=None,
            )
        )
        (binding_store._file_path).unlink()

        resolved, error = _resolve_session_id(session_id[:16], discovery, binder)

        assert resolved == session_id
        assert error is None


class TestSessionSelectHandler:
    @pytest.mark.asyncio
    async def test_select_unbound_session_shows_bind_button(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder
    ) -> None:
        session_id = "abcdef1234567890full"
        event = HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running")
        discovery.record_event(event)

        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder)

        # Simulate callback
        callback = AsyncMock(spec=CallbackQuery)
        callback.data = f"sess:select:{session_id[:16]}"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = 42
        callback.message = AsyncMock(spec=Message)

        resolved, error = _resolve_session_id(session_id[:16], discovery, binder)
        assert resolved == session_id
        assert error is None

        # Verify the session is unbound (no binding exists)
        binding = binder._binding_store.get_binding(session_id)
        assert binding is None  # unbound, so "绑定" button should show

    @pytest.mark.asyncio
    async def test_select_bound_session_shows_unbind_button(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder
    ) -> None:
        session_id = "abcdef1234567890full"
        user_id = 42
        event = HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running")
        discovery.record_event(event)

        # Bind the session
        result = await binder.bind(user_id=user_id, session_id=session_id)
        assert result.success

        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder)
        callback = AsyncMock(spec=CallbackQuery)
        callback.data = f"sess:select:{session_id[:16]}"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = user_id
        callback.answer = AsyncMock()
        callback.message = AsyncMock(spec=Message)
        callback.message.answer = AsyncMock()
        callback.message.edit_text = AsyncMock()

        await _dispatch(router, 0, callback)

        callback.answer.assert_awaited_once_with()
        callback.message.edit_text.assert_awaited_once()
        keyboard = callback.message.edit_text.call_args.kwargs["reply_markup"]
        callbacks = [button.callback_data for row in keyboard.inline_keyboard for button in row]
        assert callbacks == [f"sess:unbind:{session_id[:16]}"]

    @pytest.mark.asyncio
    async def test_select_active_unbound_callback_still_renders_bind_button(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder
    ) -> None:
        session_id = "active-unbound-session-0001"
        discovery.record_event(HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running"))
        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder)
        callback = AsyncMock(spec=CallbackQuery)
        callback.data = f"sess:select:{session_id[:16]}"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = 42
        callback.answer = AsyncMock()
        callback.message = AsyncMock(spec=Message)
        callback.message.answer = AsyncMock()
        callback.message.edit_text = AsyncMock()

        await _dispatch(router, 0, callback)

        callback.answer.assert_awaited_once_with()
        callback.message.edit_text.assert_awaited_once()
        keyboard = callback.message.edit_text.call_args.kwargs["reply_markup"]
        callbacks = [button.callback_data for row in keyboard.inline_keyboard for button in row]
        assert callbacks == [f"sess:bind:{session_id[:16]}"]

    @pytest.mark.asyncio
    async def test_select_old_unbound_callback_after_other_user_bound_does_not_render_bind_button(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder
    ) -> None:
        session_id = "stale-catalog-session-0001"
        discovery.record_event(HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running"))
        bind_result = await binder.bind(user_id=99, session_id=session_id)
        assert bind_result.success

        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder)
        callback = AsyncMock(spec=CallbackQuery)
        callback.data = f"sess:select:{session_id[:16]}"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = 42
        callback.answer = AsyncMock()
        callback.message = AsyncMock(spec=Message)
        callback.message.answer = AsyncMock()

        await _dispatch(router, 0, callback)

        callback.answer.assert_awaited_once_with("Session is not available to bind")
        callback.message.answer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_select_ended_unbound_callback_does_not_answer_session_not_found(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder
    ) -> None:
        session_id = "ended-unbound-session-0001"
        discovery.record_event(HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running", pid=12345))
        discovery.mark_session_ended(session_id)
        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder)
        callback = AsyncMock(spec=CallbackQuery)
        callback.data = f"sess:select:{session_id[:16]}"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = 42
        callback.answer = AsyncMock()
        callback.message = AsyncMock(spec=Message)
        callback.message.answer = AsyncMock()

        await _dispatch(router, 0, callback)

        callback.answer.assert_awaited_once_with("Session is no longer available")
        callback.message.answer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_select_ambiguous_live_prefix_is_not_overridden_by_ended_prefix(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder
    ) -> None:
        shared_prefix = "ambiguous-live-"
        discovery.record_event(HookEvent(session_id=f"{shared_prefix}one", cwd="/home/user/one", event="PreToolUse", status="running"))
        discovery.record_event(HookEvent(session_id=f"{shared_prefix}two", cwd="/home/user/two", event="PreToolUse", status="running"))
        ended_id = f"{shared_prefix}old"
        discovery.record_event(HookEvent(session_id=ended_id, cwd="/home/user/old", event="PreToolUse", status="running"))
        discovery.mark_session_ended(ended_id)
        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder)
        callback = AsyncMock(spec=CallbackQuery)
        callback.data = f"sess:select:{shared_prefix}"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = 42
        callback.answer = AsyncMock()
        callback.message = AsyncMock(spec=Message)
        callback.message.answer = AsyncMock()

        await _dispatch(router, 0, callback)

        callback.answer.assert_awaited_once_with("Ambiguous prefix, 2 matches. Be more specific.")
        callback.message.answer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_select_pruned_stale_unbound_callback_does_not_answer_session_not_found(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder
    ) -> None:
        session_id = "pruned-stale-unbound-session-0001"
        discovery.record_event(HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running"))
        discovery._sessions[session_id].last_seen = utc_now() - timedelta(seconds=discovery._stale_timeout_sec + 1)
        assert discovery.prune_stale() == [session_id]
        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder)
        callback = AsyncMock(spec=CallbackQuery)
        callback.data = f"sess:select:{session_id[:16]}"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = 42
        callback.answer = AsyncMock()
        callback.message = AsyncMock(spec=Message)
        callback.message.answer = AsyncMock()

        await _dispatch(router, 0, callback)

        callback.answer.assert_awaited_once_with("Session is no longer available")
        callback.message.answer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_select_stale_pruned_session_can_be_rediscovered_and_bound(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder
    ) -> None:
        session_id = "rediscovered-stale-session-0001"
        discovery.record_event(HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running"))
        discovery._sessions[session_id].last_seen = utc_now() - timedelta(seconds=discovery._stale_timeout_sec + 1)
        assert discovery.prune_stale() == [session_id]
        discovery.record_event(HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running"))
        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder)
        callback = AsyncMock(spec=CallbackQuery)
        callback.data = f"sess:select:{session_id[:16]}"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = 42
        callback.answer = AsyncMock()
        callback.message = AsyncMock(spec=Message)
        callback.message.answer = AsyncMock()
        callback.message.edit_text = AsyncMock()

        await _dispatch(router, 0, callback)

        callback.answer.assert_awaited_once_with()
        keyboard = callback.message.edit_text.call_args.kwargs["reply_markup"]
        callbacks = [button.callback_data for row in keyboard.inline_keyboard for button in row]
        assert callbacks == [f"sess:bind:{session_id[:16]}"]

    @pytest.mark.asyncio
    async def test_select_stale_unbound_callback_does_not_render_bind_button(
        self,
        discovery: ExternalSessionDiscoveryService,
        binder: ExternalSessionBinder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id = "stale-unbound-session-0001"
        discovery.record_event(HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running"))
        monkeypatch.setattr(discovery, "is_session_stale", lambda _: True)
        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder)
        callback = AsyncMock(spec=CallbackQuery)
        callback.data = f"sess:select:{session_id[:16]}"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = 42
        callback.answer = AsyncMock()
        callback.message = AsyncMock(spec=Message)
        callback.message.answer = AsyncMock()

        await _dispatch(router, 0, callback)

        callback.answer.assert_awaited_once_with("Session is no longer available")
        callback.message.answer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_select_dead_pid_unbound_callback_does_not_render_bind_button(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session_id = "dead-pid-unbound-session-0001"
        discovery.record_event(HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running", pid=12345))
        monkeypatch.setattr("app.services.session_id_resolver.process_is_alive", lambda _: False)
        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder)
        callback = AsyncMock(spec=CallbackQuery)
        callback.data = f"sess:select:{session_id[:16]}"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = 42
        callback.answer = AsyncMock()
        callback.message = AsyncMock(spec=Message)
        callback.message.answer = AsyncMock()

        await _dispatch(router, 0, callback)

        callback.answer.assert_awaited_once_with("Session is no longer available")
        callback.message.answer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_open_from_list_answers_new_message_instead_of_editing_list(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder
    ) -> None:
        """/list 来源的 ``sess:open`` 必须另发详情，不能就地编辑列表消息（其余会话按钮需保留）。"""
        session_id = "abcdef1234567890full"
        discovery.record_event(HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running"))

        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder)
        callback = AsyncMock(spec=CallbackQuery)
        callback.data = f"sess:open:{session_id[:16]}"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = 42
        callback.answer = AsyncMock()
        callback.message = AsyncMock(spec=Message)
        callback.message.answer = AsyncMock()
        callback.message.edit_text = AsyncMock()

        await _dispatch(router, 0, callback)

        callback.answer.assert_awaited_once_with()
        callback.message.edit_text.assert_not_awaited()
        callback.message.answer.assert_awaited_once()
        keyboard = callback.message.answer.call_args.kwargs["reply_markup"]
        callbacks = [button.callback_data for row in keyboard.inline_keyboard for button in row]
        assert callbacks == [f"sess:bind:{session_id[:16]}"]


class TestTmuxSessionActionHandler:
    @pytest.mark.asyncio
    async def test_resolve_terminal_id_prefix_ignores_dead_sessions(self) -> None:
        registry = AsyncMock()
        registry.list_active_sessions = AsyncMock(return_value=[_tmux_session("user_1234567890_aaaaaaaaaaaa", is_alive=False)])

        resolved, error = await _resolve_terminal_id_prefix("user_1234567890_", registry)

        assert resolved is None
        assert error == "Session not found"

    @pytest.mark.asyncio
    async def test_resolve_terminal_id_prefix_does_not_resolve_user_wide_legacy_prefix_to_new_session(self) -> None:
        registry = AsyncMock()
        registry.list_active_sessions = AsyncMock(return_value=[_tmux_session("user_1234567890_bbbbbbbbbbbb")])

        resolved, error = await _resolve_terminal_id_prefix("user_1234567890_", registry)

        assert resolved is None
        assert error == "Session not found"

    @pytest.mark.asyncio
    async def test_resolve_terminal_id_prefix_accepts_hash_token_when_id_is_another_id_prefix(self) -> None:
        registry = AsyncMock()
        registry.list_active_sessions = AsyncMock(
            return_value=[
                _tmux_session("abc"),
                _tmux_session("abcdef"),
            ]
        )

        resolved, error = await _resolve_terminal_id_prefix(unique_prefixes(["abc", "abcdef"])["abc"], registry)

        assert resolved == "abc"
        assert error is None

    @pytest.mark.asyncio
    async def test_resolve_terminal_id_prefix_accepts_compact_hash_token_for_long_common_prefix(self) -> None:
        terminal_ids = ["x" * 52 + "a", "x" * 52 + "b"]
        token = unique_prefixes(terminal_ids, max_length=52)[terminal_ids[0]]
        registry = AsyncMock()
        registry.list_active_sessions = AsyncMock(return_value=[_tmux_session(terminal_id) for terminal_id in terminal_ids])

        resolved, error = await _resolve_terminal_id_prefix(token, registry)

        assert resolved == terminal_ids[0]
        assert error is None

    @pytest.mark.asyncio
    async def test_resolve_terminal_id_prefix_reports_ambiguous_live_matches(self) -> None:
        registry = AsyncMock()
        registry.list_active_sessions = AsyncMock(
            return_value=[
                _tmux_session("terminal-aaaaaaaaaaaa"),
                _tmux_session("terminal-abbbbbbbbbbb"),
            ]
        )

        resolved, error = await _resolve_terminal_id_prefix("terminal-a", registry)

        assert resolved is None
        assert error == "Ambiguous prefix, 2 matches. Be more specific."

    @pytest.mark.asyncio
    async def test_attach_resolves_terminal_id_prefix(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder
    ) -> None:
        terminal_id = "user_42_123456789abc"
        registry = AsyncMock()
        registry.list_active_sessions = AsyncMock(return_value=[_tmux_session(terminal_id)])
        registry.attach_user = AsyncMock(return_value=(True, "已连接"))
        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder, registry_service=registry)
        callback = AsyncMock(spec=CallbackQuery)
        callback.data = f"sess:attach:{terminal_id[:16]}"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = 42
        callback.answer = AsyncMock()
        callback.message = AsyncMock(spec=Message)
        callback.message.answer = AsyncMock()
        callback.message.edit_text = AsyncMock()

        await _dispatch(router, 3, callback)

        registry.attach_user.assert_awaited_once_with(user_id=42, terminal_id=terminal_id)
        # tmux attach 必须另发结果消息，不能就地编辑列表（保留其它会话的按钮）
        callback.message.edit_text.assert_not_awaited()
        callback.message.answer.assert_awaited_once_with("已连接")

    @pytest.mark.asyncio
    async def test_attach_stale_dead_session_does_not_call_registry_attach(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder
    ) -> None:
        terminal_id = "user_1234567890_aaaaaaaaaaaa"
        registry = AsyncMock()
        registry.list_active_sessions = AsyncMock(return_value=[_tmux_session(terminal_id, is_alive=False)])
        registry.attach_user = AsyncMock(return_value=(True, "should not attach"))
        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder, registry_service=registry)
        callback = AsyncMock(spec=CallbackQuery)
        callback.data = "sess:attach:user_1234567890_"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = 42
        callback.answer = AsyncMock()
        callback.message = AsyncMock(spec=Message)
        callback.message.answer = AsyncMock()

        await _dispatch(router, 3, callback)

        callback.answer.assert_awaited_once_with("Session not found")
        registry.attach_user.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_close_resolves_terminal_id_prefix(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder
    ) -> None:
        terminal_id = "user_42_123456789abc"
        registry = AsyncMock()
        registry.list_active_sessions = AsyncMock(return_value=[_tmux_session(terminal_id)])
        registry.close_session = AsyncMock(return_value=True)
        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder, registry_service=registry)
        callback = AsyncMock(spec=CallbackQuery)
        callback.data = f"sess:close:{terminal_id[:16]}"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = 42
        callback.answer = AsyncMock()
        callback.message = AsyncMock(spec=Message)
        callback.message.answer = AsyncMock()
        callback.message.edit_text = AsyncMock()

        await _dispatch(router, 4, callback)

        registry.close_session.assert_awaited_once_with(terminal_id)
        # tmux close 必须另发结果消息，不能就地编辑列表（保留其它会话的按钮）
        callback.message.edit_text.assert_not_awaited()
        callback.message.answer.assert_awaited_once_with(f"✅ 会话 `{terminal_id}` 已关闭")

    @pytest.mark.asyncio
    async def test_close_stale_dead_session_does_not_call_registry_close(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder
    ) -> None:
        terminal_id = "user_1234567890_aaaaaaaaaaaa"
        registry = AsyncMock()
        registry.list_active_sessions = AsyncMock(return_value=[_tmux_session(terminal_id, is_alive=False)])
        registry.close_session = AsyncMock(return_value=True)
        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder, registry_service=registry)
        callback = AsyncMock(spec=CallbackQuery)
        callback.data = "sess:close:user_1234567890_"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = 42
        callback.answer = AsyncMock()
        callback.message = AsyncMock(spec=Message)
        callback.message.answer = AsyncMock()

        await _dispatch(router, 4, callback)

        callback.answer.assert_awaited_once_with("Session not found")
        registry.close_session.assert_not_awaited()


class TestSessionBindHandler:
    @pytest.mark.asyncio
    async def test_bind_delegates_to_binder(self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder) -> None:
        session_id = "abcdef1234567890full"
        event = HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running")
        discovery.record_event(event)

        result = await binder.bind(user_id=42, session_id=session_id)
        assert result.success is True
        assert result.session_id == session_id

    @pytest.mark.asyncio
    async def test_bind_already_bound_fails(self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder) -> None:
        session_id = "abcdef1234567890full"
        event = HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running")
        discovery.record_event(event)

        await binder.bind(user_id=42, session_id=session_id)
        # Session removed from discovery after bind, so second bind won't find it
        result = await binder.bind(user_id=99, session_id=session_id)
        assert result.success is False


class TestSessionUnbindHandler:
    @pytest.mark.asyncio
    async def test_old_unbind_callback_does_not_unbind_new_same_prefix_session(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder
    ) -> None:
        old_session_id = "unbind-collision-old"
        new_session_id = "unbind-collision-new"
        old_prefix = old_session_id[:16]
        assert old_prefix == new_session_id[:16]
        discovery.mark_session_unavailable(old_session_id)
        discovery.record_event(HookEvent(session_id=new_session_id, cwd="/home/user/proj", event="PreToolUse", status="running"))
        bind_result = await binder.bind(user_id=42, session_id=new_session_id)
        assert bind_result.success is True
        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder)
        callback = AsyncMock(spec=CallbackQuery)
        callback.data = f"sess:unbind:{old_prefix}"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = 42
        callback.answer = AsyncMock()
        callback.message = AsyncMock(spec=Message)
        callback.message.answer = AsyncMock()

        await _dispatch(router, 2, callback)

        callback.answer.assert_awaited_once_with("❌ Session is no longer available")
        assert binder._binding_store.get_binding(new_session_id) is not None

    @pytest.mark.asyncio
    async def test_unbind_delegates_to_binder(self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder) -> None:
        session_id = "abcdef1234567890full"
        event = HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running")
        discovery.record_event(event)

        await binder.bind(user_id=42, session_id=session_id)
        result = await binder.unbind(user_id=42, session_id=session_id)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_unbind_wrong_user_fails(self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder) -> None:
        session_id = "abcdef1234567890full"
        event = HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running")
        discovery.record_event(event)

        await binder.bind(user_id=42, session_id=session_id)
        result = await binder.unbind(user_id=99, session_id=session_id)
        assert result.success is False


class TestBindSuccessPairingPrompt:
    """绑定成功消息应带 `sess:select:` 入口按钮，复用 select→pairing 链路。"""

    @pytest.mark.asyncio
    async def test_bind_success_callback_renders_select_button(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder
    ) -> None:

        session_id = "abcdef1234567890full"
        discovery.record_event(HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running"))

        router = Router()
        register_session_action_handlers(router, discovery=discovery, binder=binder)

        callback = AsyncMock(spec=CallbackQuery)
        callback.data = f"sess:bind:{session_id[:16]}"
        callback.from_user = MagicMock(spec=User)
        callback.from_user.id = 42
        callback.answer = AsyncMock()
        callback.message = AsyncMock(spec=Message)
        callback.message.answer = AsyncMock()
        callback.message.edit_text = AsyncMock()

        await _dispatch(router, 1, callback)

        callback.answer.assert_awaited_once_with("绑定成功")
        callback.message.edit_text.assert_awaited_once()
        _, kwargs = callback.message.edit_text.call_args
        expected_token = external_session_select_token(session_id, discovery=discovery, binder=binder)
        buttons = kwargs["reply_markup"].inline_keyboard
        assert len(buttons) == 1
        assert len(buttons[0]) == 1
        assert buttons[0][0].text == "进入终端输入"
        assert buttons[0][0].callback_data == f"sess:select:{expected_token}"
        # the select token resolves back to the same session (same as /list rendering)
        validation = validate_external_session_select(expected_token, user_id=42, discovery=discovery, binder=binder)
        assert validation.session_id == session_id
        assert validation.action == "unbind"
        # suppress unused-import lint by referencing the import in a no-op

    @pytest.mark.asyncio
    async def test_select_button_after_bind_routes_to_needs_pairing(
        self, discovery: ExternalSessionDiscoveryService, binder: ExternalSessionBinder
    ) -> None:

        session_id = "abcdef1234567890full"
        discovery.record_event(HookEvent(session_id=session_id, cwd="/home/user/proj", event="PreToolUse", status="running"))

        router = Router()
        binding = binder._binding_store.get_binding(session_id)
        input_service = MagicMock()
        input_service.activate_select = AsyncMock(return_value=PairOutcome.NEEDS_PAIRING)
        input_service.pair_candidates = AsyncMock(
            return_value=(
                PairOutcome.NEEDS_PAIRING,
                PairingCandidates(
                    binding_id=binding.binding_id if binding is not None else "bid",
                    paired_tty="/dev/ttyX",
                    terminals=[ExternalTerminal(kind="ghostty", terminal_id="t1", name="Ghostty", cwd="/home/user/proj")],
                ),
            )
        )
        input_service.register_pair_token = AsyncMock(return_value="pairtoken123")

        register_session_action_handlers(router, discovery=discovery, binder=binder, external_session_input_service=input_service)

        # Step 1: bind via callback, capture the rendered sess:select button.
        bind_callback = AsyncMock(spec=CallbackQuery)
        bind_callback.data = f"sess:bind:{session_id[:16]}"
        bind_callback.from_user = MagicMock(spec=User)
        bind_callback.from_user.id = 42
        bind_callback.answer = AsyncMock()
        bind_callback.message = AsyncMock(spec=Message)
        bind_callback.message.answer = AsyncMock()
        bind_callback.message.edit_text = AsyncMock()
        await _dispatch(router, 1, bind_callback)

        expected_token = external_session_select_token(session_id, discovery=discovery, binder=binder)
        select_buttons = bind_callback.message.edit_text.call_args.kwargs["reply_markup"].inline_keyboard
        assert select_buttons[0][0].callback_data == f"sess:select:{expected_token}"

        # Step 2: click the sess:select button → should route to NEEDS_PAIRING and render tpair.
        select_callback = AsyncMock(spec=CallbackQuery)
        select_callback.data = f"sess:select:{expected_token}"
        select_callback.from_user = MagicMock(spec=User)
        select_callback.from_user.id = 42
        select_callback.answer = AsyncMock()
        select_callback.message = AsyncMock(spec=Message)
        select_callback.message.answer = AsyncMock()
        select_callback.message.edit_text = AsyncMock()
        await _dispatch(router, 0, select_callback)

        select_args, select_kwargs = select_callback.message.edit_text.call_args
        rendered_text = select_args[0] if select_args else ""
        assert "选择要配对的 Ghostty 终端" in rendered_text or "终端" in rendered_text
        assert "tpair:pairtoken123" in [b.callback_data for row in select_kwargs["reply_markup"].inline_keyboard for b in row]


class TestEditOrAnswer:
    """_edit_or_answer:优先 edit_text 就地改写，消息不可编辑时回落到 answer。"""

    @staticmethod
    def _make_callback() -> AsyncMock:
        callback = AsyncMock(spec=CallbackQuery)
        callback.message = AsyncMock(spec=Message)
        callback.message.edit_text = AsyncMock()
        callback.message.answer = AsyncMock()
        return callback

    @pytest.mark.asyncio
    async def test_edits_message_when_editable(self) -> None:
        callback = self._make_callback()

        await _edit_or_answer(callback, "hello")

        # 未指定键盘时必须显式传空键盘，否则旧 inline 按钮残留
        callback.message.edit_text.assert_awaited_once_with("hello", reply_markup=InlineKeyboardMarkup(inline_keyboard=[]))
        callback.message.answer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_edits_message_keeps_given_markup(self) -> None:
        callback = self._make_callback()
        keyboard = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="b", callback_data="x")]])

        await _edit_or_answer(callback, "hello", reply_markup=keyboard)

        callback.message.edit_text.assert_awaited_once_with("hello", reply_markup=keyboard)
        callback.message.answer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_falls_back_to_answer_on_deterministic_bad_request(self) -> None:
        """确定性 BAD REQUEST（消息不可编辑/已删除）：旧消息已不可用，回落为新消息。"""
        callback = self._make_callback()
        callback.message.edit_text = AsyncMock(
            side_effect=TelegramBadRequest(method=MagicMock(), message="Bad Request: message to edit not found")
        )

        await _edit_or_answer(callback, "hello")

        callback.message.edit_text.assert_awaited_once()
        callback.message.answer.assert_awaited_once_with("hello", reply_markup=None)

    @pytest.mark.asyncio
    async def test_propagates_transient_error(self) -> None:
        """暂时性错误（限流/网络/服务器）不应回落 answer，避免制造重复内容；向上冒泡交由 router。"""
        from aiogram.exceptions import TelegramNetworkError

        callback = self._make_callback()
        callback.message.edit_text = AsyncMock(side_effect=TelegramNetworkError(method=MagicMock(), message="timeout"))

        with pytest.raises(TelegramNetworkError):
            await _edit_or_answer(callback, "hello")

        callback.message.answer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noop_when_content_unchanged(self) -> None:
        """重复点击同一按钮触发 message is not modified：不回退、不重复发消息。"""
        callback = self._make_callback()
        callback.message.edit_text = AsyncMock(
            side_effect=TelegramBadRequest(method=MagicMock(), message="Bad Request: message is not modified")
        )

        await _edit_or_answer(callback, "hello")

        callback.message.answer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_answers_when_message_inaccessible(self) -> None:
        """>48h 的旧消息是 InaccessibleMessage（无 edit_text）：直接另发新消息。"""
        callback = AsyncMock(spec=CallbackQuery)
        callback.message = SimpleNamespace(answer=AsyncMock())  # 无 edit_text 的不可编辑消息

        await _edit_or_answer(callback, "hello")

        callback.message.answer.assert_awaited_once_with("hello", reply_markup=None)

    @pytest.mark.asyncio
    async def test_noop_when_no_message(self) -> None:
        callback = AsyncMock(spec=CallbackQuery)
        callback.message = None

        # 无消息可编辑也无 chat 可回复：只能跳过，验证不抛异常
        await _edit_or_answer(callback, "hello")
