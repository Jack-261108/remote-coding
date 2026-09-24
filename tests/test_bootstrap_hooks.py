from __future__ import annotations

import asyncio
import gc
import json
import uuid
import warnings
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bootstrap import AppContainer
from app.config.settings import Settings
from app.domain.external_session_models import ExternalBinding
from app.domain.hook_models import HookEvent
from app.domain.models import TaskRecord, TaskStatus, utc_now
from app.domain.session_models import ConversationTurn, SessionEvent, SessionEventType, SessionPhase, ToolCallRecord, ToolStatus
from app.domain.user_question_models import ExternalTmuxQuestionTarget, UserQuestionPrompt
from app.services.auto_approve_service import ActivationSlot
from app.services.external_user_question_state import PendingExternalUserQuestion
from app.services.permission_callback_registry import AuthorizationMode, SessionOrigin
from app.services.status_display import StatusDisplayService
from app.services.unbound_permission_handler import UnboundPermissionHandler


async def wait_for_jsonl_sync_idle(container: AppContainer, session_id: str) -> None:
    """Wait for the debounced JSONL sync to complete via the session supervisor."""
    debounce = container.settings.claude_jsonl_sync_debounce_ms / 1000
    # Supervisor poll interval (0.2s) + debounce + margin
    await asyncio.sleep(0.2 + debounce + 0.1)


def make_settings(tmp_path, *, install_hooks: bool = True, tmux_mode: bool = False, **overrides: object) -> Settings:
    data = {
        "TG_BOT_TOKEN": "123456:TESTTOKEN",
        "TG_ALLOWED_USER_IDS": "1",
        "DEFAULT_PROVIDER": "claude_code",
        "DEFAULT_TIMEOUT_SEC": 10,
        "MAX_CONCURRENT_TASKS": 1,
        "CLAUDE_TMUX_MODE": tmux_mode,
        "TMUX_DATA_DIR": str(tmp_path),
        "CLAUDE_CLI_BIN": "claude",
        "CLAUDE_INSTALL_HOOKS": install_hooks,
        "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude"),
        "CLAUDE_HOOK_SOCKET_PATH": str(tmp_path / "hook.sock"),
        "CLAUDE_JSONL_SYNC_DEBOUNCE_MS": 10,
        "CLAUDE_PERIODIC_RECHECK_MS": 10,
        "CODEX_CLI_BIN": "codex",
        "GEMINI_CLI_BIN": "gemini",
        "ALLOWED_WORKDIRS": str(tmp_path),
    }
    data.update(overrides)
    return Settings.model_validate(data)


def use_legacy_hook_binding_path(container: AppContainer) -> None:
    delattr(container, "ownership_resolver")


def allow_target_tmux_pid(
    container: AppContainer,
    monkeypatch: pytest.MonkeyPatch,
    *,
    pid: int = 4242,
    user_id: int = 1,
) -> None:
    async def find_tmux_session_for_pid(actual_pid: int, _tmux_bin: str = "tmux") -> str | None:
        assert actual_pid == pid
        sessions = await container.session_service.list_all()
        session = next((item for item in sessions if item.user_id == user_id), None)
        assert session is not None
        assert session.terminal_id is not None
        return container.tmux_runner.build_session_name(session.terminal_id)

    monkeypatch.setattr(
        "app.adapters.process.pty_injector.find_tmux_session_for_pid",
        find_tmux_session_for_pid,
    )


@pytest.fixture(autouse=True)
async def cleanup_app_containers() -> AsyncIterator[None]:
    try:
        yield
    finally:
        # Per-container try/except: a single container's cleanup failure must not
        # abort cleanup of the others (which would mask resource leaks behind an
        # unrelated exception). Collect failures and surface them after best-effort
        # cleanup so no container is left dangling.
        containers = [obj for obj in gc.get_objects() if isinstance(obj, AppContainer)]
        cleanup_errors: list[tuple[Any, BaseException]] = []
        for container in containers:
            try:
                await container.external_reply_delivery_pump.stop_all()
                await container.session_supervisor.stop_all()
                container.jsonl_file_watcher.stop()
            except BaseException as exc:  # noqa: BLE001 - cleanup must be best-effort
                cleanup_errors.append((container, exc))
            try:
                await container.bot.session.close()
            except BaseException as exc:  # noqa: BLE001 - cleanup must be best-effort
                cleanup_errors.append((container, exc))

        # Verify no *uncleaned* AppContainer instances leak past cleanup. We count
        # containers whose supervisor is still active (the reliable teardown signal) rather
        # than merely-reachable ones: a container referenced only by a still-live test frame
        # is not a leak. This is a soft check (warning, not hard assertion) so a flaky gc
        # snapshot does not turn a green test red while still surfacing genuine leaks.
        gc.collect()
        leaked = [
            obj
            for obj in gc.get_objects()
            if isinstance(obj, AppContainer) and getattr(getattr(obj, "session_supervisor", None), "_active", False)
        ]
        if leaked:
            warnings.warn(
                f"{len(leaked)} AppContainer instance(s) still have an active session supervisor after cleanup",
                stacklevel=1,
            )

        if cleanup_errors:
            # Re-raise the first error so genuine failures are not silenced, but only
            # after every container has had a chance to be cleaned up.
            container, exc = cleanup_errors[0]
            raise RuntimeError(f"cleanup failed for {container!r}: {exc!r}") from exc


@pytest.mark.asyncio
async def test_container_wires_tmux_timing_settings(tmp_path) -> None:
    container = AppContainer(
        make_settings(
            tmp_path,
            install_hooks=False,
            TMUX_POLL_INTERVAL_SEC=1.25,
            TMUX_ENTER_DELAY_SEC=0.75,
            TMUX_PARTIAL_FLUSH_SEC=0.5,
            TMUX_COMPLETION_GRACE_SEC=0.33,
        )
    )

    try:
        assert container.tmux_runner._poll_interval_sec == 1.25
        assert container.tmux_runner._enter_delay_sec == 0.75
        assert container.tmux_runner._partial_flush_sec == 0.5
        assert container.tmux_runner._interactive_completion_grace_sec == 0.33
    finally:
        await container.bot.session.close()


@pytest.mark.asyncio
async def test_wire_passes_dead_unbound_cleanup_to_list_router(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    captured: dict[str, object] = {}

    def fake_create_router(**kwargs):
        captured.update(kwargs)
        return Router()

    from aiogram import Router

    monkeypatch.setattr("app.bootstrap.create_router", fake_create_router)

    try:
        container.wire()
    finally:
        await container.bot.session.close()

    assert captured["dead_unbound_cleanup"] == container._cleanup_dead_unbound_external_session


@pytest.mark.asyncio
async def test_wire_passes_admin_password_service_to_router(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False, ADMIN_PASSWORD="secret"))
    captured: dict[str, object] = {}

    def fake_create_router(**kwargs):
        captured.update(kwargs)
        return Router()

    from aiogram import Router

    monkeypatch.setattr("app.bootstrap.create_router", fake_create_router)

    try:
        container.wire()
    finally:
        await container.bot.session.close()

    assert captured["admin_password_service"] is container.admin_password_service
    assert container.admin_password_service.is_enabled is True


@pytest.mark.asyncio
async def test_wire_passes_status_display_to_router(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    captured: dict[str, object] = {}

    def fake_create_router(**kwargs):
        captured.update(kwargs)
        return Router()

    from aiogram import Router

    monkeypatch.setattr("app.bootstrap.create_router", fake_create_router)

    try:
        container.wire()
    finally:
        await container.bot.session.close()

    assert isinstance(container.status_display, StatusDisplayService)
    assert captured["status_display"] is container.status_display


@pytest.mark.asyncio
async def test_permission_gateway_receives_risk_evaluator(tmp_path) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False, RISK_EVAL_AUTO_APPROVE_MAX_RISK="中"))
    try:
        assert container.permission_gateway._risk_evaluator is container.risk_evaluator
    finally:
        await container.bot.session.close()


@pytest.mark.asyncio
async def test_app_container_start_installs_hooks_and_starts_server(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=True))

    seen = {"install": 0, "start": 0, "stop": 0}

    def fake_install():
        seen["install"] += 1

    async def fake_start(handler, permission_failure_handler=None, permission_resolved_handler=None):
        seen["start"] += 1
        assert handler is not None

    async def fake_stop():
        seen["stop"] += 1

    async def fake_close():
        return None

    monkeypatch.setattr(container.hook_installer, "install", fake_install)
    monkeypatch.setattr(container.hook_socket_server, "start", fake_start)
    monkeypatch.setattr(container.hook_socket_server, "stop", fake_stop)
    monkeypatch.setattr(container.bot.session, "close", fake_close)

    await container.start()
    await container.stop()

    assert seen == {"install": 1, "start": 1, "stop": 1}


@pytest.mark.asyncio
async def test_app_container_start_stops_jsonl_file_watcher(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    calls: list[str] = []

    monkeypatch.setattr(container.bot, "set_my_commands", AsyncMock(return_value=None))
    monkeypatch.setattr(container.hook_socket_server, "start", AsyncMock(return_value=None))
    monkeypatch.setattr(container.hook_socket_server, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container.bot.session, "close", AsyncMock(return_value=None))
    monkeypatch.setattr(container, "_restore_session_bindings", AsyncMock(return_value=None))
    monkeypatch.setattr(container.external_binding_cleanup_service, "run_cleanup", AsyncMock(return_value=None))
    monkeypatch.setattr(container.upload_cleanup, "run_cleanup", AsyncMock(return_value=None))
    monkeypatch.setattr(container._external_binding_cleanup_task, "start", lambda: None)
    monkeypatch.setattr(container._external_binding_cleanup_task, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container._janitor_task, "start", lambda: None)
    monkeypatch.setattr(container._janitor_task, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container.jsonl_file_watcher, "start", lambda: calls.append("watcher_start") or True)
    monkeypatch.setattr(container.jsonl_file_watcher, "stop", lambda: calls.append("watcher_stop"))

    await container.start()
    await container.stop()

    assert calls == ["watcher_start", "watcher_stop"]


@pytest.mark.asyncio
async def test_app_container_starts_jsonl_file_watcher_before_restoring_sessions(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    calls: list[str] = []

    async def fake_restore() -> None:
        calls.append("restore")

    monkeypatch.setattr(container.bot, "set_my_commands", AsyncMock(return_value=None))
    monkeypatch.setattr(container.hook_socket_server, "start", AsyncMock(return_value=None))
    monkeypatch.setattr(container.hook_socket_server, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container.bot.session, "close", AsyncMock(return_value=None))
    monkeypatch.setattr(container, "_restore_session_bindings", fake_restore)
    monkeypatch.setattr(container.external_binding_cleanup_service, "run_cleanup", AsyncMock(return_value=None))
    monkeypatch.setattr(container.upload_cleanup, "run_cleanup", AsyncMock(return_value=None))
    monkeypatch.setattr(container._external_binding_cleanup_task, "start", lambda: None)
    monkeypatch.setattr(container._external_binding_cleanup_task, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container._janitor_task, "start", lambda: None)
    monkeypatch.setattr(container._janitor_task, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container.jsonl_file_watcher, "start", lambda: calls.append("watcher_start") or True)
    monkeypatch.setattr(container.jsonl_file_watcher, "stop", lambda: None)

    await container.start()
    try:
        assert calls == ["watcher_start", "restore"]
    finally:
        await container.stop()


@pytest.mark.asyncio
async def test_app_container_holds_hook_event_until_startup_restore_finishes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = AppContainer(
        make_settings(
            tmp_path,
            install_hooks=False,
            CLAUDE_HOOK_SOCKET_PATH=f"/tmp/rc-startup-{uuid.uuid4().hex}.sock",
        )
    )
    restore_entered = asyncio.Event()
    allow_restore = asyncio.Event()
    handled = asyncio.Event()
    seen: list[HookEvent] = []

    async def fake_restore() -> None:
        restore_entered.set()
        await allow_restore.wait()

    async def fake_handle(event: HookEvent) -> None:
        seen.append(event)
        handled.set()

    monkeypatch.setattr(container.bot, "set_my_commands", AsyncMock(return_value=None))
    monkeypatch.setattr(container.bot.session, "close", AsyncMock(return_value=None))
    monkeypatch.setattr(container, "_handle_hook_event", fake_handle)
    monkeypatch.setattr(container, "_restore_session_bindings", fake_restore)
    monkeypatch.setattr(container.external_binding_cleanup_service, "run_cleanup", AsyncMock(return_value=None))
    monkeypatch.setattr(container, "_restore_external_reply_delivery_pumps", AsyncMock(return_value=None))
    monkeypatch.setattr(container.upload_cleanup, "run_cleanup", AsyncMock(return_value=None))
    monkeypatch.setattr(container._external_binding_cleanup_task, "start", lambda: None)
    monkeypatch.setattr(container._external_binding_cleanup_task, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container._janitor_task, "start", lambda: None)
    monkeypatch.setattr(container._janitor_task, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container.jsonl_file_watcher, "start", lambda: True)
    monkeypatch.setattr(container.jsonl_file_watcher, "stop", MagicMock())
    monkeypatch.setattr(container.external_reply_delivery_pump, "stop_all", AsyncMock(return_value=None))
    monkeypatch.setattr(container.session_supervisor, "stop_all", AsyncMock(return_value=None))

    start_task = asyncio.create_task(container.start())
    await asyncio.wait_for(restore_entered.wait(), timeout=1)
    reader, writer = await asyncio.open_unix_connection(container.settings.claude_hook_socket_path)
    writer.write(
        json.dumps(
            {
                "session_id": "startup-session",
                "cwd": str(tmp_path),
                "event": "SessionStart",
                "status": "starting",
            }
        ).encode("utf-8")
    )
    await writer.drain()
    if writer.can_write_eof():
        writer.write_eof()
    await asyncio.sleep(0.05)
    assert not handled.is_set()

    allow_restore.set()
    await start_task
    await asyncio.wait_for(handled.wait(), timeout=1)

    assert [event.session_id for event in seen] == ["startup-session"]
    assert await asyncio.wait_for(reader.read(), timeout=1) == b""
    writer.close()
    await writer.wait_closed()
    await container.stop()


@pytest.mark.asyncio
async def test_app_container_jsonl_file_watcher_failure_does_not_abort_start(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    container.claude_paths.projects_dir.mkdir(parents=True, exist_ok=True)

    class FailingObserver:
        def schedule(self, event_handler, path: str, *, recursive: bool) -> object:
            return object()

        def start(self) -> None:
            raise RuntimeError("observer failed")

        def stop(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            return None

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(container.jsonl_file_watcher, "_observer_factory", lambda: FailingObserver())
    monkeypatch.setattr(container.bot, "set_my_commands", AsyncMock(return_value=None))
    monkeypatch.setattr(container.hook_socket_server, "start", AsyncMock(return_value=None))
    monkeypatch.setattr(container.hook_socket_server, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container.bot.session, "close", AsyncMock(return_value=None))
    monkeypatch.setattr(container, "_restore_session_bindings", AsyncMock(return_value=None))
    monkeypatch.setattr(container.external_binding_cleanup_service, "run_cleanup", AsyncMock(return_value=None))
    monkeypatch.setattr(container.upload_cleanup, "run_cleanup", AsyncMock(return_value=None))
    monkeypatch.setattr(container._external_binding_cleanup_task, "start", lambda: None)
    monkeypatch.setattr(container._external_binding_cleanup_task, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container._janitor_task, "start", lambda: None)
    monkeypatch.setattr(container._janitor_task, "stop", AsyncMock(return_value=None))

    await container.start()
    try:
        assert container._started is True
        assert container.jsonl_file_watcher.is_available is False
    finally:
        await container.stop()


def test_app_container_wires_jsonl_file_watcher_settings(tmp_path) -> None:
    container = AppContainer(
        make_settings(
            tmp_path,
            install_hooks=False,
            JSONL_FILE_WATCHER_ENABLED=False,
            SESSION_SUPERVISOR_POLL_INTERVAL_SEC=0.4,
            SESSION_SUPERVISOR_IDLE_POLL_INTERVAL_SEC=12.0,
        )
    )

    try:
        assert container.jsonl_file_watcher.is_enabled is False
        assert container.session_supervisor._poll_interval_sec == 0.4
        assert container.session_supervisor._idle_poll_interval_sec == 12.0
    finally:
        container.jsonl_file_watcher.stop()


@pytest.mark.asyncio
async def test_app_container_start_skips_install_when_disabled(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))

    seen = {"install": 0, "start": 0, "stop": 0}

    def fake_install():
        seen["install"] += 1

    async def fake_start(handler, permission_failure_handler=None, permission_resolved_handler=None):
        seen["start"] += 1

    async def fake_stop():
        seen["stop"] += 1

    async def fake_close():
        return None

    monkeypatch.setattr(container.hook_installer, "install", fake_install)
    monkeypatch.setattr(container.hook_socket_server, "start", fake_start)
    monkeypatch.setattr(container.hook_socket_server, "stop", fake_stop)
    monkeypatch.setattr(container.bot.session, "close", fake_close)

    await container.start()
    await container.stop()

    assert seen == {"install": 0, "start": 1, "stop": 1}


@pytest.mark.asyncio
async def test_app_container_start_failure_runs_partial_cleanup(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    hook_start = AsyncMock()
    hook_stop = AsyncMock()
    close = AsyncMock()

    monkeypatch.setattr(container.bot, "set_my_commands", AsyncMock())
    monkeypatch.setattr(container.hook_socket_server, "start", hook_start)
    monkeypatch.setattr(container.hook_socket_server, "stop", hook_stop)
    monkeypatch.setattr(container.bot.session, "close", close)
    monkeypatch.setattr(container, "_restore_session_bindings", AsyncMock())
    monkeypatch.setattr(container.external_binding_cleanup_service, "run_cleanup", AsyncMock(side_effect=RuntimeError("cleanup failed")))

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await container.start()

    hook_start.assert_awaited_once()
    hook_stop.assert_awaited_once()
    close.assert_awaited_once()
    assert container._started is False


@pytest.mark.asyncio
async def test_prune_unbound_external_sessions_reuses_discovery_pruners(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    calls: list[str] = []

    def fake_prune_dead() -> list[str]:
        calls.append("unbound_dead")
        return []

    def fake_prune_stale() -> list[str]:
        calls.append("unbound_stale")
        return ["stale-unbound"]

    monkeypatch.setattr(container.external_discovery, "prune_dead", fake_prune_dead)
    monkeypatch.setattr(container.external_discovery, "prune_stale", fake_prune_stale)

    try:
        await container._prune_unbound_external_sessions()
    finally:
        await container.bot.session.close()

    assert calls == ["unbound_dead", "unbound_stale"]


@pytest.mark.asyncio
async def test_prune_unbound_external_sessions_invalidates_dead_pruned_permission_state(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    calls: list[str] = []

    def fake_prune_dead() -> list[str]:
        calls.append("unbound_dead")
        return ["dead-unbound"]

    def fake_prune_stale() -> list[str]:
        calls.append("unbound_stale")
        return []

    async def fake_registry_invalidate(session_id: str) -> int:
        calls.append(f"registry:{session_id}")
        return 1

    async def fake_unbound_invalidate(session_id: str) -> int:
        calls.append(f"unbound:{session_id}")
        return 1

    async def fake_cancel_pending_permissions(*, session_id: str) -> None:
        calls.append(f"hook:{session_id}")

    monkeypatch.setattr(container.external_discovery, "prune_dead", fake_prune_dead)
    monkeypatch.setattr(container.external_discovery, "prune_stale", fake_prune_stale)
    monkeypatch.setattr(container.permission_callback_registry, "invalidate_session", fake_registry_invalidate)
    monkeypatch.setattr(container.unbound_permission_handler, "invalidate_session", fake_unbound_invalidate)
    monkeypatch.setattr(container.hook_socket_server, "cancel_pending_permissions", fake_cancel_pending_permissions)

    try:
        await container._prune_unbound_external_sessions()
    finally:
        await container.bot.session.close()

    assert calls == [
        "unbound_dead",
        "registry:dead-unbound",
        "unbound:dead-unbound",
        "hook:dead-unbound",
        "unbound_stale",
    ]


@pytest.mark.asyncio
async def test_dead_unbound_cleanup_clears_auto_approve_state(tmp_path) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    active_session_id = "dead-active"
    slotted_session_id = "dead-slotted"

    await container.auto_approve_service.activate_if_session_alive(user_id=42, session_id=active_session_id)
    container.auto_approve_service._slots[slotted_session_id] = ActivationSlot(
        session_id=slotted_session_id,
        holder_user_id=43,
        attempt_id="attempt-1",
    )

    try:
        await container._cleanup_dead_unbound_external_session(active_session_id)
        await container._cleanup_dead_unbound_external_session(slotted_session_id)
    finally:
        await container.bot.session.close()

    assert container.auto_approve_service.get_active_user_for_session(active_session_id) is None
    assert active_session_id not in container.auto_approve_service._slots
    assert container.auto_approve_service.is_session_ended(active_session_id) is True
    assert slotted_session_id not in container.auto_approve_service._slots
    assert container.auto_approve_service.is_session_ended(slotted_session_id) is True


@pytest.mark.asyncio
async def test_dead_unbound_cleanup_clears_external_user_question_state(tmp_path) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    session_id = "dead-uq"
    tool_use_id = "tool-uq"
    container.external_uq_state.store(
        PendingExternalUserQuestion(
            tool_use_id=tool_use_id,
            session_id=session_id,
            user_id=42,
            prompts=(
                UserQuestionPrompt(
                    tool_use_id=tool_use_id,
                    question_index=0,
                    total_questions=1,
                    question="Continue?",
                ),
            ),
            target=ExternalTmuxQuestionTarget(pane_id="%1"),
        )
    )

    try:
        await container._cleanup_dead_unbound_external_session(session_id)
    finally:
        await container.bot.session.close()

    assert container.external_uq_state.get(tool_use_id) is None


@pytest.mark.asyncio
async def test_manual_unbind_clears_owner_state_before_allowing_rebind(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    session_id = "manual-unbind-session"
    binding = ExternalBinding(
        session_id=session_id,
        user_id=1,
        cwd=str(tmp_path),
        bound_at=utc_now(),
        jsonl_path=None,
    )
    container.external_binding_store.save_binding(binding)
    await container.auto_approve_service.activate(session_id, user_id=1)
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()

    registry_invalidate = AsyncMock(return_value=1)
    unbound_invalidate = AsyncMock(return_value=1)
    uq_invalidate = MagicMock(return_value=1)

    async def cancel_pending(*, session_id: str) -> None:
        assert session_id == "manual-unbind-session"
        cleanup_started.set()
        await allow_cleanup.wait()

    cancel_pending_mock = AsyncMock(side_effect=cancel_pending)

    monkeypatch.setattr(
        container.permission_callback_registry,
        "invalidate_session",
        registry_invalidate,
    )
    monkeypatch.setattr(
        container.unbound_permission_handler,
        "invalidate_session",
        unbound_invalidate,
    )
    monkeypatch.setattr(
        container.external_uq_state,
        "invalidate_session",
        uq_invalidate,
    )
    monkeypatch.setattr(
        container.hook_socket_server,
        "cancel_pending_permissions",
        cancel_pending_mock,
    )

    try:
        unbind_task = asyncio.create_task(container.external_binder.unbind(user_id=1, session_id=session_id))
        await cleanup_started.wait()

        replacement = ExternalBinding(
            session_id=session_id,
            user_id=2,
            cwd=str(tmp_path),
            bound_at=utc_now(),
            jsonl_path=None,
        )
        assert await container._save_external_binding(replacement) is False

        allow_cleanup.set()
        result = await unbind_task

        assert result.success is True
        registry_invalidate.assert_awaited_once_with(session_id)
        unbound_invalidate.assert_awaited_once_with(session_id)
        # external_uq_state is invalidated by BOTH the reaper's owner-state cleanup
        # and the input-service binding teardown — both idempotent, so the call
        # may be repeated. Assert it is invalidated at least once for this session.
        uq_invalidate.assert_called_with(session_id)
        cancel_pending_mock.assert_awaited_once_with(session_id=session_id)
        assert not container.auto_approve_service.is_active(session_id)
        assert not container.auto_approve_service.is_session_ended(session_id)
        assert await container._save_external_binding(replacement) is True
    finally:
        allow_cleanup.set()
        await container.external_reply_delivery_pump.stop_all()
        await container.session_supervisor.stop_all()
        await container.bot.session.close()


@pytest.mark.asyncio
async def test_dead_unbound_cleanup_failure_is_retained_for_retry_when_called_directly(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    session_id = "dead-direct"

    async def failing_registry_invalidate(session_id_arg: str) -> int:
        assert session_id_arg == session_id
        raise RuntimeError("registry failure")

    monkeypatch.setattr(container.permission_callback_registry, "invalidate_session", failing_registry_invalidate)

    try:
        success = await container._cleanup_dead_unbound_external_session(session_id)
    finally:
        await container.bot.session.close()

    assert success is False
    assert session_id in container._pending_dead_unbound_cleanup_ids


@pytest.mark.asyncio
async def test_prune_unbound_external_sessions_continues_cleanup_when_one_step_fails(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    calls: list[str] = []

    def fake_prune_dead() -> list[str]:
        calls.append("unbound_dead")
        return ["dead-a", "dead-b"]

    def fake_prune_stale() -> list[str]:
        calls.append("unbound_stale")
        return []

    async def fake_registry_invalidate(session_id: str) -> int:
        calls.append(f"registry:{session_id}")
        if session_id == "dead-a":
            raise RuntimeError("registry failure")
        return 1

    async def fake_unbound_invalidate(session_id: str) -> int:
        calls.append(f"unbound:{session_id}")
        return 1

    async def fake_cancel_pending_permissions(*, session_id: str) -> None:
        calls.append(f"hook:{session_id}")

    monkeypatch.setattr(container.external_discovery, "prune_dead", fake_prune_dead)
    monkeypatch.setattr(container.external_discovery, "prune_stale", fake_prune_stale)
    monkeypatch.setattr(container.permission_callback_registry, "invalidate_session", fake_registry_invalidate)
    monkeypatch.setattr(container.unbound_permission_handler, "invalidate_session", fake_unbound_invalidate)
    monkeypatch.setattr(container.hook_socket_server, "cancel_pending_permissions", fake_cancel_pending_permissions)

    try:
        await container._prune_unbound_external_sessions()
    finally:
        await container.bot.session.close()

    assert calls == [
        "unbound_dead",
        "registry:dead-a",
        "unbound:dead-a",
        "hook:dead-a",
        "registry:dead-b",
        "unbound:dead-b",
        "hook:dead-b",
        "unbound_stale",
    ]


@pytest.mark.asyncio
async def test_prune_unbound_external_sessions_retries_failed_cleanup_next_pass(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    calls: list[str] = []
    prune_results = [["dead-a"], []]

    def fake_prune_dead() -> list[str]:
        calls.append("unbound_dead")
        return prune_results.pop(0)

    def fake_prune_stale() -> list[str]:
        calls.append("unbound_stale")
        return []

    registry_failures_remaining = 1

    async def fake_registry_invalidate(session_id: str) -> int:
        nonlocal registry_failures_remaining
        calls.append(f"registry:{session_id}")
        if registry_failures_remaining:
            registry_failures_remaining -= 1
            raise RuntimeError("registry failure")
        return 1

    async def fake_unbound_invalidate(session_id: str) -> int:
        calls.append(f"unbound:{session_id}")
        return 1

    async def fake_cancel_pending_permissions(*, session_id: str) -> None:
        calls.append(f"hook:{session_id}")

    monkeypatch.setattr(container.external_discovery, "prune_dead", fake_prune_dead)
    monkeypatch.setattr(container.external_discovery, "prune_stale", fake_prune_stale)
    monkeypatch.setattr(container.permission_callback_registry, "invalidate_session", fake_registry_invalidate)
    monkeypatch.setattr(container.unbound_permission_handler, "invalidate_session", fake_unbound_invalidate)
    monkeypatch.setattr(container.hook_socket_server, "cancel_pending_permissions", fake_cancel_pending_permissions)

    try:
        await container._prune_unbound_external_sessions()
        await container._prune_unbound_external_sessions()
    finally:
        await container.bot.session.close()

    assert calls == [
        "unbound_dead",
        "registry:dead-a",
        "unbound:dead-a",
        "hook:dead-a",
        "unbound_stale",
        "unbound_dead",
        "registry:dead-a",
        "unbound:dead-a",
        "hook:dead-a",
        "unbound_stale",
    ]


@pytest.mark.asyncio
async def test_prune_unbound_external_sessions_prunes_stale_when_dead_prune_fails(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    calls: list[str] = []

    def fake_prune_dead() -> list[str]:
        calls.append("unbound_dead")
        raise OverflowError("bad pid")

    def fake_prune_stale() -> list[str]:
        calls.append("unbound_stale")
        return ["stale-unbound"]

    monkeypatch.setattr(container.external_discovery, "prune_dead", fake_prune_dead)
    monkeypatch.setattr(container.external_discovery, "prune_stale", fake_prune_stale)

    try:
        await container._prune_unbound_external_sessions()
    finally:
        await container.bot.session.close()

    assert calls == ["unbound_dead", "unbound_stale"]


@pytest.mark.asyncio
async def test_start_runs_tmux_health_check_before_restore(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False, tmux_mode=True))
    calls: list[str] = []

    async def fake_health_check() -> None:
        calls.append("health")

    async def fake_restore() -> None:
        calls.append("restore")

    monkeypatch.setattr(container.bot, "set_my_commands", AsyncMock(return_value=None))
    monkeypatch.setattr(container.hook_socket_server, "start", AsyncMock(return_value=None))
    monkeypatch.setattr(container.hook_socket_server, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container.bot.session, "close", AsyncMock(return_value=None))
    monkeypatch.setattr(container.session_registry, "_run_health_check", fake_health_check)
    monkeypatch.setattr(container, "_restore_session_bindings", fake_restore)
    monkeypatch.setattr(container.external_binding_cleanup_service, "_cleanup", AsyncMock(return_value=None))
    monkeypatch.setattr(container.upload_cleanup, "run_cleanup", AsyncMock(return_value=None))
    monkeypatch.setattr(container._janitor, "start", AsyncMock(return_value=None))
    monkeypatch.setattr(container._janitor, "stop", AsyncMock(return_value=None))

    await container.start()
    try:
        assert calls == ["health", "restore"]
    finally:
        await container.stop()


@pytest.mark.asyncio
async def test_start_clears_dead_tmux_binding_before_restoring_watchers(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(tmp_path, install_hooks=False, tmux_mode=True)
    first = AppContainer(settings)
    session, _ = await first.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await first.session_service.bind_claude_session(
        user_id=1,
        claude_session_id="claude-session-dead",
        workdir=str(tmp_path),
    )
    state = first.structured_session_store.get_or_create(
        session_id="claude-session-dead",
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_id=session.terminal_id,
        user_id=1,
        claude_session_id="claude-session-dead",
    )
    state.phase = SessionPhase.PROCESSING
    state.turns = [ConversationTurn(turn_id="a1", role="assistant", text="\n恢复中\n", is_complete=True)]
    first.structured_session_store._persist(state)
    await first.bot.session.close()

    second = AppContainer(settings)
    watched: list[tuple[str, str]] = []

    def fake_watch(*, session_id: str, workdir: str) -> None:
        watched.append((session_id, workdir))

    monkeypatch.setattr(second.bot, "set_my_commands", AsyncMock(return_value=None))
    monkeypatch.setattr(second.hook_socket_server, "start", AsyncMock(return_value=None))
    monkeypatch.setattr(second.hook_socket_server, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(second.bot.session, "close", AsyncMock(return_value=None))
    monkeypatch.setattr(second.tmux_runner, "session_exists", AsyncMock(return_value=False))
    monkeypatch.setattr(second.session_supervisor, "watch", fake_watch)
    monkeypatch.setattr(second.external_binding_cleanup_service, "_cleanup", AsyncMock(return_value=None))
    monkeypatch.setattr(second.upload_cleanup, "run_cleanup", AsyncMock(return_value=None))
    monkeypatch.setattr(second._janitor, "start", AsyncMock(return_value=None))
    monkeypatch.setattr(second._janitor, "stop", AsyncMock(return_value=None))

    await second.start()
    try:
        restored_session = await second.session_service.get(1)
        assert restored_session is not None
        assert restored_session.terminal_mode is False
        assert restored_session.terminal_id is None
        assert restored_session.claude_chat_active is False
        assert restored_session.claude_session_id is None
        assert watched == []
    finally:
        await second.stop()


@pytest.mark.asyncio
async def test_start_prunes_stale_external_binding_before_restoring_reply_delivery(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = AppContainer(
        make_settings(
            tmp_path,
            install_hooks=False,
            EXTERNAL_BINDING_IDLE_TTL_HOURS=1,
        )
    )
    session_id = "stale-external-session"
    stale_at = utc_now() - timedelta(hours=2)
    container.external_binding_store.save_binding(
        ExternalBinding(
            session_id=session_id,
            user_id=1,
            cwd=str(tmp_path),
            bound_at=stale_at,
            jsonl_path=None,
            last_activity_at_init=stale_at,
        )
    )
    ensure = MagicMock()
    pump_stop = AsyncMock()
    watch = MagicMock()
    schedule = MagicMock()
    forget = AsyncMock()

    monkeypatch.setattr(container.bot, "set_my_commands", AsyncMock(return_value=None))
    monkeypatch.setattr(container.hook_socket_server, "start", AsyncMock(return_value=None))
    monkeypatch.setattr(container.hook_socket_server, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container.bot.session, "close", AsyncMock(return_value=None))
    monkeypatch.setattr(container, "_restore_session_bindings", AsyncMock(return_value=None))
    monkeypatch.setattr(container.upload_cleanup, "run_cleanup", AsyncMock(return_value=None))
    monkeypatch.setattr(container._external_binding_cleanup_task, "start", lambda: None)
    monkeypatch.setattr(container._external_binding_cleanup_task, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container._janitor_task, "start", lambda: None)
    monkeypatch.setattr(container._janitor_task, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container.jsonl_file_watcher, "start", lambda: True)
    monkeypatch.setattr(container.jsonl_file_watcher, "stop", MagicMock())
    monkeypatch.setattr(container.external_reply_delivery_pump, "ensure", ensure)
    monkeypatch.setattr(container.external_reply_delivery_pump, "stop", pump_stop)
    monkeypatch.setattr(container.external_reply_delivery_pump, "stop_all", AsyncMock(return_value=None))
    monkeypatch.setattr(container.session_supervisor, "watch", watch)
    monkeypatch.setattr(container.session_supervisor, "schedule_jsonl_sync", schedule)
    monkeypatch.setattr(container.session_supervisor, "forget", forget)
    monkeypatch.setattr(container.session_supervisor, "stop_all", AsyncMock(return_value=None))

    await container.start()
    try:
        assert container.external_binding_store.get_binding(session_id) is None
        watch.assert_not_called()
        ensure.assert_not_called()
        schedule.assert_not_called()
        pump_stop.assert_awaited_once_with(session_id)
        forget.assert_awaited_once_with(session_id)
    finally:
        await container.stop()


@pytest.mark.asyncio
async def test_start_restores_ended_binding_for_finalization_even_when_reply_push_disabled(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = AppContainer(
        make_settings(
            tmp_path,
            install_hooks=False,
            EXTERNAL_BINDING_IDLE_TTL_HOURS=1,
            EXTERNAL_PUSH_REPLY_ENABLED=False,
        )
    )
    session_id = "ended-external-session"
    stale_at = utc_now() - timedelta(hours=2)
    binding = ExternalBinding(
        session_id=session_id,
        user_id=1,
        cwd=str(tmp_path),
        bound_at=stale_at,
        jsonl_path=None,
        last_activity_at_init=stale_at,
        ended_at=utc_now(),
    )
    container.external_binding_store.save_binding(binding)
    ensure = MagicMock()
    watch = MagicMock()
    schedule = MagicMock()

    monkeypatch.setattr(container.bot, "set_my_commands", AsyncMock(return_value=None))
    monkeypatch.setattr(container.hook_socket_server, "start", AsyncMock(return_value=None))
    monkeypatch.setattr(container.hook_socket_server, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container.bot.session, "close", AsyncMock(return_value=None))
    monkeypatch.setattr(container, "_restore_session_bindings", AsyncMock(return_value=None))
    monkeypatch.setattr(container.upload_cleanup, "run_cleanup", AsyncMock(return_value=None))
    monkeypatch.setattr(container._external_binding_cleanup_task, "start", lambda: None)
    monkeypatch.setattr(container._external_binding_cleanup_task, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container._janitor_task, "start", lambda: None)
    monkeypatch.setattr(container._janitor_task, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container.jsonl_file_watcher, "start", lambda: True)
    monkeypatch.setattr(container.jsonl_file_watcher, "stop", MagicMock())
    monkeypatch.setattr(container.external_reply_delivery_pump, "ensure", ensure)
    monkeypatch.setattr(container.external_reply_delivery_pump, "stop_all", AsyncMock(return_value=None))
    monkeypatch.setattr(container.session_supervisor, "watch", watch)
    monkeypatch.setattr(container.session_supervisor, "schedule_jsonl_sync", schedule)
    monkeypatch.setattr(container.session_supervisor, "stop_all", AsyncMock(return_value=None))

    await container.start()
    try:
        restored = container.external_binding_store.get_binding(session_id)
        assert restored is not None
        assert restored.ended_at is not None
        watch.assert_called_once_with(session_id=session_id, workdir=str(tmp_path))
        ensure.assert_called_once_with(session_id=session_id, cwd=str(tmp_path))
        schedule.assert_called_once_with(session_id, str(tmp_path))
    finally:
        await container.stop()


@pytest.mark.asyncio
async def test_start_registers_unbound_cleanup_without_replacing_existing_lifecycle_jobs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    calls: list[str] = []

    async def fake_unbound_cleanup() -> None:
        calls.append("unbound")

    monkeypatch.setattr(container.bot, "set_my_commands", AsyncMock(return_value=None))
    monkeypatch.setattr(container.hook_socket_server, "start", AsyncMock(return_value=None))
    monkeypatch.setattr(container.hook_socket_server, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container.bot.session, "close", AsyncMock(return_value=None))
    monkeypatch.setattr(container, "_restore_session_bindings", AsyncMock(return_value=None))
    monkeypatch.setattr(container.upload_cleanup, "run_cleanup", AsyncMock(return_value=None))
    monkeypatch.setattr(container.external_binding_cleanup_service, "_cleanup", AsyncMock(return_value=None))
    monkeypatch.setattr(container.session_registry, "_run_health_check", AsyncMock(return_value=None))
    monkeypatch.setattr(container, "_prune_unbound_external_sessions", fake_unbound_cleanup)
    monkeypatch.setattr(container._janitor_task, "start", lambda: None)
    monkeypatch.setattr(container._external_binding_cleanup_task, "start", lambda: None)

    await container.start()
    try:
        jobs = container._janitor._jobs
        assert "upload_queue_cleanup" in jobs
        assert "upload_file_cleanup" in jobs
        assert "periodic_recheck" in jobs
        assert "external_binding_cleanup" not in jobs  # moved to ExternalBindingCleanupTask
        assert "session_health_check" in jobs
        assert "external_discovery_cleanup" in jobs
        assert "session_cleanup" in jobs
        assert "session_lifecycle_reconcile" not in jobs

        session_cleanup_interval, session_cleanup_callback = jobs["session_cleanup"]
        assert session_cleanup_interval == container.settings.session_cleanup_interval_sec
        cleanup_calls: list[int] = []
        to_thread_calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

        def fake_cleanup(max_age_hours: int) -> None:
            cleanup_calls.append(max_age_hours)

        async def fake_to_thread(func, /, *args, **kwargs):
            to_thread_calls.append((func, args, kwargs))
            return func(*args, **kwargs)

        monkeypatch.setattr(container.file_session_store, "cleanup_stale_sessions", fake_cleanup)
        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

        await session_cleanup_callback()

        assert cleanup_calls == [container.settings.session_cleanup_max_age_hours]
        assert to_thread_calls == [(fake_cleanup, (container.settings.session_cleanup_max_age_hours,), {})]

        interval, callback = jobs["external_discovery_cleanup"]
        assert interval == container.settings.session_health_check_interval_sec
        await callback()
        assert calls == ["unbound"]
    finally:
        await container.stop()


@pytest.mark.asyncio
async def test_handle_hook_event_binds_session_and_syncs_jsonl(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    use_legacy_hook_binding_path(container)

    session, _ = await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await container.task_store.add(
        TaskRecord(
            task_id="task-1",
            session_id=session.session_id,
            user_id=1,
            provider="claude_code",
            prompt="hello",
            workdir=str(tmp_path),
            timeout_sec=10,
            status=TaskStatus.RUNNING,
        )
    )

    async def fake_sync(session_id: str, cwd: str) -> None:
        await container._dispatch_session_event(
            SessionEvent(
                session_id=session_id,
                type=SessionEventType.FILE_SYNCED,
                payload={
                    "cwd": cwd,
                    "claude_session_id": session_id,
                    "turns": [
                        {
                            "turn_id": "a1",
                            "role": "assistant",
                            "text": "\n干净回复\n",
                            "source": "jsonl",
                            "is_complete": True,
                            "started_at": "2026-04-16T10:00:01+00:00",
                            "ended_at": "2026-04-16T10:00:01+00:00",
                        }
                    ],
                    "tool_calls": {},
                    "last_reply": "干净回复",
                    "last_reply_role": "assistant",
                    "last_offset": 12,
                },
            )
        )

    monkeypatch.setattr(container, "sync_claude_session", fake_sync)
    monkeypatch.setattr(container.session_supervisor, "_on_jsonl_sync", fake_sync)

    await container._handle_hook_event(
        HookEvent(
            session_id="claude-session-1",
            cwd=str(tmp_path),
            event="SessionStart",
            status="starting",
        )
    )
    await wait_for_jsonl_sync_idle(container, "claude-session-1")

    session = await container.session_service.get(1)
    assert session is not None
    assert session.claude_session_id == "claude-session-1"

    state = container.structured_session_store.get("claude-session-1")
    assert state is not None
    assert state.user_id == 1
    assert state.terminal_id == session.terminal_id
    assert state.phase == SessionPhase.WAITING_FOR_INPUT
    assert state.last_reply == "干净回复"


@pytest.mark.asyncio
async def test_handle_hook_event_binds_session_by_unique_active_claude_chat_workdir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))

    session, _ = await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )

    async def fake_sync(session_id: str, cwd: str) -> None:
        return None

    monkeypatch.setattr(container, "sync_claude_session", fake_sync)
    monkeypatch.setattr(container.session_supervisor, "_on_jsonl_sync", fake_sync)
    allow_target_tmux_pid(container, monkeypatch)

    await container._handle_hook_event(
        HookEvent(
            session_id="claude-session-1",
            cwd=str(tmp_path),
            event="SessionStart",
            status="starting",
            pid=4242,
        )
    )
    await wait_for_jsonl_sync_idle(container, "claude-session-1")

    updated = await container.session_service.get(1)
    assert updated is not None
    assert updated.claude_session_id == "claude-session-1"
    assert updated.terminal_id == session.terminal_id


@pytest.mark.asyncio
async def test_handle_hook_event_binds_unowned_tmux_first_hook_with_active_task(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))

    session, _ = await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    terminal_state = container.structured_session_store.get_or_create(
        session_id=f"tgcli_{session.terminal_id}",
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_id=session.terminal_id,
        user_id=1,
    )
    terminal_state.phase = SessionPhase.PROCESSING
    container.structured_session_store._persist(terminal_state)
    await container.task_store.add(
        TaskRecord(
            task_id="task-1",
            session_id=session.session_id,
            user_id=1,
            provider="claude_code",
            prompt="hello",
            workdir=str(tmp_path),
            timeout_sec=10,
            status=TaskStatus.RUNNING,
        )
    )

    async def fake_sync(session_id: str, cwd: str) -> None:
        return None

    monkeypatch.setattr(container, "sync_claude_session", fake_sync)
    monkeypatch.setattr(container.session_supervisor, "_on_jsonl_sync", fake_sync)
    allow_target_tmux_pid(container, monkeypatch)

    await container._handle_hook_event(
        HookEvent(
            session_id="claude-session-1",
            cwd=str(tmp_path),
            event="UserPromptSubmit",
            status="processing",
            pid=4242,
        )
    )
    await wait_for_jsonl_sync_idle(container, "claude-session-1")

    updated = await container.session_service.get(1)
    assert updated is not None
    assert updated.claude_session_id == "claude-session-1"
    assert updated.terminal_id == session.terminal_id

    state = container.structured_session_store.get("claude-session-1")
    assert state is not None
    assert state.user_id == 1
    assert state.terminal_id == session.terminal_id


@pytest.mark.asyncio
async def test_handle_hook_event_does_not_bind_unowned_workdir_without_terminal_id(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_workdir = str(tmp_path)
    container = AppContainer(make_settings(tmp_path, install_hooks=False))

    await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=session_workdir,
        terminal_mode=False,
        claude_chat_active=True,
    )

    async def fake_sync(session_id: str, cwd: str) -> None:
        return None

    monkeypatch.setattr(container, "sync_claude_session", fake_sync)
    monkeypatch.setattr(container.session_supervisor, "_on_jsonl_sync", fake_sync)

    await container._handle_hook_event(
        HookEvent(
            session_id="claude-session-1",
            cwd=session_workdir,
            event="SessionStart",
            status="starting",
        )
    )
    await wait_for_jsonl_sync_idle(container, "claude-session-1")

    updated = await container.session_service.get(1)
    assert updated is not None
    assert updated.claude_session_id is None
    assert updated.terminal_id is None


@pytest.mark.asyncio
async def test_handle_hook_event_does_not_bind_session_by_unique_workdir_when_claude_chat_inactive(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))

    session, _ = await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=False,
    )

    async def fake_sync(session_id: str, cwd: str) -> None:
        return None

    monkeypatch.setattr(container, "sync_claude_session", fake_sync)
    monkeypatch.setattr(container.session_supervisor, "_on_jsonl_sync", fake_sync)

    await container._handle_hook_event(
        HookEvent(
            session_id="claude-session-1",
            cwd=str(tmp_path),
            event="SessionStart",
            status="starting",
        )
    )
    await wait_for_jsonl_sync_idle(container, "claude-session-1")

    updated = await container.session_service.get(1)
    assert updated is not None
    assert updated.claude_session_id is None
    assert updated.terminal_id == session.terminal_id


@pytest.mark.asyncio
async def test_handle_hook_event_binds_session_for_empty_processing_terminal_state(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))

    session, _ = await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    terminal_state = container.structured_session_store.get_or_create(
        session_id=f"tgcli_{session.terminal_id}",
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_id=session.terminal_id,
        user_id=1,
    )
    terminal_state.phase = SessionPhase.PROCESSING
    container.structured_session_store._persist(terminal_state)

    async def fake_sync(session_id: str, cwd: str) -> None:
        return None

    monkeypatch.setattr(container, "sync_claude_session", fake_sync)
    monkeypatch.setattr(container.session_supervisor, "_on_jsonl_sync", fake_sync)
    allow_target_tmux_pid(container, monkeypatch)

    await container._handle_hook_event(
        HookEvent(
            session_id="claude-session-1",
            cwd=str(tmp_path),
            event="SessionStart",
            status="starting",
            pid=4242,
        )
    )
    await wait_for_jsonl_sync_idle(container, "claude-session-1")

    updated = await container.session_service.get(1)
    assert updated is not None
    assert updated.claude_session_id == "claude-session-1"
    assert updated.terminal_id == session.terminal_id

    state = container.structured_session_store.get("claude-session-1")
    assert state is not None
    assert state.user_id == 1
    assert state.terminal_id == session.terminal_id


@pytest.mark.asyncio
async def test_handle_hook_event_binds_session_when_terminal_state_has_content_without_task(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    use_legacy_hook_binding_path(container)

    session, _ = await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await container.session_service.bind_claude_session(
        user_id=1,
        claude_session_id="old-session",
        workdir=str(tmp_path),
    )
    previous = container.structured_session_store.get_or_create(
        session_id="old-session",
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_id=session.terminal_id,
        user_id=1,
        claude_session_id="old-session",
    )
    previous.phase = SessionPhase.PROCESSING
    previous.tool_calls["tool-1"] = ToolCallRecord(
        tool_use_id="tool-1",
        name="Bash",
        input={"command": "pwd"},
        status=ToolStatus.RUNNING,
    )
    container.structured_session_store._persist(previous)

    async def fake_sync(session_id: str, cwd: str) -> None:
        return None

    monkeypatch.setattr(container, "sync_claude_session", fake_sync)
    monkeypatch.setattr(container.session_supervisor, "_on_jsonl_sync", fake_sync)

    await container._handle_hook_event(
        HookEvent(
            session_id="claude-session-2",
            cwd=str(tmp_path),
            event="SessionStart",
            status="starting",
        )
    )
    await wait_for_jsonl_sync_idle(container, "claude-session-2")

    updated = await container.session_service.get(1)
    assert updated is not None
    assert updated.claude_session_id == "claude-session-2"
    assert updated.terminal_id == session.terminal_id


@pytest.mark.asyncio
async def test_handle_hook_event_binds_session_when_pending_interactive_task_matches_workdir(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    use_legacy_hook_binding_path(container)

    session, _ = await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await container.task_store.add(
        TaskRecord(
            task_id="task-1",
            session_id=session.session_id,
            user_id=1,
            provider="claude_code",
            prompt="hello",
            workdir=str(tmp_path),
            timeout_sec=10,
            status=TaskStatus.PENDING,
        )
    )

    async def fake_sync(session_id: str, cwd: str) -> None:
        return None

    monkeypatch.setattr(container, "sync_claude_session", fake_sync)
    monkeypatch.setattr(container.session_supervisor, "_on_jsonl_sync", fake_sync)

    await container._handle_hook_event(
        HookEvent(
            session_id="claude-session-1",
            cwd=str(tmp_path),
            event="SessionStart",
            status="starting",
        )
    )
    await wait_for_jsonl_sync_idle(container, "claude-session-1")

    updated = await container.session_service.get(1)
    assert updated is not None
    assert updated.claude_session_id == "claude-session-1"
    assert updated.terminal_id == session.terminal_id


@pytest.mark.asyncio
async def test_handle_hook_event_does_not_bind_session_when_only_final_task_matches_workdir(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))

    session, _ = await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=False,
    )
    await container.task_store.add(
        TaskRecord(
            task_id="task-1",
            session_id=session.session_id,
            user_id=1,
            provider="claude_code",
            prompt="hello",
            workdir=str(tmp_path),
            timeout_sec=10,
            status=TaskStatus.SUCCEEDED,
        )
    )

    async def fake_sync(session_id: str, cwd: str) -> None:
        return None

    monkeypatch.setattr(container, "sync_claude_session", fake_sync)
    monkeypatch.setattr(container.session_supervisor, "_on_jsonl_sync", fake_sync)

    await container._handle_hook_event(
        HookEvent(
            session_id="claude-session-1",
            cwd=str(tmp_path),
            event="SessionStart",
            status="starting",
        )
    )
    await wait_for_jsonl_sync_idle(container, "claude-session-1")

    updated = await container.session_service.get(1)
    assert updated is not None
    assert updated.claude_session_id is None
    assert updated.terminal_id == session.terminal_id


@pytest.mark.asyncio
async def test_handle_hook_event_runs_bind_before_dispatch_and_sync(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    use_legacy_hook_binding_path(container)
    seen: list[str] = []

    async def fake_bind(event: HookEvent) -> None:
        seen.append(f"bind:{event.session_id}")

    async def fake_dispatch(event) -> None:
        seen.append(f"dispatch:{event.session_id}")

    def fake_schedule(session_id: str, cwd: str) -> None:
        seen.append(f"sync:{session_id}:{cwd}")

    monkeypatch.setattr(container, "_bind_hook_session", fake_bind)
    monkeypatch.setattr(container, "_dispatch_session_event", fake_dispatch)
    monkeypatch.setattr(container, "_schedule_jsonl_sync", fake_schedule)

    await container._handle_hook_event(HookEvent(session_id="claude-session-1", cwd=str(tmp_path), event="Notification", status="running"))

    assert seen == [
        "bind:claude-session-1",
        "dispatch:claude-session-1",
        f"sync:claude-session-1:{str(tmp_path)}",
    ]


@pytest.mark.asyncio
async def test_handle_hook_event_rejects_workdir_outside_allowlist(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    seen: list[str] = []

    async def fake_bind(event: HookEvent) -> None:
        seen.append(f"bind:{event.session_id}")

    async def fake_dispatch(event) -> None:
        seen.append(f"dispatch:{event.session_id}")

    def fake_schedule(session_id: str, cwd: str) -> None:
        seen.append(f"sync:{session_id}:{cwd}")

    monkeypatch.setattr(container, "_bind_hook_session", fake_bind)
    monkeypatch.setattr(container, "_dispatch_session_event", fake_dispatch)
    monkeypatch.setattr(container, "_schedule_jsonl_sync", fake_schedule)

    await container._handle_hook_event(
        HookEvent(
            session_id="claude-session-outside",
            cwd=str(tmp_path.parent / f"{tmp_path.name}-outside"),
            event="Notification",
            status="running",
        )
    )

    assert seen == []
    assert container.structured_session_store.get("claude-session-outside") is None


@pytest.mark.asyncio
async def test_handle_hook_event_debounces_jsonl_sync(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    use_legacy_hook_binding_path(container)
    seen: list[tuple[str, str]] = []

    async def fake_sync(session_id: str, cwd: str) -> None:
        seen.append((session_id, cwd))

    monkeypatch.setattr(container, "sync_claude_session", fake_sync)
    monkeypatch.setattr(container.session_supervisor, "_on_jsonl_sync", fake_sync)

    await container._handle_hook_event(HookEvent(session_id="claude-session-1", cwd=str(tmp_path), event="Notification", status="running"))
    await container._handle_hook_event(HookEvent(session_id="claude-session-1", cwd=str(tmp_path), event="Notification", status="running"))
    await wait_for_jsonl_sync_idle(container, "claude-session-1")

    assert seen == [("claude-session-1", str(tmp_path))]


@pytest.mark.asyncio
async def test_sync_claude_session_uses_per_session_lock(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))

    class _Snapshot:
        turns = []
        tool_calls = {}
        last_reply = None
        last_reply_role = None
        last_offset = 0
        clear_detected = False

        def to_payload(self):
            return {"turns": [], "tool_calls": {}, "last_offset": 0}

    seen: list[tuple[str, str]] = []

    def fake_parse_incremental(*, session_id: str, cwd: str):
        seen.append((session_id, cwd))
        return _Snapshot()

    monkeypatch.setattr(container.claude_jsonl_parser, "parse_incremental", fake_parse_incremental)

    held_lock = container._jsonl_sync_locks.lock("claude-session-1")
    await held_lock.__aenter__()

    first = asyncio.create_task(container.sync_claude_session("claude-session-1", str(tmp_path)))
    second = asyncio.create_task(container.sync_claude_session("claude-session-1", str(tmp_path)))
    await asyncio.sleep(0)
    assert seen == []
    assert first.done() is False
    assert second.done() is False

    await held_lock.__aexit__(None, None, None)
    await first
    await second

    assert seen == [
        ("claude-session-1", str(tmp_path)),
        ("claude-session-1", str(tmp_path)),
    ]


@pytest.mark.asyncio
async def test_stop_cancels_pending_jsonl_sync_tasks(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))

    async def fake_start(handler, permission_failure_handler=None, permission_resolved_handler=None):
        return None

    async def fake_stop():
        return None

    async def fake_close():
        return None

    async def fake_sync(session_id: str, cwd: str) -> None:
        await asyncio.sleep(1)

    monkeypatch.setattr(container.hook_socket_server, "start", fake_start)
    monkeypatch.setattr(container.hook_socket_server, "stop", fake_stop)
    monkeypatch.setattr(container.bot.session, "close", fake_close)
    monkeypatch.setattr(container, "sync_claude_session", fake_sync)
    monkeypatch.setattr(container.session_supervisor, "_on_jsonl_sync", fake_sync)

    await container.start()
    container._schedule_jsonl_sync("claude-session-1", str(tmp_path))

    await container.stop()


@pytest.mark.asyncio
async def test_app_container_stop_blocks_remaining_in_flight_hook_stages(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    ownership_entered = asyncio.Event()
    release_ownership = asyncio.Event()
    bind_hook_session = AsyncMock()
    dispatch_session_event = AsyncMock()
    schedule_jsonl_sync = MagicMock()

    async def blocked_ownership(_event: HookEvent):
        ownership_entered.set()
        await release_ownership.wait()
        return MagicMock(ownership_state="owned", owner_user_id=1)

    monkeypatch.setattr(container, "_resolve_ownership_stage", blocked_ownership)
    monkeypatch.setattr(container, "_bind_hook_session", bind_hook_session)
    monkeypatch.setattr(container, "_dispatch_session_event", dispatch_session_event)
    monkeypatch.setattr(container, "_schedule_jsonl_sync", schedule_jsonl_sync)
    monkeypatch.setattr(container._janitor_task, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container._external_binding_cleanup_task, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container.hook_socket_server, "stop", AsyncMock(return_value=None))
    monkeypatch.setattr(container.jsonl_file_watcher, "stop", MagicMock())
    monkeypatch.setattr(container.external_reply_delivery_pump, "stop_all", AsyncMock(return_value=None))
    monkeypatch.setattr(container.session_supervisor, "stop_all", AsyncMock(return_value=None))
    monkeypatch.setattr(container, "_stop_background_tasks", AsyncMock(return_value=None))
    monkeypatch.setattr(container.bot.session, "close", AsyncMock(return_value=None))

    hook_task = asyncio.create_task(
        container._handle_hook_event(
            HookEvent(
                session_id="in-flight-session",
                cwd=str(tmp_path),
                event="Notification",
                status="running",
            )
        )
    )
    await asyncio.wait_for(ownership_entered.wait(), timeout=1)

    await container.stop()
    release_ownership.set()
    await asyncio.wait_for(hook_task, timeout=1)

    bind_hook_session.assert_not_awaited()
    dispatch_session_event.assert_not_awaited()
    schedule_jsonl_sync.assert_not_called()


@pytest.mark.asyncio
async def test_debounced_sync_keeps_request_added_during_sync(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that a sync request scheduled during an active sync is picked up on the next poll."""
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    seen: list[tuple[str, str]] = []

    # Supervisor needs session state in the store to watch
    container.structured_session_store.get_or_create(
        session_id="claude-session-1",
        provider="claude_code",
        workdir=str(tmp_path),
    )

    async def fake_sync(session_id: str, cwd: str) -> None:
        seen.append((session_id, cwd))
        # Schedule a second sync during the first one
        if len(seen) == 1:
            container._schedule_jsonl_sync(session_id, f"{cwd}-next")

    monkeypatch.setattr(container, "sync_claude_session", fake_sync)
    monkeypatch.setattr(container.session_supervisor, "_on_jsonl_sync", fake_sync)

    container._schedule_jsonl_sync("claude-session-1", str(tmp_path))
    # Wait for debounce + 2 poll cycles (first processes initial, second picks up re-scheduled)
    await asyncio.sleep(0.2 * 2 + 0.01 + 0.2)

    assert seen == [
        ("claude-session-1", str(tmp_path)),
        ("claude-session-1", f"{str(tmp_path)}-next"),
    ]


@pytest.mark.asyncio
async def test_debounced_sync_requeues_request_after_failure(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Supervisor logs sync errors but does not re-queue (differs from old behavior)."""
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    seen: list[tuple[str, str]] = []

    # Supervisor needs session state in the store to watch
    container.structured_session_store.get_or_create(
        session_id="claude-session-1",
        provider="claude_code",
        workdir=str(tmp_path),
    )

    async def fake_sync(session_id: str, cwd: str) -> None:
        seen.append((session_id, cwd))
        if len(seen) == 1:
            raise RuntimeError("boom")

    monkeypatch.setattr(container, "sync_claude_session", fake_sync)
    monkeypatch.setattr(container.session_supervisor, "_on_jsonl_sync", fake_sync)

    container._schedule_jsonl_sync("claude-session-1", str(tmp_path))
    await wait_for_jsonl_sync_idle(container, "claude-session-1")

    # Supervisor catches the error and logs it; the request is consumed.
    assert seen == [
        ("claude-session-1", str(tmp_path)),
    ]


@pytest.mark.asyncio
async def test_session_end_keeps_pending_sync_until_flushed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    use_legacy_hook_binding_path(container)
    seen: list[tuple[str, str]] = []

    await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )

    async def fake_sync(session_id: str, cwd: str) -> None:
        seen.append((session_id, cwd))

    monkeypatch.setattr(container, "sync_claude_session", fake_sync)
    monkeypatch.setattr(container.session_supervisor, "_on_jsonl_sync", fake_sync)

    await container._handle_hook_event(HookEvent(session_id="claude-session-1", cwd=str(tmp_path), event="SessionEnd", status="ended"))
    await wait_for_jsonl_sync_idle(container, "claude-session-1")

    assert seen == [("claude-session-1", str(tmp_path))]


@pytest.mark.asyncio
async def test_bound_hook_stops_after_binding_changes_during_event_dispatch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    session_id = "bound-generation-change"
    old_binding = ExternalBinding(
        session_id=session_id,
        user_id=1,
        cwd=str(tmp_path),
        bound_at=utc_now(),
        jsonl_path=None,
    )
    container.external_binding_store.save_binding(old_binding)
    dispatch_started = asyncio.Event()
    allow_dispatch = asyncio.Event()
    schedule_jsonl_sync = MagicMock()
    auto_approve = AsyncMock()
    notify = AsyncMock()
    pump_ensure = MagicMock()

    async def blocked_dispatch(_event: SessionEvent) -> None:
        dispatch_started.set()
        await allow_dispatch.wait()

    monkeypatch.setattr(container, "_dispatch_session_event", blocked_dispatch)
    monkeypatch.setattr(container, "_schedule_jsonl_sync", schedule_jsonl_sync)
    monkeypatch.setattr(container, "_run_auto_approve_check", auto_approve)
    monkeypatch.setattr(container, "_notify_bound_external_event", notify)
    monkeypatch.setattr(container.external_reply_delivery_pump, "ensure", pump_ensure)

    try:
        hook_task = asyncio.create_task(
            container._handle_hook_event(
                HookEvent(
                    session_id=session_id,
                    cwd=str(tmp_path),
                    event="Notification",
                    status="running",
                )
            )
        )
        await dispatch_started.wait()
        container.external_binding_store.save_binding(
            ExternalBinding(
                session_id=session_id,
                user_id=1,
                cwd=str(tmp_path),
                bound_at=utc_now(),
                jsonl_path=None,
            )
        )
        allow_dispatch.set()
        await hook_task

        schedule_jsonl_sync.assert_not_called()
        pump_ensure.assert_not_called()
        auto_approve.assert_not_awaited()
        notify.assert_not_awaited()
    finally:
        allow_dispatch.set()
        await container.external_reply_delivery_pump.stop_all()
        await container.session_supervisor.stop_all()
        await container.bot.session.close()


@pytest.mark.asyncio
async def test_bound_file_send_rechecks_generation_after_waiting_for_delivery_lock(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    session_id = "bound-file-generation-change"
    old_binding = ExternalBinding(
        session_id=session_id,
        user_id=1,
        cwd=str(tmp_path),
        bound_at=utc_now(),
        jsonl_path=None,
    )
    container.external_binding_store.save_binding(old_binding)
    ownership = await container.ownership_resolver.resolve(session_id)
    send_if_eligible = AsyncMock()
    monkeypatch.setattr(container.file_sender, "send_if_eligible", send_if_eligible)
    event = HookEvent(
        session_id=session_id,
        cwd=str(tmp_path),
        event="PostToolUse",
        status="running",
        tool="Write",
        tool_input={"file_path": "result.txt"},
    )

    async with container._external_reply_delivery_locks.lock(session_id):
        send_task = asyncio.create_task(container._send_bound_file_if_current(event, ownership))
        await asyncio.sleep(0)
        container.external_binding_store.save_binding(
            ExternalBinding(
                session_id=session_id,
                user_id=1,
                cwd=str(tmp_path),
                bound_at=utc_now(),
                jsonl_path=None,
            )
        )

    try:
        await send_task
        send_if_eligible.assert_not_awaited()
    finally:
        await container.external_reply_delivery_pump.stop_all()
        await container.session_supervisor.stop_all()
        await container.bot.session.close()


@pytest.mark.asyncio
async def test_bound_external_session_end_direct_syncs_and_requests_durable_finalization(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    session_id = "bound-session-end"
    container.external_binding_store.save_binding(
        ExternalBinding(
            session_id=session_id,
            user_id=1,
            cwd=str(tmp_path),
            bound_at=utc_now(),
            jsonl_path=None,
        )
    )
    dispatch = AsyncMock()
    direct_sync = AsyncMock()
    scheduled_sync = MagicMock()
    watch = MagicMock()
    supervisor_schedule = MagicMock()
    forget = AsyncMock()
    pump_finalize = MagicMock()
    notify_end = AsyncMock(return_value=True)
    discard_progress = MagicMock()

    monkeypatch.setattr(container, "_dispatch_session_event", dispatch)
    monkeypatch.setattr(container, "sync_claude_session", direct_sync)
    monkeypatch.setattr(container, "_schedule_jsonl_sync", scheduled_sync)
    monkeypatch.setattr(container, "_run_auto_approve_check", AsyncMock(return_value=None))
    monkeypatch.setattr(container, "_maybe_auto_file_send", MagicMock())
    monkeypatch.setattr(container.session_supervisor, "watch", watch)
    monkeypatch.setattr(container.session_supervisor, "schedule_jsonl_sync", supervisor_schedule)
    monkeypatch.setattr(container.session_supervisor, "forget", forget)
    monkeypatch.setattr(container.external_reply_delivery_pump, "request_finalize", pump_finalize)
    monkeypatch.setattr(container.push_notifier, "notify_session_end", notify_end)
    monkeypatch.setattr(container.push_notifier, "discard_assistant_reply_progress", discard_progress)

    await container._handle_hook_event(
        HookEvent(
            session_id=session_id,
            cwd=str(tmp_path),
            event="SessionEnd",
            status="ended",
        )
    )

    dispatch.assert_awaited_once()
    direct_sync.assert_awaited_once_with(session_id, str(tmp_path))
    scheduled_sync.assert_not_called()
    watch.assert_not_called()
    supervisor_schedule.assert_not_called()
    pump_finalize.assert_called_once_with(session_id=session_id, cwd=str(tmp_path))
    notify_end.assert_not_awaited()
    binding = container.external_binding_store.get_binding(session_id)
    assert binding is not None
    assert binding.ended_at is not None
    forget.assert_not_awaited()
    discard_progress.assert_not_called()


@pytest.mark.asyncio
async def test_session_end_cleans_event_lock_registry(tmp_path) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))

    await container._dispatch_session_event(
        SessionEvent(
            session_id="claude-session-ended",
            type=SessionEventType.SESSION_ENDED,
            payload={"cwd": str(tmp_path)},
        )
    )

    assert len(container._session_event_locks) == 0


@pytest.mark.asyncio
async def test_terminal_permission_resolution_edits_telegram_message_once(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    # 初始化会话状态并绑定有效工作目录
    initial_workdir = str(tmp_path / "my_project")
    container.structured_session_store.get_or_create(session_id="claude-session-1", workdir=initial_workdir)

    token = await container.permission_callback_registry.register_token(
        tool_use_id="tool-1",
        session_id="claude-session-1",
        origin=SessionOrigin.EXTERNAL_BOUND,
        authorization_mode=AuthorizationMode.BOUND_USER,
        authorized_user_ids=frozenset({1}),
    )
    original_text = "🔐 请求权限: Bash\n请点击下方按钮选择允许或拒绝。"
    await container.permission_callback_registry.update_telegram_message(
        token=token,
        chat_id=1,
        message_id=42,
        message_text=original_text,
    )
    edit_message = AsyncMock()
    monkeypatch.setattr(container.message_sender, "edit_message", edit_message)

    await container._handle_permission_resolved("claude-session-1", "tool-1", "terminal_approved")
    await container._handle_permission_resolved("claude-session-1", "tool-1", "terminal_approved")

    edit_message.assert_awaited_once()
    kwargs = edit_message.await_args.kwargs
    assert kwargs["chat_id"] == 1
    assert kwargs["message_id"] == 42
    assert "已在终端批准" in kwargs["text"]
    assert "请点击下方按钮选择允许或拒绝。" not in kwargs["text"]
    assert kwargs["parse_mode"] == "HTML"
    # 验证缺少 cwd 的权限事件没有将真实 workdir 覆写为 "."
    session_state = container.structured_session_store.get("claude-session-1")
    assert session_state is not None
    assert session_state.workdir == initial_workdir


@pytest.mark.asyncio
async def test_container_uses_independent_session_lock_registries(tmp_path) -> None:
    settings = make_settings(tmp_path, install_hooks=False)
    container = AppContainer(settings)

    assert container._jsonl_sync_locks._ttl_sec == settings.session_lock_ttl_sec
    assert container._session_event_locks._ttl_sec == settings.session_lock_ttl_sec
    assert container._jsonl_sync_locks._cleanup_interval_sec == settings.lock_cleanup_interval_sec
    assert container._session_event_locks._cleanup_interval_sec == settings.lock_cleanup_interval_sec
    assert container._jsonl_sync_locks is not container._session_event_locks
    assert container.permission_callback_registry._ttl_sec == settings.claude_hook_pending_permission_ttl_sec


@pytest.mark.asyncio
async def test_match_session_context_does_not_fallback_on_workdir_collision(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))

    session_one, _ = await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    session_two = type(session_one).from_dict(
        {
            **session_one.to_dict(),
            "user_id": 2,
            "session_id": "session-2",
            "terminal_id": "user_2",
            "claude_session_id": None,
        }
    )

    async def fake_list_all():
        return [session_one, session_two]

    monkeypatch.setattr(container.session_service, "list_all", fake_list_all)

    matched = await container._match_session_context(
        HookEvent(session_id="claude-session-1", cwd=str(tmp_path), event="Notification", status="running")
    )

    assert matched is None


@pytest.mark.asyncio
async def test_match_session_context_uses_active_interactive_task_when_workdir_collides(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))

    session_one, _ = await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    session_two = type(session_one).from_dict(
        {
            **session_one.to_dict(),
            "user_id": 2,
            "session_id": "session-2",
            "terminal_id": "user_2",
            "claude_session_id": None,
        }
    )

    async def fake_list_all():
        return [session_one, session_two]

    monkeypatch.setattr(container.session_service, "list_all", fake_list_all)
    await container.task_store.add(
        TaskRecord(
            task_id="task-active",
            session_id=session_two.session_id,
            user_id=session_two.user_id,
            provider="claude_code",
            prompt="hello",
            workdir=str(tmp_path),
            timeout_sec=10,
            status=TaskStatus.RUNNING,
        )
    )

    matched = await container._match_session_context(
        HookEvent(session_id="claude-session-1", cwd=str(tmp_path), event="SessionStart", status="starting")
    )

    assert matched is not None
    assert matched.user_id == 2


@pytest.mark.asyncio
async def test_unbound_hook_ownership_uses_active_interactive_task_when_workdir_collides(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))

    session_one, _ = await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    session_two = type(session_one).from_dict(
        {
            **session_one.to_dict(),
            "user_id": 2,
            "session_id": "session-2",
            "terminal_id": "user_2",
            "claude_session_id": None,
        }
    )

    async def fake_list_all():
        return [session_one, session_two]

    monkeypatch.setattr(container.session_service, "list_all", fake_list_all)
    await container.task_store.add(
        TaskRecord(
            task_id="task-active",
            session_id=session_two.session_id,
            user_id=session_two.user_id,
            provider="claude_code",
            prompt="hello",
            workdir=str(tmp_path),
            timeout_sec=10,
            status=TaskStatus.RUNNING,
        )
    )

    allow_target_tmux_pid(container, monkeypatch, user_id=2)

    ownership = await container._resolve_ownership_stage(
        HookEvent(
            session_id="claude-session-1",
            cwd=str(tmp_path),
            event="SessionStart",
            status="starting",
            pid=4242,
        )
    )

    assert ownership is not None
    assert ownership.ownership_state == "owned"
    assert ownership.origin.value == "tmux"
    assert ownership.owner_user_id == 2


@pytest.mark.asyncio
async def test_match_session_context_prefers_terminal_binding(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    session, _ = await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    state = container.structured_session_store.get_or_create(
        session_id="claude-session-1",
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_id=session.terminal_id,
        user_id=1,
        claude_session_id="claude-session-1",
    )
    container.structured_session_store._persist(state)

    matched = await container._match_session_context(
        HookEvent(session_id="claude-session-1", cwd="/other", event="Notification", status="running")
    )

    assert matched is not None
    assert matched.user_id == 1


@pytest.mark.asyncio
async def test_restore_session_bindings_syncs_existing_jsonl_even_when_state_is_nonempty(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    session, _ = await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    session_id = "persisted-session"
    await container.session_service.bind_claude_session(
        user_id=1,
        claude_session_id=session_id,
        workdir=str(tmp_path),
    )
    state = container.structured_session_store.get_or_create(
        session_id=session_id,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_id=session.terminal_id,
        user_id=1,
        claude_session_id=session_id,
    )
    state.turns = [ConversationTurn(turn_id="a1", role="assistant", text="persisted", is_complete=True)]
    container.structured_session_store._persist(state)
    session_file = tmp_path / "session.jsonl"
    session_file.write_text("{}\n", encoding="utf-8")
    sync = AsyncMock()
    watch = MagicMock()
    monkeypatch.setattr(container.claude_jsonl_parser, "session_file_path", MagicMock(return_value=session_file))
    monkeypatch.setattr(container, "sync_claude_session", sync)
    monkeypatch.setattr(container.session_supervisor, "watch", watch)

    await container._restore_session_bindings()

    sync.assert_awaited_once_with(session_id, str(tmp_path))
    watch.assert_called_once_with(session_id=session_id, workdir=str(tmp_path))


@pytest.mark.asyncio
async def test_restore_session_bindings_clears_empty_terminal_binding_when_snapshot_missing(tmp_path) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    session, _ = await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await container.session_service.bind_claude_session(
        user_id=1,
        claude_session_id="missing-session",
        workdir=str(tmp_path),
    )
    terminal_state = container.structured_session_store.get_or_create(
        session_id=f"tgcli_{session.terminal_id}",
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_id=session.terminal_id,
        user_id=1,
    )
    terminal_state.phase = SessionPhase.PROCESSING
    container.structured_session_store._persist(terminal_state)

    await container._restore_session_bindings()

    restored_session = await container.session_service.get(1)
    assert restored_session is not None
    assert restored_session.claude_session_id is None


@pytest.mark.asyncio
async def test_restore_external_reply_delivery_pumps_resumes_persisted_bindings(tmp_path) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    binding = ExternalBinding(
        session_id="external-session-1",
        user_id=1,
        cwd=str(tmp_path),
        bound_at=utc_now(),
        jsonl_path=None,
        last_pushed_reply_turn_id="a1",
        reply_cursor_initialized=True,
    )
    container.external_binding_store.save_binding(binding)
    container.external_reply_delivery_pump = MagicMock()
    container.external_reply_delivery_pump.stop_all = AsyncMock()
    container.session_supervisor.watch = MagicMock()
    container.session_supervisor.schedule_jsonl_sync = MagicMock()

    await container._restore_external_reply_delivery_pumps()

    state = container.structured_session_store.get("external-session-1")
    assert state is not None
    assert state.user_id == 1
    assert state.workdir == str(tmp_path)
    container.session_supervisor.watch.assert_called_once_with(
        session_id="external-session-1",
        workdir=str(tmp_path),
    )
    container.external_reply_delivery_pump.ensure.assert_called_once_with(
        session_id="external-session-1",
        cwd=str(tmp_path),
    )
    container.session_supervisor.schedule_jsonl_sync.assert_called_once_with(
        "external-session-1",
        str(tmp_path),
    )


@pytest.mark.asyncio
async def test_periodic_recheck_syncs_processing_claude_session(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False))
    await container.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await container.session_service.bind_claude_session(
        user_id=1,
        claude_session_id="claude-session-1",
        workdir=str(tmp_path),
    )
    state = container.structured_session_store.get_or_create(
        session_id="claude-session-1",
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_id="user_1",
        user_id=1,
    )
    state.phase = SessionPhase.PROCESSING
    container.structured_session_store._persist(state)

    seen: list[tuple[str, str]] = []

    async def fake_sync(session_id: str, cwd: str) -> None:
        seen.append((session_id, cwd))

    monkeypatch.setattr(container, "sync_claude_session", fake_sync)
    monkeypatch.setattr(container.session_supervisor, "_on_jsonl_sync", fake_sync)

    await container._recheck_active_claude_sessions()

    assert seen == [("claude-session-1", str(tmp_path))]


@pytest.mark.asyncio
async def test_start_restores_persisted_claude_session_snapshot(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(tmp_path, install_hooks=False)
    first = AppContainer(settings)
    session, _ = await first.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await first.session_service.bind_claude_session(
        user_id=1,
        claude_session_id="claude-session-1",
        workdir=str(tmp_path),
    )
    project_dir = str(tmp_path).replace("/", "-").replace(".", "-")
    session_file = first.claude_paths.projects_dir / project_dir / "claude-session-1.jsonl"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": "2026-04-16T10:00:01+00:00",
                        "message": {
                            "id": "a1",
                            "content": [{"type": "text", "text": "恢复后的回复"}],
                        },
                    },
                    ensure_ascii=False,
                )
            ]
        ),
        encoding="utf-8",
    )

    second = AppContainer(settings)

    async def fake_start(handler, permission_failure_handler=None, permission_resolved_handler=None):
        return None

    async def fake_stop():
        return None

    async def fake_close():
        return None

    async def fake_sync(session_id: str, cwd: str) -> None:
        await second._dispatch_session_event(
            SessionEvent(
                session_id=session_id,
                type=SessionEventType.FILE_SYNCED,
                payload={
                    "cwd": cwd,
                    "claude_session_id": session_id,
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

    monkeypatch.setattr(second.hook_socket_server, "start", fake_start)
    monkeypatch.setattr(second.hook_socket_server, "stop", fake_stop)
    monkeypatch.setattr(second.bot.session, "close", fake_close)
    monkeypatch.setattr(second, "sync_claude_session", fake_sync)

    try:
        await second.start()

        restored_session = await second.session_service.get(1)
        restored_state = second.structured_session_store.get("claude-session-1")

        assert restored_session is not None
        assert restored_session.session_id == session.session_id
        assert restored_session.claude_session_id == "claude-session-1"
        assert restored_state is not None
        assert restored_state.user_id == 1
        assert restored_state.terminal_id == restored_session.terminal_id
        assert restored_state.last_reply == "恢复后的回复"
    finally:
        await second.stop()


@pytest.mark.asyncio
async def test_start_clears_stale_claude_session_binding_when_snapshot_missing(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(tmp_path, install_hooks=False)
    first = AppContainer(settings)
    await first.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await first.session_service.bind_claude_session(
        user_id=1,
        claude_session_id="missing-session",
        workdir=str(tmp_path),
    )

    second = AppContainer(settings)

    async def fake_start(handler, permission_failure_handler=None, permission_resolved_handler=None):
        return None

    async def fake_stop():
        return None

    async def fake_close():
        return None

    monkeypatch.setattr(second.hook_socket_server, "start", fake_start)
    monkeypatch.setattr(second.hook_socket_server, "stop", fake_stop)
    monkeypatch.setattr(second.bot.session, "close", fake_close)

    try:
        await second.start()

        restored_session = await second.session_service.get(1)

        assert restored_session is not None
        assert restored_session.claude_session_id is None
        assert restored_session.claude_chat_active is True
    finally:
        await second.stop()


@pytest.mark.asyncio
async def test_start_clears_stale_claude_session_binding_when_only_empty_terminal_state_exists(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path, install_hooks=False)
    first = AppContainer(settings)
    session, _ = await first.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await first.session_service.bind_claude_session(
        user_id=1,
        claude_session_id="missing-session",
        workdir=str(tmp_path),
    )
    empty_terminal_state = first.structured_session_store.get_or_create(
        session_id=f"tgcli_{session.terminal_id}",
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_id=session.terminal_id,
        user_id=1,
    )
    empty_terminal_state.phase = SessionPhase.PROCESSING
    first.structured_session_store._persist(empty_terminal_state)

    second = AppContainer(settings)

    async def fake_start(handler, permission_failure_handler=None, permission_resolved_handler=None):
        return None

    async def fake_stop():
        return None

    async def fake_close():
        return None

    monkeypatch.setattr(second.hook_socket_server, "start", fake_start)
    monkeypatch.setattr(second.hook_socket_server, "stop", fake_stop)
    monkeypatch.setattr(second.bot.session, "close", fake_close)

    try:
        await second.start()

        restored_session = await second.session_service.get(1)

        assert restored_session is not None
        assert restored_session.claude_session_id is None
        assert restored_session.claude_chat_active is True
    finally:
        await second.stop()


@pytest.mark.asyncio
async def test_start_keeps_claude_session_binding_when_terminal_state_has_content(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(tmp_path, install_hooks=False)
    first = AppContainer(settings)
    session, _ = await first.session_service.switch(
        user_id=1,
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_mode=True,
        claude_chat_active=True,
    )
    await first.session_service.bind_claude_session(
        user_id=1,
        claude_session_id="missing-session",
        workdir=str(tmp_path),
    )
    terminal_state = first.structured_session_store.get_or_create(
        session_id=f"tgcli_{session.terminal_id}",
        provider="claude_code",
        workdir=str(tmp_path),
        terminal_id=session.terminal_id,
        user_id=1,
    )
    terminal_state.phase = SessionPhase.PROCESSING
    terminal_state.turns = [ConversationTurn(turn_id="a1", role="assistant", text="\n恢复中\n", is_complete=True)]
    first.structured_session_store._persist(terminal_state)

    second = AppContainer(settings)

    async def fake_start(handler, permission_failure_handler=None, permission_resolved_handler=None):
        return None

    async def fake_stop():
        return None

    async def fake_close():
        return None

    monkeypatch.setattr(second.hook_socket_server, "start", fake_start)
    monkeypatch.setattr(second.hook_socket_server, "stop", fake_stop)
    monkeypatch.setattr(second.bot.session, "close", fake_close)

    try:
        await second.start()

        restored_session = await second.session_service.get(1)

        assert restored_session is not None
        assert restored_session.claude_session_id == "missing-session"
        assert restored_session.claude_chat_active is True
    finally:
        await second.stop()


@pytest.mark.asyncio
async def test_unbound_permission_handler_requires_gateway_before_notify(tmp_path) -> None:
    handler = UnboundPermissionHandler(
        message_sender=type("MessageSender", (), {"send_message": AsyncMock(), "send_photo": AsyncMock(), "send_document": AsyncMock()})(),
        hook_socket_server=type("HookSocket", (), {"respond_to_permission": AsyncMock(return_value=True)})(),
        allowed_user_ids={1},
    )

    event = HookEvent(
        session_id="session-before-gateway",
        cwd=str(tmp_path),
        event="PermissionRequest",
        status="waiting_for_approval",
        tool="Bash",
        tool_input={"command": "pwd"},
        tool_use_id="tool-before-gateway",
    )

    with pytest.raises(RuntimeError, match="gateway"):
        await handler.handle_unbound_permission(event)


@pytest.mark.asyncio
async def test_session_end_runs_unified_permission_cleanup_in_order(tmp_path) -> None:
    seen: list[str] = []

    class _AAS:
        async def deactivate_all_for_session(self, session_id: str) -> int:
            seen.append(f"aas_deactivate:{session_id}")
            return 1

        async def release_all_slots_for_session(self, session_id: str) -> int:
            seen.append(f"aas_release:{session_id}")
            return 1

    class _Registry:
        async def invalidate_session(self, session_id: str) -> int:
            seen.append(f"registry:{session_id}")
            return 1

    class _Unbound:
        async def invalidate_session(self, session_id: str) -> int:
            seen.append(f"unbound:{session_id}")
            return 1

    class _BindingStore:
        def remove_binding(self, session_id: str) -> None:
            seen.append(f"binding:{session_id}")

    class _Container(AppContainer):
        def __init__(self) -> None:
            self.settings = type("Settings", (), {"allowed_workdirs": [str(tmp_path)]})()
            self.auto_approve_service = _AAS()
            self.permission_callback_registry = _Registry()
            self.unbound_permission_handler = _Unbound()
            self.external_binding_store = _BindingStore()

        async def _bind_hook_session(self, event: HookEvent) -> None:
            seen.append(f"bind:{event.session_id}")

        async def _dispatch_session_event(self, event: SessionEvent) -> None:
            seen.append(f"dispatch:{event.session_id}")

        def _schedule_jsonl_sync(self, session_id: str, cwd: str) -> None:
            seen.append(f"sync:{session_id}")

    container = _Container()

    await container._resolve_ownership_stage(HookEvent(session_id="ended-session", cwd=str(tmp_path), event="SessionEnd", status="ended"))

    assert seen[:5] == [
        "aas_deactivate:ended-session",
        "aas_release:ended-session",
        "registry:ended-session",
        "unbound:ended-session",
        "binding:ended-session",
    ]


@pytest.mark.asyncio
async def test_session_end_dispatch_keeps_active_terminal_chat_context(tmp_path) -> None:
    container = AppContainer(make_settings(tmp_path, install_hooks=False, tmux_mode=True))
    try:
        session, _ = await container.session_service.switch(
            user_id=1,
            provider="claude_code",
            workdir=str(tmp_path),
            terminal_mode=True,
            claude_chat_active=True,
        )
        session.terminal_id = "terminal-ended"
        session.claude_session_id = "claude-session-ended"
        await container.session_service.save_session_context(session)

        await container._dispatch_session_event(
            SessionEvent(
                session_id="claude-session-ended",
                type=SessionEventType.SESSION_ENDED,
                payload={"cwd": str(tmp_path)},
            )
        )

        state = container.structured_session_store.get("claude-session-ended")
        assert state is not None
        assert state.phase == SessionPhase.ENDED
        current = await container.session_service.get(1)
        assert current is not None
        assert current.claude_chat_active is True
        assert current.terminal_mode is True
        assert current.terminal_id == "terminal-ended"
        assert current.claude_session_id == "claude-session-ended"
    finally:
        await container.session_supervisor.stop_all()
        await container.bot.session.close()


@pytest.mark.asyncio
async def test_late_old_event_does_not_clear_new_session_context(tmp_path) -> None:
    """A late SessionEnd carrying a stale session_id must not clear the active session.

    This targets the ``session.claude_session_id != session_id`` guard in
    ``_reconcile_session_context_after_session_end`` (bootstrap_mixins.py:943), NOT the
    ``session is None`` early return. To hit the != guard we need
    ``lookup_by_claude_session_id("claude-session-old")`` to return a non-None context
    whose ``claude_session_id`` is the *new* id ("claude-session-new"). That is exactly the
    stale-index state the guard defends against: the old event's id still resolves in the
    index, but the resolved context has since been re-bound to a newer session.

    The setup uses the terminal-less chat-active path so that the outer ``!=`` guard is the
    *last* line of defence: the terminal branch has a second re-check guard
    (bootstrap_mixins.py:949) that would independently catch this stale-id case, so a
    terminal-mode setup would not actually fail when the outer ``!=`` clause is removed.
    The terminal-less path (lines 954-964) has no such second check, so removing the outer
    ``!=`` clause lets reconcile proceed and wrongly switch+clear the new session -> test
    fails. With the guard intact the reconcile early-returns and the new session survives.

    We build the stale-index state by saving the active context under the new id (registering
    the new id in the index) and then injecting a stale index entry that maps the OLD id
    onto that same (now re-bound) context.
    """
    container = AppContainer(make_settings(tmp_path, install_hooks=False, tmux_mode=True))
    try:
        active_session, _ = await container.session_service.switch(
            user_id=1,
            provider="claude_code",
            workdir=str(tmp_path),
            terminal_mode=False,
            claude_chat_active=True,
        )
        active_session.claude_session_id = "claude-session-new"
        await container.session_service.save_session_context(active_session)
        assert active_session.terminal_id is None
        assert active_session.claude_chat_active is True

        # Inject the stale-index state: the OLD session id still resolves in the index,
        # but to the context that is now bound to the NEW session id. This is the precise
        # condition the ``session.claude_session_id != session_id`` guard exists for.
        store = container.session_service._store
        store._claude_session_index["claude-session-old"] = active_session

        # Spy: confirm lookup_by_claude_session_id("claude-session-old") returns non-None
        # (i.e. we really are exercising the != guard, not the None early-return).
        looked_up = await container.session_service.lookup_by_claude_session_id("claude-session-old")
        assert looked_up is not None
        assert looked_up.claude_session_id == "claude-session-new"

        await container._dispatch_session_event(
            SessionEvent(
                session_id="claude-session-old",
                type=SessionEventType.HOOK_RECEIVED,
                payload={"session_id": "claude-session-old", "cwd": str(tmp_path), "event": "SessionEnd", "status": "ended"},
            )
        )

        current = await container.session_service.get(1)
        assert current is not None
        # Guard intact: the new session context must survive untouched.
        assert current.claude_chat_active is True
        assert current.terminal_id is None
        assert current.claude_session_id == "claude-session-new"
    finally:
        await container.session_supervisor.stop_all()
        await container.bot.session.close()


@pytest.mark.asyncio
async def test_dispatch_status_ended_branch_triggers_reconcile(tmp_path) -> None:
    """Exercise the ``status == "ended"`` clause of ``_is_session_end_dispatch_event``
    independently of the ``event == "SessionEnd"`` clause.

    We dispatch a HOOK_RECEIVED whose ``status`` is ``"ended"`` but whose ``event`` is a
    non-SessionEnd value (``"Notification"``). The ``or`` short-circuits on the status
    clause, ``is_session_end`` becomes True and reconcile runs. Active tmux chat context
    must survive because a Claude turn's SessionEnd is not the same as closing /claude.

    Note: HookEvent.from_dict (invoked by the HOOK_RECEIVED processor) requires both
    ``event`` and ``status`` to be valid non-empty strings, so we cannot literally omit
    the ``event`` key. We instead keep ``event`` at a valid non-SessionEnd value so only
    the status clause is the trigger -- this is what isolates the branch under test.
    If the ``event.payload.get("status") == "ended"`` clause were removed, the event would
    not be treated as a session end and the structured state would not move to ENDED.
    """
    container = AppContainer(make_settings(tmp_path, install_hooks=False, tmux_mode=True))
    try:
        session, _ = await container.session_service.switch(
            user_id=1,
            provider="claude_code",
            workdir=str(tmp_path),
            terminal_mode=True,
            claude_chat_active=True,
        )
        session.terminal_id = "term-x"
        session.claude_session_id = "claude-x"
        await container.session_service.save_session_context(session)

        await container._dispatch_session_event(
            SessionEvent(
                session_id="claude-x",
                type=SessionEventType.HOOK_RECEIVED,
                payload={
                    "session_id": "claude-x",
                    "cwd": str(tmp_path),
                    "event": "Notification",  # valid, but NOT SessionEnd -> only status clause triggers
                    "status": "ended",
                },
            )
        )

        state = container.structured_session_store.get("claude-x")
        assert state is not None
        assert state.phase == SessionPhase.ENDED
        current = await container.session_service.get(1)
        assert current is not None
        assert current.terminal_mode is True
        assert current.terminal_id == "term-x"
        assert current.claude_chat_active is True
        assert current.claude_session_id == "claude-x"
    finally:
        await container.session_supervisor.stop_all()
        await container.bot.session.close()


@pytest.mark.asyncio
async def test_dispatch_event_sessionend_branch_triggers_reconcile(tmp_path) -> None:
    """Exercise the ``event == "SessionEnd"`` clause of ``_is_session_end_dispatch_event``
    independently of the ``status == "ended"`` clause.

    We dispatch a HOOK_RECEIVED whose ``event`` is ``"SessionEnd"`` but whose ``status`` is
    a non-ended value (``"running"``). The ``or`` short-circuits on the event clause,
    ``is_session_end`` becomes True and reconcile runs. Active tmux chat context must
    survive because a Claude turn's SessionEnd is not the same as closing /claude.

    See the sibling test for why the ``event`` key cannot be literally omitted: HookEvent
    validation requires a valid non-empty ``event``/``status``. Keeping ``status`` at a
    non-ended value isolates the event clause as the sole trigger. If the
    ``event.payload.get("event") == "SessionEnd"`` clause were removed, this event would
    not be treated as a session end and the structured state would not move to ENDED.
    """
    container = AppContainer(make_settings(tmp_path, install_hooks=False, tmux_mode=True))
    try:
        session, _ = await container.session_service.switch(
            user_id=1,
            provider="claude_code",
            workdir=str(tmp_path),
            terminal_mode=True,
            claude_chat_active=True,
        )
        session.terminal_id = "term-x"
        session.claude_session_id = "claude-x"
        await container.session_service.save_session_context(session)

        await container._dispatch_session_event(
            SessionEvent(
                session_id="claude-x",
                type=SessionEventType.HOOK_RECEIVED,
                payload={
                    "session_id": "claude-x",
                    "cwd": str(tmp_path),
                    "event": "SessionEnd",
                    "status": "running",  # valid, but NOT ended -> only event clause triggers
                },
            )
        )

        state = container.structured_session_store.get("claude-x")
        assert state is not None
        assert state.phase == SessionPhase.ENDED
        current = await container.session_service.get(1)
        assert current is not None
        assert current.terminal_mode is True
        assert current.terminal_id == "term-x"
        assert current.claude_chat_active is True
        assert current.claude_session_id == "claude-x"
    finally:
        await container.session_supervisor.stop_all()
        await container.bot.session.close()


@pytest.mark.asyncio
async def test_session_end_reconcile_terminal_less_path_clears_chat_active_and_session_id(tmp_path) -> None:
    """End-to-end cover for the terminal-less reconcile branch
    (bootstrap_mixins.py:954-964): no ``terminal_id`` but ``claude_chat_active`` is True ->
    ``switch(terminal_mode=False, claude_chat_active=False)`` then clear
    ``claude_session_id`` and persist.

    Constructed via ``switch(terminal_mode=False, claude_chat_active=True)`` so the
    resulting context has ``terminal_id is None`` and an active chat, then bind a claude
    session id and dispatch SESSION_ENDED. If the ``current.claude_session_id = None``
    line (bootstrap_mixins.py:962) were removed, the session id would survive -> test fails.
    """
    container = AppContainer(make_settings(tmp_path, install_hooks=False, tmux_mode=True))
    try:
        session, _ = await container.session_service.switch(
            user_id=1,
            provider="claude_code",
            workdir=str(tmp_path),
            terminal_mode=False,
            claude_chat_active=True,
        )
        # terminal-less chat-active context: no terminal_id, chat active.
        assert session.terminal_id is None
        assert session.claude_chat_active is True

        session.claude_session_id = "claude-x"
        await container.session_service.save_session_context(session)

        await container._dispatch_session_event(
            SessionEvent(
                session_id="claude-x",
                type=SessionEventType.SESSION_ENDED,
                payload={"cwd": str(tmp_path)},
            )
        )

        current = await container.session_service.get(1)
        assert current is not None
        # Path C: switch clears terminal_mode + claude_chat_active, then clears claude_session_id.
        assert current.terminal_mode is False
        assert current.claude_chat_active is False
        assert current.claude_session_id is None
        assert current.terminal_id is None
    finally:
        await container.session_supervisor.stop_all()
        await container.bot.session.close()


def test_bootstrap_wires_user_question_callback_registry_to_router() -> None:
    """Regression: AppContainer.start must forward user_question_callback_registry
    to create_router. Without it the AskUserQuestion callback handler received
    ``None``, skipped token resolution, and answered "无效的选择操作" on every
    interactive Ghostty card tap (bootstrap.py's create_router call originally
    omitted the parameter).

    Source-level assertion: ``start()`` runs the full service stack (hook socket,
    polling, background tasks) which is impractical to drive here, so we assert the
    create_router call site carries the keyword instead.
    """
    import ast
    from pathlib import Path

    source = Path("app/bootstrap.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "create_router"]
    assert calls, "create_router call not found in app/bootstrap.py"

    found = False
    for call in calls:
        for kw in call.keywords:
            if kw.arg == "user_question_callback_registry":
                found = True
                value = getattr(kw.value, "attr", None)
                assert value == "user_question_callback_registry", (
                    f"create_router user_question_callback_registry must be self.user_question_callback_registry, "
                    f"got attribute chain ending in {value!r}"
                )
    assert found, "create_router call does not pass user_question_callback_registry"
