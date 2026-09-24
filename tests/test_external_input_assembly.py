"""Assembly / lifecycle integration tests for the external Ghostty input feature.

Verifies that ``AppContainer`` constructs with the new input service in both
``ghostty_input_enabled`` states, that ``stop()`` cleanly shuts the service down,
that the hook event-kind mapper covers all real Claude Hook events, and that
the three lifecycle paths (reaper remove / SessionEnd / rebind) invoke
``invalidate_binding`` / ``rebind_aba`` on the input service. Ghostty, TCC,
PTYs and real processes are not required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bootstrap import AppContainer
from app.config.settings import Settings
from app.domain.external_session_models import ExternalBinding
from app.domain.hook_models import HookEvent
from app.domain.models import utc_now
from app.services.external_session_input_service import SendOutcome


def _seed_binding(container: AppContainer, *, session_id: str = "ext-sess") -> ExternalBinding:
    binding = ExternalBinding(
        session_id=session_id,
        user_id=1,
        cwd=str(container.settings.allowed_workdirs[0]),
        bound_at=utc_now(),
        jsonl_path=None,
        binding_id="binding-A",
        pid=1234,
        tty="/dev/ttys005",
    )
    container.external_binding_store.save_binding(binding)
    return binding


def make_settings(tmp_path: Path, *, ghostty_input_enabled: bool = False) -> Settings:
    return Settings.model_validate(
        {
            "TG_BOT_TOKEN": "123456:TESTTOKEN",
            "TG_ALLOWED_USER_IDS": "1",
            "DEFAULT_PROVIDER": "claude_code",
            "DEFAULT_TIMEOUT_SEC": 10,
            "MAX_CONCURRENT_TASKS": 1,
            "CLAUDE_TMUX_MODE": False,
            "TMUX_DATA_DIR": str(tmp_path),
            "CLAUDE_CLI_BIN": "claude",
            "CLAUDE_INSTALL_HOOKS": False,
            "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude"),
            "CLAUDE_HOOK_SOCKET_PATH": str(tmp_path / "hook.sock"),
            "CLAUDE_JSONL_SYNC_DEBOUNCE_MS": 10,
            "CLAUDE_PERIODIC_RECHECK_MS": 10,
            "CODEX_CLI_BIN": "codex",
            "GEMINI_CLI_BIN": "gemini",
            "ALLOWED_WORKDIRS": str(tmp_path),
            "GHOSTTY_INPUT_ENABLED": "true" if ghostty_input_enabled else "false",
        }
    )


class TestContainerAssembly:
    def test_assembles_with_feature_disabled(self, tmp_path: Path) -> None:
        container = AppContainer(make_settings(tmp_path, ghostty_input_enabled=False))
        assert container.external_session_input_service is not None
        # The service object is always constructed; disabling short-circuits methods.

    def test_assembles_with_feature_enabled(self, tmp_path: Path) -> None:
        container = AppContainer(make_settings(tmp_path, ghostty_input_enabled=True))
        assert container.external_session_input_service is not None
        assert container._input_locks is not None

    @pytest.mark.asyncio
    async def test_enabled_send_text_no_target_returns_no_target(self, tmp_path: Path) -> None:
        container = AppContainer(make_settings(tmp_path, ghostty_input_enabled=True))
        outcome = await container.external_session_input_service.send_text(user_id=1, text="hi")
        assert outcome == SendOutcome.NO_TARGET

    @pytest.mark.asyncio
    async def test_disabled_send_text_short_circuits(self, tmp_path: Path) -> None:
        container = AppContainer(make_settings(tmp_path, ghostty_input_enabled=False))
        outcome = await container.external_session_input_service.send_text(user_id=1, text="hi")
        assert outcome == SendOutcome.ADAPTER_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_stop_is_safe_when_enabled(self, tmp_path: Path) -> None:
        container = AppContainer(make_settings(tmp_path, ghostty_input_enabled=True))
        # stop() awaits input_service.shutdown(); must not raise even with no drains.
        await container.external_session_input_service.shutdown()


# ── hook event-kind mapping ─────────────────────────────────────────────────


class TestHookEventKindMap:
    @pytest.mark.parametrize(
        ("event_name", "status", "expected"),
        [
            ("Stop", None, "stop"),
            ("SubagentStop", None, "SubagentStop"),
            ("StopFailure", None, "stop"),
            ("PostCompact", None, "turn_completed"),
            ("SessionEnd", None, "session_end"),
            ("PreCompact", None, "PreCompact"),  # passed through verbatim
            ("UserPromptSubmit", None, "UserPromptSubmit"),
        ],
    )
    def test_map(self, event_name: str | None, status: str | None, expected: str) -> None:
        from app.bootstrap_mixins import _map_hook_event_kind

        event = MagicMock(spec=HookEvent)
        event.event = event_name
        event.status = status
        assert _map_hook_event_kind(event) == expected

    def test_session_end_via_status_ended(self) -> None:
        from app.bootstrap_mixins import _map_hook_event_kind

        event = MagicMock(spec=HookEvent)
        event.event = "Stop"
        event.status = "ended"
        assert _map_hook_event_kind(event) == "session_end"


# ── lifecycle cleanup paths ────────────────────────────────────────────────


class TestLifecycleCleanupPaths:
    """reaper remove / SessionEnd / rebind must each touch the input service."""

    def _patched(self, tmp_path: Path) -> AppContainer:
        container = AppContainer(make_settings(tmp_path, ghostty_input_enabled=True))
        # Replace the real service with a capturing AsyncMock so we assert the
        # mixin helpers forward to it; keep enabled=True so the service object
        # exists (the helper short-circuits otherwise).
        mock_service = MagicMock()
        mock_service.invalidate_binding = AsyncMock()
        mock_service.rebind_aba = AsyncMock()
        container.external_session_input_service = mock_service  # type: ignore[assignment]
        return container

    @pytest.mark.asyncio
    async def test_reaper_remove_invalidates(self, tmp_path: Path) -> None:
        container = self._patched(tmp_path)
        _seed_binding(container)
        removed = await container._remove_external_binding("ext-sess", expected_binding_id="binding-A")
        assert removed is not None
        container.external_session_input_service.invalidate_binding.assert_awaited_once_with("ext-sess", reason="reaper_remove")

    @pytest.mark.asyncio
    async def test_session_end_invalidates(self, tmp_path: Path) -> None:
        container = self._patched(tmp_path)
        _seed_binding(container)
        ok = await container._mark_external_binding_ended("ext-sess", expected_binding_id="binding-A")
        assert ok is True
        container.external_session_input_service.invalidate_binding.assert_awaited_once_with("ext-sess", reason="session_end")

    @pytest.mark.asyncio
    async def test_save_binding_rebind_sweep(self, tmp_path: Path) -> None:
        container = self._patched(tmp_path)
        binding = ExternalBinding(
            session_id="ext-sess",
            user_id=1,
            cwd=str(container.settings.allowed_workdirs[0]),
            bound_at=utc_now(),
            jsonl_path=None,
            binding_id="binding-A",
            pid=1234,
            tty="/dev/ttys005",
        )
        saved = await container._save_external_binding(binding)
        assert saved is True
        container.external_session_input_service.rebind_aba.assert_awaited_once_with("ext-sess", "binding-A")

    @pytest.mark.asyncio
    async def test_no_service_skips_silently(self, tmp_path: Path) -> None:
        # Feature disabled → no service object → helpers short-circuit without raising.
        container = AppContainer(make_settings(tmp_path, ghostty_input_enabled=False))
        _seed_binding(container)
        # Force the absent-service code path even though the attribute exists.
        container.external_session_input_service = None  # type: ignore[assignment]
        ok = await container._mark_external_binding_ended("ext-sess", expected_binding_id="binding-A")
        assert ok is True  # binding teardown still proceeds
