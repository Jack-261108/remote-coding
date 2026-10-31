"""Hook routing for external Ghostty AskUserQuestion (design §F).

``_try_external_terminal_user_question`` routes a bound, paired Ghostty session's
AskUserQuestion to the interactive transport: it stores a pending question and
pushes an interactive ``ask:``-tokenised card, deferring Hook ``allow`` until
the user answers. Returns True to short-circuit the generic permission card;
returns False (and pushes nothing) when the binding is missing, ended, owned by
another user, or has no Ghostty target — letting the caller fall back.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.bootstrap import AppContainer
from app.config.settings import Settings
from app.domain.external_session_models import ExternalBinding, ExternalTerminalTarget
from app.domain.hook_models import HookEvent
from app.domain.models import utc_now
from app.domain.user_question_models import (
    ExternalTerminalQuestionTarget,
    UserQuestionOption,
    UserQuestionPrompt,
)
from app.services.user_question_callback_registry import UserQuestionCallbackOrigin
from tests.fakes.external_session import make_hook_event


def _make_settings(tmp_path, *, install_hooks: bool = False) -> Settings:
    data = {
        "TG_BOT_TOKEN": "123456:TESTTOKEN",
        "TG_ALLOWED_USER_IDS": "1",
        "DEFAULT_PROVIDER": "claude_code",
        "DEFAULT_TIMEOUT_SEC": 10,
        "MAX_CONCURRENT_TASKS": 1,
        "CLAUDE_TMUX_MODE": False,
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
    return Settings.model_validate(data)


def _ghostty_binding(session_id: str, user_id: int = 1) -> ExternalBinding:
    binding = ExternalBinding(
        session_id=session_id,
        user_id=user_id,
        cwd="/project",
        bound_at=utc_now(),
        jsonl_path=None,
        binding_id="bind-1",
        pid=1234,
        tty="/dev/ttys005",
    )
    binding.terminal_target = ExternalTerminalTarget(
        kind="ghostty",
        terminal_id="term-1",
        paired_tty="/dev/ttys005",
        paired_at=utc_now(),
        binding_id="bind-1",
        name="claude — project",
        cwd="/project",
    )
    return binding


def _prompts(tool_use_id: str = "tuid-1") -> tuple[UserQuestionPrompt, ...]:
    return (
        UserQuestionPrompt(
            tool_use_id=tool_use_id,
            question_index=0,
            total_questions=1,
            question="Pick?",
            options=(UserQuestionOption(label="A"), UserQuestionOption(label="B")),
        ),
    )


def _event(session_id: str, tool_use_id: str = "tuid-1") -> HookEvent:
    return make_hook_event(
        session_id=session_id,
        cwd="/project",
        event="PreToolUse",
        tool="AskUserQuestion",
        tool_use_id=tool_use_id,
        tool_input={"questions": [{"question": "Pick?", "options": [{"label": "A"}, {"label": "B"}]}]},
        status="running",
    )


@pytest.mark.asyncio
async def test_ghostty_question_stores_pending_and_pushes_interactive_card(tmp_path) -> None:
    container = AppContainer(_make_settings(tmp_path))
    session_id = "ghostty-q"
    container.external_binding_store.save_binding(_ghostty_binding(session_id, user_id=1))

    notify = AsyncMock(return_value=True)
    container.push_notifier.notify_user_question = notify  # type: ignore[method-assign]

    handled = await container._try_external_terminal_user_question(event=_event(session_id), user_id=1, prompts=_prompts())

    assert handled is True
    notify.assert_awaited_once()
    assert notify.call_args.kwargs["interactive"] is True
    assert notify.call_args.kwargs["origin"] is UserQuestionCallbackOrigin.EXTERNAL_TERMINAL
    snapshot = container.external_uq_state.get_active("tuid-1")
    assert snapshot is not None
    assert snapshot.user_id == 1
    assert isinstance(snapshot.target, ExternalTerminalQuestionTarget)
    assert snapshot.target.binding_id == "bind-1"


@pytest.mark.asyncio
async def test_ghostty_question_falls_back_when_no_terminal_target(tmp_path) -> None:
    container = AppContainer(_make_settings(tmp_path))
    session_id = "no-target"
    binding = _ghostty_binding(session_id)
    binding.terminal_target = None
    container.external_binding_store.save_binding(binding)

    notify = AsyncMock(return_value=True)
    container.push_notifier.notify_user_question = notify  # type: ignore[method-assign]

    handled = await container._try_external_terminal_user_question(event=_event(session_id), user_id=1, prompts=_prompts())

    assert handled is False
    notify.assert_not_awaited()
    assert container.external_uq_state.get_active("tuid-1") is None


@pytest.mark.asyncio
async def test_ghostty_question_falls_back_when_owner_mismatch(tmp_path) -> None:
    container = AppContainer(_make_settings(tmp_path))
    session_id = "owner-mismatch"
    container.external_binding_store.save_binding(_ghostty_binding(session_id, user_id=1))

    notify = AsyncMock(return_value=True)
    container.push_notifier.notify_user_question = notify  # type: ignore[method-assign]

    handled = await container._try_external_terminal_user_question(event=_event(session_id), user_id=2, prompts=_prompts())

    assert handled is False
    notify.assert_not_awaited()
    assert container.external_uq_state.get_active("tuid-1") is None


@pytest.mark.asyncio
async def test_ghostty_question_falls_back_when_binding_ended(tmp_path) -> None:
    container = AppContainer(_make_settings(tmp_path))
    session_id = "ended"
    binding = _ghostty_binding(session_id)
    binding.ended_at = utc_now()
    container.external_binding_store.save_binding(binding)

    notify = AsyncMock(return_value=True)
    container.push_notifier.notify_user_question = notify  # type: ignore[method-assign]

    handled = await container._try_external_terminal_user_question(event=_event(session_id), user_id=1, prompts=_prompts())

    assert handled is False
    notify.assert_not_awaited()
    assert container.external_uq_state.get_active("tuid-1") is None
