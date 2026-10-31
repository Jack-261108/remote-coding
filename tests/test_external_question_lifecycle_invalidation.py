"""Lifecycle invalidation matrix for external Ghostty AskUserQuestion.

Each failure point that tears down external state must also invalidate the
matching pending question and its opaque callback tokens:

    | failure point            | external_uq_state              | registry            |
    |--------------------------|--------------------------------|---------------------|
    | SessionEnd / invalidate_binding  | invalidate_session       | invalidate_session  |
    | dead-unbind               | invalidate_session             | invalidate_session  |
    | reaper manual_unbind/pid_dead   | invalidate_session       | invalidate_session  |
    | reaper idle_ttl_expired   | (kept by design)               | (kept by design)    |
    | rebind_aba                | invalidate_stale_bindings       | invalidate_session  |
    | re-pair consume_pair_token| invalidate_terminal_target      | invalidate_session  |

These tests verify that after each teardown the pending question is gone
(``get_active`` is None) and the registry token no longer resolves
(``UserQuestionCallbackNotFound``), so stale Telegram buttons cannot drive a
dead binding. ABA rebind additionally clears only the *old* binding_id;
re-pair additionally invalidates the previously-registered token.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.adapters.storage.file_session_store import FileSessionStore
from app.domain.external_session_models import ExternalBinding, ExternalTerminalTarget
from app.domain.models import utc_now
from app.domain.user_question_models import (
    ExternalTerminalQuestionTarget,
    UserQuestionOption,
    UserQuestionPrompt,
)
from app.infra.lock_registry import RefCountedLockRegistry
from app.services.auto_approve_service import AutoApproveService
from app.services.external_binding_reaper import ExternalBindingReaper
from app.services.external_binding_store import ExternalBindingStore
from app.services.external_input_mode_state import ExternalInputTargetStore
from app.services.external_input_queue import ExternalInputQueue
from app.services.external_session_input_service import ExternalSessionInputService
from app.services.external_user_question_state import (
    ExternalUserQuestionState,
    PendingExternalUserQuestion,
)
from app.services.pairing_callback_registry import PairingCallbackRegistry
from app.services.session_store import SessionStore
from app.services.user_question_callback_registry import (
    UserQuestionCallbackNotFound,
    UserQuestionCallbackOrigin,
    UserQuestionCallbackRegistry,
    UserQuestionCallbackResolved,
)
from tests.fakes.external_session import make_binding
from tests.fakes.ghostty import FakeGhosttyTerminalAdapter
from tests.fakes.process_probe import FakeLocalProcessProbe


def _binding(session_id: str, binding_id: str, *, paired: bool = True) -> ExternalBinding:
    binding = make_binding(
        session_id=session_id,
        user_id=42,
        cwd="/project",
        binding_id=binding_id,
        pid=1234,
        tty="/dev/ttys005",
    )
    if paired:
        binding.terminal_target = ExternalTerminalTarget(
            kind="ghostty",
            terminal_id="term-1",
            paired_tty="/dev/ttys005",
            paired_at=utc_now(),
            binding_id=binding_id,
            name="claude — project",
            cwd="/project",
        )
    return binding


def _pending(session_id: str, binding_id: str, tool_use_id: str = "tuid-1") -> PendingExternalUserQuestion:
    target = ExternalTerminalQuestionTarget(
        binding_id=binding_id,
        terminal_id="term-1",
        paired_tty="/dev/ttys005",
        paired_at=utc_now(),
        kind="ghostty",
    )
    return PendingExternalUserQuestion(
        tool_use_id=tool_use_id,
        session_id=session_id,
        user_id=42,
        prompts=(
            UserQuestionPrompt(
                tool_use_id=tool_use_id,
                question_index=0,
                total_questions=1,
                question="Pick?",
                options=(UserQuestionOption(label="A"), UserQuestionOption(label="B")),
            ),
        ),
        target=target,
    )


async def _register_token(
    registry: UserQuestionCallbackRegistry,
    *,
    session_id: str,
    tool_use_id: str,
) -> str:
    tokens = await registry.register_question_tokens(
        owner_user_id=42,
        session_id=session_id,
        tool_use_id=tool_use_id,
        question_index=0,
        option_count=1,
        multi_select=False,
        origin=UserQuestionCallbackOrigin.EXTERNAL_TERMINAL,
    )
    return tokens.select_tokens[0]


class _Harness:
    """Lightweight wiring around an ExternalSessionInputService for invalidation tests."""

    def __init__(self, tmp_path: Path, *, session_id: str = "sess-1", binding_id: str = "bind-1") -> None:
        root = tmp_path / "harness"
        self.binding_store = ExternalBindingStore(root / "binding")
        self.session_id = session_id
        self.binding_id = binding_id
        self.binding = _binding(session_id, binding_id)
        self.binding_store.save_binding(self.binding)

        self.session_store = SessionStore(FileSessionStore(str(root / "state")))
        state = self.session_store.get_or_create(session_id=session_id, user_id=42, workdir="/project", claude_session_id=session_id)
        self.session_store.save(state)

        self.adapter = FakeGhosttyTerminalAdapter()
        self.probe = FakeLocalProcessProbe()
        self.pairing = PairingCallbackRegistry(ttl_sec=60)
        self.external_uq_state = ExternalUserQuestionState(ttl_sec=60)
        self.registry = UserQuestionCallbackRegistry(ttl_sec=60)
        self.mode_store = ExternalInputTargetStore()
        self.queue = ExternalInputQueue(max_size=5, ttl_sec=60)
        self.locks = RefCountedLockRegistry(ttl_sec=60, cleanup_interval_sec=60, cleanup_batch_size=50)

        self.input_service = ExternalSessionInputService(
            enabled=True,
            binding_store=self.binding_store,
            session_store=self.session_store,
            backends={"ghostty": self.adapter},
            process_probe=self.probe,  # type: ignore[arg-type]
            pairing_registry=self.pairing,
            input_mode_store=self.mode_store,
            input_queue=self.queue,
            input_locks=self.locks,
            external_user_question_state=self.external_uq_state,
            user_question_callback_registry=self.registry,
            drain_publish_wait_timeout_sec=0.05,
        )
        # Hook socket double reaper ultimately needs nothing here; constructed in tests below.


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[_Harness]:
    h = _Harness(tmp_path)
    yield h
    await h.input_service.shutdown()


@pytest.mark.asyncio
async def test_invalidate_binding_clears_pending_and_token(harness: _Harness) -> None:
    harness.external_uq_state.store(_pending(harness.session_id, harness.binding_id))
    token = await _register_token(harness.registry, session_id=harness.session_id, tool_use_id="tuid-1")
    assert harness.external_uq_state.get_active("tuid-1") is not None
    assert isinstance(await harness.registry.resolve(token, user_id=42), UserQuestionCallbackResolved)

    await harness.input_service.invalidate_binding(harness.session_id, reason="session_end")

    assert harness.external_uq_state.get_active("tuid-1") is None
    resolved = await harness.registry.resolve(token, user_id=42)
    assert isinstance(resolved, UserQuestionCallbackNotFound)


@pytest.mark.asyncio
async def test_rebind_aba_clears_old_binding_pending_and_token(harness: _Harness) -> None:
    new_binding_id = "bind-2"
    harness.external_uq_state.store(_pending(harness.session_id, harness.binding_id, tool_use_id="old"))
    new_pending = _pending(harness.session_id, new_binding_id, tool_use_id="new")
    harness.external_uq_state.store(new_pending)
    old_token = await _register_token(harness.registry, session_id=harness.session_id, tool_use_id="old")

    await harness.input_service.rebind_aba(harness.session_id, new_binding_id)

    # old binding_id pending is gone; new binding_id pending survives
    assert harness.external_uq_state.get_active("old") is None
    assert harness.external_uq_state.get_active("new") is not None
    resolved = await harness.registry.resolve(old_token, user_id=42)
    assert isinstance(resolved, UserQuestionCallbackNotFound)


class _NoopHook:
    async def cancel_pending_permissions(self, *, session_id: str) -> None:
        return None


def _make_reaper(harness: _Harness) -> ExternalBindingReaper:
    """Construct the reaper the invalidation tests share.

    Optional collaborators default to None inside the reaper; passing them
    explicitly here keeps the three reaper-based tests identical and makes it
    obvious none of them wire a discovery/tombstone/permission collaborator.
    """
    return ExternalBindingReaper(
        binding_store=harness.binding_store,
        auto_approve_service=AutoApproveService(),
        hook_socket_server=_NoopHook(),  # type: ignore[arg-type]
        external_uq_state=harness.external_uq_state,
        user_question_callback_registry=harness.registry,
    )


@pytest.mark.asyncio
async def test_reaper_manual_unbind_clears_pending_and_token(harness: _Harness) -> None:
    harness.external_uq_state.store(_pending(harness.session_id, harness.binding_id))
    token = await _register_token(harness.registry, session_id=harness.session_id, tool_use_id="tuid-1")

    reaper = _make_reaper(harness)
    await reaper.remove_with_cleanup(harness.session_id, reason="manual_unbind")

    assert harness.external_uq_state.get_active("tuid-1") is None
    resolved = await harness.registry.resolve(token, user_id=42)
    assert isinstance(resolved, UserQuestionCallbackNotFound)


@pytest.mark.asyncio
async def test_reaper_pid_dead_clears_pending_and_token(harness: _Harness) -> None:
    harness.external_uq_state.store(_pending(harness.session_id, harness.binding_id))
    token = await _register_token(harness.registry, session_id=harness.session_id, tool_use_id="tuid-1")

    reaper = _make_reaper(harness)
    await reaper.remove_with_cleanup(harness.session_id, reason="pid_dead")

    assert harness.external_uq_state.get_active("tuid-1") is None
    assert isinstance(await harness.registry.resolve(token, user_id=42), UserQuestionCallbackNotFound)


@pytest.mark.asyncio
async def test_reaper_idle_ttl_keeps_pending_by_design(harness: _Harness) -> None:
    """idle_ttl_expired deliberately does not invalidate question state (design H)."""
    harness.external_uq_state.store(_pending(harness.session_id, harness.binding_id))
    token = await _register_token(harness.registry, session_id=harness.session_id, tool_use_id="tuid-1")

    reaper = _make_reaper(harness)
    await reaper.remove_with_cleanup(harness.session_id, reason="idle_ttl_expired")

    # By design, idle expiry keeps question state and tokens.
    assert harness.external_uq_state.get_active("tuid-1") is not None
    assert isinstance(await harness.registry.resolve(token, user_id=42), UserQuestionCallbackResolved)
