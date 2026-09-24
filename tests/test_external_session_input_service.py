"""Unit tests for ExternalSessionInputService.

Covers pairing/owner/ABA checks, send state gates, argv-safe adapter outcomes,
per-session serialisation, busy FIFO drain, and lifecycle cleanup. Real Ghostty,
TCC, PTYs and process tables are not required; fakes exercise the service
orchestration while the adapter/probe have their own focused tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest

from app.adapters.process.ghostty_terminal_adapter import GhosttyTerminal, InjectionOutcome
from app.domain.external_session_models import ExternalBinding, GhosttyInputTarget
from app.domain.models import utc_now
from app.domain.session_models import PendingPermission, SessionPhase
from app.domain.user_question_models import (
    ExternalGhosttyQuestionTarget,
    ExternalQuestionActionStatus,
    ExternalUserQuestionContext,
    ExternalUserQuestionPhase,
    UserQuestionOption,
    UserQuestionPrompt,
)
from app.services.external_session_input_service import (
    ExternalSessionInputService,
    PairOutcome,
    SendOutcome,
)
from app.services.external_user_question_state import PendingExternalUserQuestion
from tests.fakes.external_input_harness import InputHarness, build_input_harness
from tests.fakes.ghostty import FakeGhosttyTerminalAdapter
from tests.fakes.process_probe import FakeLocalProcessProbe


@pytest.fixture
async def make_harness(tmp_path: Path):
    services: list[ExternalSessionInputService] = []
    counter = 0

    def _make(
        *,
        session_id: str = "session-1",
        user_id: int = 42,
        phase: SessionPhase = SessionPhase.IDLE,
        paired: bool = True,
        enabled: bool = True,
        adapter: FakeGhosttyTerminalAdapter | None = None,
        probe: FakeLocalProcessProbe | None = None,
        queue_max_size: int = 5,
        drain_wait: float = 0.05,
        in_flight_ttl: float = 2.0,
        monotonic: Callable[[], float] | None = None,
    ) -> InputHarness:
        nonlocal counter
        counter += 1
        harness = build_input_harness(
            tmp_path,
            name=f"h-{counter}",
            session_id=session_id,
            user_id=user_id,
            phase=phase,
            paired=paired,
            enabled=enabled,
            adapter=adapter,
            probe=probe,
            queue_max_size=queue_max_size,
            drain_wait=drain_wait,
            in_flight_ttl=in_flight_ttl,
            monotonic=monotonic,
        )
        services.append(harness.service)
        return harness

    yield _make

    await asyncio.gather(*(service.shutdown() for service in services), return_exceptions=True)


async def _activate(harness: InputHarness, *, user_id: int = 42) -> None:
    await harness.mode_store.set_target(
        user_id=user_id,
        session_id=harness.binding.session_id,
        binding_id=harness.binding.binding_id,
    )


def _seed_external_question(
    harness: InputHarness,
    *,
    tool_use_id: str = "tool-question",
    multi_select: bool = False,
) -> ExternalUserQuestionContext:
    target = harness.binding.ghostty_target
    assert target is not None
    prompt = UserQuestionPrompt(
        tool_use_id=tool_use_id,
        question_index=0,
        total_questions=1,
        question="Pick one",
        options=(
            UserQuestionOption(label="A"),
            UserQuestionOption(label="B"),
        ),
        multi_select=multi_select,
    )
    state = harness.session_store.get(harness.binding.session_id)
    assert state is not None
    state.phase = SessionPhase.WAITING_FOR_APPROVAL
    state.pending_permission = PendingPermission(
        tool_use_id=tool_use_id,
        tool_name="AskUserQuestion",
        tool_input={
            "questions": [
                {
                    "question": prompt.question,
                    "options": [{"label": option.label} for option in prompt.options],
                    "multiSelect": multi_select,
                }
            ]
        },
    )
    harness.session_store.save(state)
    ghostty_target = ExternalGhosttyQuestionTarget(
        binding_id=target.binding_id,
        terminal_id=target.terminal_id,
        paired_tty=target.paired_tty,
        paired_at=target.paired_at,
    )
    harness.external_uq_state.store(
        PendingExternalUserQuestion(
            tool_use_id=tool_use_id,
            session_id=harness.binding.session_id,
            user_id=harness.binding.user_id,
            prompts=(prompt,),
            target=ghostty_target,
        )
    )
    return ExternalUserQuestionContext(
        tool_use_id=tool_use_id,
        session_id=harness.binding.session_id,
        user_id=harness.binding.user_id,
        target=ghostty_target,
    )


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 1.0) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout=timeout)


# ─── pairing ---------------------------------------------------------------


async def test_pair_candidates_requires_owner_live_process_and_adapter(make_harness) -> None:
    harness = make_harness(paired=False)

    outcome, candidates = await harness.service.pair_candidates(user_id=7, session_id="session-1")
    assert outcome is PairOutcome.NOT_OWNER and candidates is None

    harness.adapter.available = False
    outcome, candidates = await harness.service.pair_candidates(user_id=42, session_id="session-1")
    assert outcome is PairOutcome.ADAPTER_UNAVAILABLE and candidates is None

    harness.adapter.available = True
    harness.probe.valid = False
    outcome, candidates = await harness.service.pair_candidates(user_id=42, session_id="session-1")
    assert outcome is PairOutcome.PROCESS_INVALID and candidates is None

    harness.probe.valid = True
    outcome, candidates = await harness.service.pair_candidates(user_id=42, session_id="session-1")
    assert outcome is PairOutcome.NEEDS_PAIRING
    assert candidates is not None
    assert candidates.binding_id == harness.binding.binding_id
    assert candidates.paired_tty == "/dev/ttys005"
    assert [terminal.terminal_id for terminal in candidates.terminals] == ["term-1"]


async def test_pair_candidates_prioritises_exact_cwd_without_auto_selecting(make_harness) -> None:
    adapter = FakeGhosttyTerminalAdapter(
        terminals=[
            GhosttyTerminal(terminal_id="other-1", name="Claude Code", cwd="/other"),
            GhosttyTerminal(terminal_id="match-claude", name="Claude Code", cwd="/project"),
            GhosttyTerminal(terminal_id="match-shell", name="~/project", cwd="/project"),
            GhosttyTerminal(terminal_id="other-2", name="shell", cwd="/elsewhere"),
        ]
    )
    harness = make_harness(paired=False, adapter=adapter)

    outcome, candidates = await harness.service.pair_candidates(user_id=42, session_id="session-1")

    assert outcome is PairOutcome.NEEDS_PAIRING
    assert candidates is not None
    assert [terminal.terminal_id for terminal in candidates.terminals] == [
        "match-claude",
        "match-shell",
        "other-1",
        "other-2",
    ]


async def test_pair_candidates_resolves_tty_from_pid_when_binding_missing_tty(make_harness) -> None:
    harness = make_harness(paired=False)
    harness.binding.tty = None
    harness.binding_store.save_binding(harness.binding)

    outcome, candidates = await harness.service.pair_candidates(user_id=42, session_id="session-1")
    assert outcome is PairOutcome.NEEDS_PAIRING
    assert candidates is not None and candidates.paired_tty == "/dev/ttys005"
    assert harness.probe.tty_calls == [1234]


async def test_pair_token_registration_enforces_displayed_binding_generation(make_harness) -> None:
    harness = make_harness(paired=False)
    stale = await harness.service.register_pair_token(
        user_id=42,
        session_id="session-1",
        expected_binding_id="old-generation",
        terminal_id="term-1",
    )
    assert stale is None

    token = await harness.service.register_pair_token(
        user_id=42,
        session_id="session-1",
        expected_binding_id=harness.binding.binding_id,
        terminal_id="term-1",
    )
    assert token is not None


async def test_consume_pair_token_persists_snapshot_and_activates(make_harness) -> None:
    harness = make_harness(paired=False)
    token = await harness.service.register_pair_token(
        user_id=42,
        session_id="session-1",
        expected_binding_id=harness.binding.binding_id,
        terminal_id="term-1",
    )
    assert token is not None

    outcome = await harness.service.consume_pair_token(token=token, user_id=42)
    assert outcome is PairOutcome.PAIRED
    binding = harness.binding_store.get_binding("session-1")
    assert binding is not None and binding.ghostty_target is not None
    assert binding.ghostty_target.terminal_id == "term-1"
    assert binding.ghostty_target.name == "claude — project"
    assert binding.ghostty_target.cwd == "/project"
    active = await harness.mode_store.get_target(42)
    assert active is not None and active.binding_id == harness.binding.binding_id


async def test_consume_pair_token_refuses_other_user_and_stale_generation(make_harness) -> None:
    harness = make_harness(paired=False)
    token = await harness.service.register_pair_token(
        user_id=42,
        session_id="session-1",
        expected_binding_id=harness.binding.binding_id,
        terminal_id="term-1",
    )
    assert token is not None
    assert await harness.service.consume_pair_token(token=token, user_id=7) is PairOutcome.TOKEN_UNAUTHORIZED

    # Unauthorized consume does not burn the token. Replace the binding before
    # the owner consumes: the token's old binding_id is now stale (ABA).
    replacement = ExternalBinding(
        session_id="session-1",
        user_id=42,
        cwd="/project",
        bound_at=utc_now(),
        jsonl_path=None,
        binding_id="new-generation",
        pid=1234,
        tty="/dev/ttys005",
    )
    harness.binding_store.save_binding(replacement)
    assert await harness.service.consume_pair_token(token=token, user_id=42) is PairOutcome.BINDING_STALE


async def test_consume_pair_token_honors_store_generation_safe_setter(make_harness, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = make_harness(paired=False)
    token = await harness.service.register_pair_token(
        user_id=42,
        session_id="session-1",
        expected_binding_id=harness.binding.binding_id,
        terminal_id="term-1",
    )
    assert token is not None
    monkeypatch.setattr(harness.binding_store, "set_ghostty_target", lambda *_args, **_kwargs: False)

    outcome = await harness.service.consume_pair_token(token=token, user_id=42)
    assert outcome is PairOutcome.BINDING_STALE
    assert await harness.mode_store.get_target(42) is None


async def test_activate_select_validates_persisted_target(make_harness) -> None:
    harness = make_harness()
    assert await harness.service.activate_select(user_id=42, session_id="session-1") is PairOutcome.ACTIVATED
    assert await harness.mode_store.get_target(42) is not None

    await harness.service.leave(user_id=42)
    harness.adapter.validate_error = InjectionOutcome.NOT_FOUND
    assert await harness.service.activate_select(user_id=42, session_id="session-1") is PairOutcome.NEEDS_PAIRING
    binding = harness.binding_store.get_binding("session-1")
    assert binding is not None and binding.ghostty_target is None


async def test_repair_same_terminal_invalidates_old_question_generation(make_harness) -> None:
    harness = make_harness(paired=True)
    context = _seed_external_question(harness)
    token = await harness.service.register_pair_token(
        user_id=42,
        session_id=harness.binding.session_id,
        expected_binding_id=harness.binding.binding_id,
        terminal_id="term-1",
    )
    assert token is not None

    outcome = await harness.service.consume_pair_token(token=token, user_id=42)

    assert outcome is PairOutcome.PAIRED
    assert harness.external_uq_state.get(context.tool_use_id) is None
    rebound = harness.binding_store.get_binding(harness.binding.session_id)
    assert rebound is not None and rebound.ghostty_target is not None
    assert rebound.ghostty_target.paired_at != context.target.paired_at


# ─── AskUserQuestion transport ---------------------------------------------


async def test_question_select_does_not_require_active_input_mode(make_harness) -> None:
    harness = make_harness(paired=True)
    context = _seed_external_question(harness)
    assert await harness.service.has_target(42) is False

    result = await harness.service.select_option(
        context=context,
        question_index=0,
        option_count=2,
        option_index=1,
        submit_after=False,
    )

    assert result.status is ExternalQuestionActionStatus.APPLIED
    assert harness.adapter.question_calls == [("term-1", "select", 2, 1, False, "")]
    assert harness.external_uq_state.get_active(context.tool_use_id) is not None


async def test_final_question_uses_two_phase_completion(make_harness) -> None:
    harness = make_harness(paired=True)
    context = _seed_external_question(harness)

    result = await harness.service.select_option(
        context=context,
        question_index=0,
        option_count=2,
        option_index=0,
        submit_after=True,
    )

    assert result.status is ExternalQuestionActionStatus.APPLIED
    pending = harness.external_uq_state.get(context.tool_use_id)
    assert pending is not None and pending.phase is ExternalUserQuestionPhase.TERMINAL_ACTION_APPLIED

    await harness.service.question_completed(context=context)
    pending = harness.external_uq_state.get(context.tool_use_id)
    assert pending is not None and pending.phase is ExternalUserQuestionPhase.COMPLETED


async def test_question_rejects_owner_and_process_mismatch_without_adapter_action(make_harness) -> None:
    harness = make_harness(paired=True)
    context = _seed_external_question(harness)

    wrong_owner = ExternalUserQuestionContext(
        tool_use_id=context.tool_use_id,
        session_id=context.session_id,
        user_id=7,
        target=context.target,
    )
    result = await harness.service.select_option(
        context=wrong_owner,
        question_index=0,
        option_count=2,
        option_index=0,
        submit_after=False,
    )
    assert result.status is ExternalQuestionActionStatus.REJECTED

    harness.probe.valid = False
    result = await harness.service.select_option(
        context=context,
        question_index=0,
        option_count=2,
        option_index=0,
        submit_after=False,
    )
    assert result.status is ExternalQuestionActionStatus.REJECTED
    assert harness.adapter.question_calls == []


async def test_question_rechecks_target_after_terminal_validation_await(make_harness) -> None:
    harness = make_harness(paired=True)
    context = _seed_external_question(harness)
    harness.adapter.validate_entered = asyncio.Event()
    harness.adapter.validate_release = asyncio.Event()

    task = asyncio.create_task(
        harness.service.select_option(
            context=context,
            question_index=0,
            option_count=2,
            option_index=0,
            submit_after=False,
        )
    )
    await asyncio.wait_for(harness.adapter.validate_entered.wait(), timeout=1)
    old_target = harness.binding.ghostty_target
    assert old_target is not None
    harness.binding.ghostty_target = GhosttyInputTarget(
        terminal_id=old_target.terminal_id,
        paired_tty=old_target.paired_tty,
        paired_at=old_target.paired_at + timedelta(seconds=1),
        binding_id=old_target.binding_id,
        name=old_target.name,
        cwd=old_target.cwd,
    )
    harness.binding_store.save_binding(harness.binding)
    harness.adapter.validate_release.set()

    result = await task
    assert result.status is ExternalQuestionActionStatus.REJECTED
    assert harness.adapter.question_calls == []


async def test_question_indeterminate_blocks_retry(make_harness) -> None:
    harness = make_harness(paired=True)
    context = _seed_external_question(harness)
    harness.adapter.question_outcomes.append(InjectionOutcome.INDETERMINATE)

    first = await harness.service.select_option(
        context=context,
        question_index=0,
        option_count=2,
        option_index=0,
        submit_after=False,
    )
    second = await harness.service.select_option(
        context=context,
        question_index=0,
        option_count=2,
        option_index=0,
        submit_after=False,
    )

    assert first.status is ExternalQuestionActionStatus.INDETERMINATE
    assert second.status is ExternalQuestionActionStatus.REJECTED
    assert len(harness.adapter.question_calls) == 1
    pending = harness.external_uq_state.get(context.tool_use_id)
    assert pending is not None and pending.phase is ExternalUserQuestionPhase.INDETERMINATE


async def test_question_text_and_multi_advance_preserve_typed_actions(make_harness) -> None:
    harness = make_harness(paired=True)
    text_context = _seed_external_question(harness, tool_use_id="tool-text")
    answer = "  自由文本\n第二行  "

    text_result = await harness.service.answer_with_text(
        context=text_context,
        question_index=0,
        option_count=2,
        text=answer,
        submit_after=False,
    )
    assert text_result.status is ExternalQuestionActionStatus.APPLIED
    assert harness.adapter.question_calls[-1] == ("term-1", "answer_text", 2, -1, False, answer)

    harness.external_uq_state.invalidate_tool("tool-text")
    multi_context = _seed_external_question(harness, tool_use_id="tool-multi", multi_select=True)
    multi_result = await harness.service.advance_after_multi_select(
        context=multi_context,
        question_index=0,
        option_count=2,
        final_question=False,
    )
    assert multi_result.status is ExternalQuestionActionStatus.APPLIED
    assert harness.adapter.question_calls[-1] == ("term-1", "advance_multi", 2, -1, False, "")


# ─── send ------------------------------------------------------------------


async def test_send_requires_active_target(make_harness) -> None:
    harness = make_harness()
    assert await harness.service.send_text(user_id=42, text="hello") is SendOutcome.NO_TARGET
    assert harness.adapter.inject_calls == []


async def test_send_normalises_newlines_and_injects(make_harness) -> None:
    harness = make_harness()
    await _activate(harness)

    outcome = await harness.service.send_text(user_id=42, text="line1\r\nline2\rline3")
    assert outcome is SendOutcome.SENT
    assert harness.adapter.inject_calls == [("term-1", "line1\nline2\nline3")]
    assert harness.probe.validation_calls[-1] == (1234, "/dev/ttys005")
    assert len(harness.probe.validation_calls) == 2
    assert harness.adapter.validate_calls[-1] == "term-1"


async def test_send_revalidates_process_after_terminal_await(make_harness) -> None:
    probe = FakeLocalProcessProbe()
    probe.validation_results.extend([True, False])
    harness = make_harness(probe=probe)
    await _activate(harness)

    outcome = await harness.service.send_text(user_id=42, text="must not hit shell")
    assert outcome is SendOutcome.PROCESS_INVALID
    assert len(probe.validation_calls) == 2
    assert harness.adapter.inject_calls == []


async def test_send_rechecks_binding_generation_after_terminal_await(make_harness) -> None:
    adapter = FakeGhosttyTerminalAdapter()
    adapter.validate_entered = asyncio.Event()
    adapter.validate_release = asyncio.Event()
    harness = make_harness(adapter=adapter)
    await _activate(harness)

    send_task = asyncio.create_task(harness.service.send_text(user_id=42, text="stale generation"))
    await asyncio.wait_for(adapter.validate_entered.wait(), timeout=1)
    replacement = ExternalBinding(
        session_id="session-1",
        user_id=7,
        cwd="/other",
        bound_at=utc_now(),
        jsonl_path=None,
        binding_id="replacement-generation",
        pid=9999,
        tty="/dev/ttys099",
    )
    harness.binding_store.save_binding(replacement)
    adapter.validate_release.set()

    assert await send_task is SendOutcome.BINDING_STALE
    assert adapter.inject_calls == []


@pytest.mark.parametrize(
    "phase",
    [
        SessionPhase.PROCESSING,
        SessionPhase.COMPACTING,
        SessionPhase.WAITING_FOR_APPROVAL,
    ],
)
async def test_busy_phases_queue_without_injection(make_harness, phase: SessionPhase) -> None:
    harness = make_harness(phase=phase)
    await _activate(harness)

    outcome = await harness.service.send_text(user_id=42, text="queued")
    assert outcome is SendOutcome.QUEUED
    assert harness.adapter.inject_calls == []
    assert await harness.queue.peek_size("session-1") == 1


async def test_ended_state_refuses_and_clears_mode(make_harness) -> None:
    harness = make_harness(phase=SessionPhase.ENDED)
    await _activate(harness)

    assert await harness.service.send_text(user_id=42, text="late") is SendOutcome.SESSION_ENDED
    assert await harness.mode_store.get_target(42) is None
    assert harness.adapter.inject_calls == []


async def test_pending_permission_and_user_question_queue(make_harness) -> None:
    harness = make_harness()
    await _activate(harness)
    state = harness.session_store.get("session-1")
    assert state is not None

    state.pending_permission = PendingPermission(tool_use_id="tool-1", tool_name="Bash")
    harness.session_store.save(state)
    assert await harness.service.send_text(user_id=42, text="after permission") is SendOutcome.QUEUED

    await harness.queue.clear("session-1")
    state.pending_permission = None
    state.structured_user_question_key = "tool-uq:0"
    harness.session_store.save(state)
    assert await harness.service.send_text(user_id=42, text="after question") is SendOutcome.QUEUED
    assert harness.adapter.inject_calls == []


async def test_rapid_second_send_queues_behind_in_flight(make_harness) -> None:
    harness = make_harness()
    await _activate(harness)

    assert await harness.service.send_text(user_id=42, text="first") is SendOutcome.SENT
    assert await harness.service.send_text(user_id=42, text="second") is SendOutcome.QUEUED
    assert harness.adapter.inject_calls == [("term-1", "first")]
    assert await harness.queue.peek_size("session-1") == 1


async def test_queue_full_refuses_new_message(make_harness) -> None:
    harness = make_harness(phase=SessionPhase.PROCESSING, queue_max_size=1)
    await _activate(harness)
    assert await harness.service.send_text(user_id=42, text="one") is SendOutcome.QUEUED
    assert await harness.service.send_text(user_id=42, text="two") is SendOutcome.QUEUE_FULL
    assert await harness.queue.peek_size("session-1") == 1


async def test_process_failure_refuses_without_queueing(make_harness) -> None:
    probe = FakeLocalProcessProbe(valid=False)
    harness = make_harness(probe=probe)
    await _activate(harness)

    assert await harness.service.send_text(user_id=42, text="unsafe") is SendOutcome.PROCESS_INVALID
    assert await harness.queue.peek_size("session-1") == 0
    assert harness.adapter.inject_calls == []


async def test_missing_terminal_clears_pairing_mode_and_queue(make_harness) -> None:
    harness = make_harness(phase=SessionPhase.PROCESSING)
    await _activate(harness)
    assert await harness.service.send_text(user_id=42, text="queued") is SendOutcome.QUEUED
    harness.adapter.validate_error = InjectionOutcome.NOT_FOUND

    assert await harness.service.send_text(user_id=42, text="new") is SendOutcome.TERMINAL_INVALID
    binding = harness.binding_store.get_binding("session-1")
    assert binding is not None and binding.ghostty_target is None
    assert await harness.mode_store.get_target(42) is None
    assert await harness.queue.peek_size("session-1") == 0


async def test_adapter_unavailable_preserves_pairing(make_harness) -> None:
    harness = make_harness()
    await _activate(harness)
    harness.adapter.available = False

    assert await harness.service.send_text(user_id=42, text="hello") is SendOutcome.ADAPTER_UNAVAILABLE
    binding = harness.binding_store.get_binding("session-1")
    assert binding is not None and binding.ghostty_target is not None
    assert await harness.mode_store.get_target(42) is not None


async def test_indeterminate_injection_is_not_retried(make_harness) -> None:
    harness = make_harness()
    await _activate(harness)
    harness.adapter.inject_outcomes.append(InjectionOutcome.INDETERMINATE)

    assert await harness.service.send_text(user_id=42, text="maybe pasted") is SendOutcome.INJECTION_INDETERMINATE
    assert harness.adapter.inject_calls == [("term-1", "maybe pasted")]
    assert await harness.queue.peek_size("session-1") == 0


async def test_active_binding_aba_refuses_and_clears_mode(make_harness) -> None:
    harness = make_harness()
    await _activate(harness)
    harness.binding.binding_id = "new-generation"

    assert await harness.service.send_text(user_id=42, text="stale") is SendOutcome.BINDING_STALE
    assert await harness.mode_store.get_target(42) is None
    assert harness.adapter.inject_calls == []


# ─── locking and drain ------------------------------------------------------


async def test_same_session_injection_serialized_second_message_queues(make_harness) -> None:
    adapter = FakeGhosttyTerminalAdapter()
    adapter.inject_entered = asyncio.Event()
    adapter.inject_release = asyncio.Event()
    harness = make_harness(adapter=adapter)
    await _activate(harness)

    first = asyncio.create_task(harness.service.send_text(user_id=42, text="first"))
    await asyncio.wait_for(adapter.inject_entered.wait(), timeout=1)
    second = asyncio.create_task(harness.service.send_text(user_id=42, text="second"))
    await asyncio.sleep(0.02)
    assert not second.done(), "second waits on the per-session input lock"

    adapter.inject_release.set()
    assert await first is SendOutcome.SENT
    assert await second is SendOutcome.QUEUED
    assert adapter.max_active_injections == 1


async def test_stop_arriving_during_injection_cannot_release_new_turn_guard(make_harness) -> None:
    adapter = FakeGhosttyTerminalAdapter()
    adapter.inject_entered = asyncio.Event()
    adapter.inject_release = asyncio.Event()
    harness = make_harness(adapter=adapter, drain_wait=0.2)
    await _activate(harness)

    first = asyncio.create_task(harness.service.send_text(user_id=42, text="first"))
    await asyncio.wait_for(adapter.inject_entered.wait(), timeout=1)
    # This may be a late Stop from the prior turn. Because AppleScript for the
    # new turn is still in progress, it must not clear the new guard.
    await harness.service.notify_hook_event(session_id="session-1", event_kind="Stop")
    adapter.inject_release.set()
    assert await first is SendOutcome.SENT

    adapter.inject_entered = None
    adapter.inject_release = None
    assert await harness.service.send_text(user_id=42, text="second") is SendOutcome.QUEUED
    assert adapter.inject_calls == [("term-1", "first")]

    await harness.service.notify_hook_event(session_id="session-1", event_kind="Stop")
    await _wait_until(lambda: len(adapter.inject_calls) == 2)
    assert adapter.inject_calls[-1] == ("term-1", "second")


async def test_different_sessions_inject_concurrently(make_harness) -> None:
    adapter = FakeGhosttyTerminalAdapter()
    adapter.inject_entered = asyncio.Event()
    adapter.inject_release = asyncio.Event()
    one = make_harness(session_id="session-a", user_id=1, adapter=adapter)
    two = make_harness(session_id="session-b", user_id=2, adapter=adapter)
    await _activate(one, user_id=1)
    await _activate(two, user_id=2)

    task_a = asyncio.create_task(one.service.send_text(user_id=1, text="a"))
    task_b = asyncio.create_task(two.service.send_text(user_id=2, text="b"))
    await _wait_until(lambda: len(adapter.inject_calls) == 2)
    assert adapter.max_active_injections == 2
    adapter.inject_release.set()
    assert await task_a is SendOutcome.SENT
    assert await task_b is SendOutcome.SENT


async def test_busy_queue_drains_after_state_publish(make_harness) -> None:
    harness = make_harness(phase=SessionPhase.PROCESSING, drain_wait=0.2)
    await _activate(harness)
    assert await harness.service.send_text(user_id=42, text="later") is SendOutcome.QUEUED

    state = harness.session_store.get("session-1")
    assert state is not None
    state.phase = SessionPhase.IDLE
    harness.session_store.save(state)
    await _wait_until(lambda: harness.adapter.inject_calls == [("term-1", "later")])
    assert await harness.queue.peek_size("session-1") == 0


async def test_inflight_fifo_drains_one_item_per_stop(make_harness) -> None:
    harness = make_harness(drain_wait=0.2)
    await _activate(harness)
    assert await harness.service.send_text(user_id=42, text="one") is SendOutcome.SENT
    assert await harness.service.send_text(user_id=42, text="two") is SendOutcome.QUEUED
    assert await harness.service.send_text(user_id=42, text="three") is SendOutcome.QUEUED

    await harness.service.notify_hook_event(session_id="session-1", event_kind="Stop")
    await _wait_until(lambda: len(harness.adapter.inject_calls) == 2)
    assert harness.adapter.inject_calls[-1] == ("term-1", "two")
    assert await harness.queue.peek_size("session-1") == 1

    await harness.service.notify_hook_event(session_id="session-1", event_kind="Stop")
    await _wait_until(lambda: len(harness.adapter.inject_calls) == 3)
    assert [text for _terminal, text in harness.adapter.inject_calls] == ["one", "two", "three"]
    assert await harness.queue.peek_size("session-1") == 0


async def test_busy_wait_does_not_drop_queued_input(make_harness) -> None:
    harness = make_harness(phase=SessionPhase.PROCESSING, drain_wait=0.01)
    await _activate(harness)
    assert await harness.service.send_text(user_id=42, text="keep me") is SendOutcome.QUEUED
    await asyncio.sleep(0.05)  # several drain timeouts while still busy
    assert await harness.queue.peek_size("session-1") == 1
    assert harness.adapter.inject_calls == []

    state = harness.session_store.get("session-1")
    assert state is not None
    state.phase = SessionPhase.IDLE
    harness.session_store.save(state)
    await harness.service.notify_hook_event(session_id="session-1", event_kind="turn_completed")
    await _wait_until(lambda: harness.adapter.inject_calls == [("term-1", "keep me")])


async def test_drain_revalidates_after_terminal_await_and_restores_fifo(make_harness) -> None:
    probe = FakeLocalProcessProbe(valid=False)
    # send enqueue precheck succeeds; drain precheck succeeds; final check
    # observes shell takeover and must restore the dequeued entry.
    probe.validation_results.extend([True, True, False])
    harness = make_harness(
        phase=SessionPhase.PROCESSING,
        probe=probe,
        drain_wait=0.2,
    )
    await _activate(harness)
    assert await harness.service.send_text(user_id=42, text="keep queued") is SendOutcome.QUEUED

    state = harness.session_store.get("session-1")
    assert state is not None
    state.phase = SessionPhase.IDLE
    harness.session_store.save(state)
    await _wait_until(lambda: len(probe.validation_calls) >= 3)
    assert harness.adapter.inject_calls == []
    assert await harness.queue.peek_size("session-1") == 1

    probe.valid = True
    await harness.service.notify_hook_event(session_id="session-1", event_kind="Stop")
    await _wait_until(lambda: harness.adapter.inject_calls == [("term-1", "keep queued")])


async def test_drain_rechecks_binding_generation_after_terminal_await(make_harness) -> None:
    adapter = FakeGhosttyTerminalAdapter()
    harness = make_harness(
        phase=SessionPhase.PROCESSING,
        adapter=adapter,
        drain_wait=0.2,
    )
    await _activate(harness)
    assert await harness.service.send_text(user_id=42, text="old generation") is SendOutcome.QUEUED

    adapter.validate_entered = asyncio.Event()
    adapter.validate_release = asyncio.Event()
    state = harness.session_store.get("session-1")
    assert state is not None
    state.phase = SessionPhase.IDLE
    harness.session_store.save(state)
    await asyncio.wait_for(adapter.validate_entered.wait(), timeout=1)

    replacement = ExternalBinding(
        session_id="session-1",
        user_id=42,
        cwd="/project",
        bound_at=utc_now(),
        jsonl_path=None,
        binding_id="replacement-generation",
        pid=2222,
        tty="/dev/ttys009",
    )
    harness.binding_store.save_binding(replacement)
    adapter.validate_release.set()

    async def _wait_queue_cleared() -> None:
        while await harness.queue.peek_size("session-1") != 0:
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_wait_queue_cleared(), timeout=1)
    assert adapter.inject_calls == []


async def test_drain_preflight_failure_restores_head_without_drops(make_harness) -> None:
    """A preflight inject failure (nothing pasted) must keep the entry queued
    and notify nobody — the drain retries once Ghostty is reachable again."""
    adapter = FakeGhosttyTerminalAdapter()
    # Fail EVERY inject, not a fixed number of retries: the retry loop runs
    # ~20x/second, so a preset outcome count only buys wall-clock slack. With
    # a persistent failure the restored head is a steady state — the
    # assertions hold no matter how many retries elapsed.
    adapter.inject_error = InjectionOutcome.GHOSTTY_NOT_RUNNING
    harness = make_harness(phase=SessionPhase.PROCESSING, adapter=adapter)
    await _activate(harness)
    assert await harness.service.send_text(user_id=42, text="hold") is SendOutcome.QUEUED

    state = harness.session_store.get("session-1")
    assert state is not None
    state.phase = SessionPhase.IDLE
    harness.session_store.save(state)

    await _wait_until(lambda: len(harness.adapter.inject_calls) >= 2)
    assert await harness.queue.peek_size("session-1") == 1
    assert harness.notices == []


async def test_drain_preflight_restore_failure_pauses_remainder(make_harness) -> None:
    """A preflight failure whose head cannot be restored (the queue refilled to
    its cap) drops exactly one entry with one notice; the remainder pauses
    instead of being retried in a backoff-free loop that would emit one notice
    and one subprocess attempt per queued entry."""
    adapter = FakeGhosttyTerminalAdapter()
    adapter.inject_outcomes.append(InjectionOutcome.GHOSTTY_NOT_RUNNING)
    harness = make_harness(
        phase=SessionPhase.PROCESSING,
        adapter=adapter,
        queue_max_size=2,
        drain_wait=0.05,
    )
    await _activate(harness)
    assert await harness.service.send_text(user_id=42, text="first") is SendOutcome.QUEUED
    assert await harness.service.send_text(user_id=42, text="second") is SendOutcome.QUEUED

    adapter.inject_entered = asyncio.Event()
    adapter.inject_release = asyncio.Event()
    state = harness.session_store.get("session-1")
    assert state is not None
    state.phase = SessionPhase.IDLE
    harness.session_store.save(state)

    await asyncio.wait_for(adapter.inject_entered.wait(), timeout=1)
    # Refill the queue to its cap while the drain is inside inject_text, so
    # the preflight failure cannot restore the dequeued head.
    await harness.queue.enqueue("session-1", text="filler", binding_id=harness.binding.binding_id)
    adapter.inject_release.set()

    await _wait_until(lambda: any("恢复失败" in t for _, t in harness.notices))
    assert [t for _term, t in harness.adapter.inject_calls] == ["first"]
    assert any("已暂缓" in t for _, t in harness.notices)
    # Several drain_wait windows: the remainder must not be retried.
    await asyncio.sleep(0.2)
    assert [t for _term, t in harness.adapter.inject_calls] == ["first"]
    assert await harness.queue.peek_size("session-1") == 2


async def test_drain_dequeue_internal_stale_head_drop_is_counted(make_harness) -> None:
    """dequeue skips and discards a stale-generation head in front of a live
    entry; that internal drop must surface as a §9 notice, not vanish."""
    harness = make_harness(phase=SessionPhase.PROCESSING)
    await _activate(harness)
    # Head from a superseded binding generation; tail from the live one
    # (send_text also spins up the drain while PROCESSING).
    await harness.queue.enqueue("session-1", text="stale", binding_id="old-generation")
    assert await harness.service.send_text(user_id=42, text="live") is SendOutcome.QUEUED

    state = harness.session_store.get("session-1")
    assert state is not None
    state.phase = SessionPhase.IDLE
    harness.session_store.save(state)

    await _wait_until(lambda: [t for _term, t in harness.adapter.inject_calls] == ["live"])
    assert any("已丢弃 1 条过期或失效" in t for _, t in harness.notices)


async def test_drop_notice_does_not_report_expired_remainder_as_paused(make_harness) -> None:
    """The "paused N" remainder count must only include live entries: an
    expired remainder is dropped (and reported) instead of being promised as
    queued in the very same notice (§9)."""
    clock = {"t": 100.0}
    adapter = FakeGhosttyTerminalAdapter()
    adapter.inject_outcomes.append(InjectionOutcome.INDETERMINATE)
    harness = make_harness(
        phase=SessionPhase.PROCESSING,
        adapter=adapter,
        monotonic=lambda: clock["t"],
        drain_wait=0.05,
    )
    await _activate(harness)
    assert await harness.service.send_text(user_id=42, text="first") is SendOutcome.QUEUED
    clock["t"] += 31.0  # "second" expires 31s later than "first"
    await harness.queue.enqueue("session-1", text="second", binding_id=harness.binding.binding_id)
    adapter.inject_entered = asyncio.Event()
    adapter.inject_release = asyncio.Event()

    state = harness.session_store.get("session-1")
    assert state is not None
    state.phase = SessionPhase.IDLE
    harness.session_store.save(state)

    await asyncio.wait_for(adapter.inject_entered.wait(), timeout=1)
    clock["t"] += 91.0  # "second" (ttl 60) expires while the inject hangs
    adapter.inject_release.set()

    await _wait_until(lambda: any("已丢弃" in t for _, t in harness.notices))
    assert any("已丢弃 1 条等待超时" in t for _, t in harness.notices)
    assert not any("已暂缓" in t for _, t in harness.notices)
    assert await harness.queue.peek_size("session-1") == 0


async def test_send_text_session_ended_clear_reports_dropped_count(make_harness) -> None:
    """send_text's SESSION_ENDED path clears the queue; the discarded count
    must be reported instead of vanishing silently (§9)."""
    harness = make_harness(phase=SessionPhase.PROCESSING)
    await _activate(harness)
    await harness.queue.enqueue("session-1", text="q1", binding_id=harness.binding.binding_id)
    await harness.queue.enqueue("session-1", text="q2", binding_id=harness.binding.binding_id)

    state = harness.session_store.get("session-1")
    assert state is not None
    state.phase = SessionPhase.ENDED
    harness.session_store.save(state)

    assert await harness.service.send_text(user_id=42, text="late") is SendOutcome.SESSION_ENDED
    await _wait_until(lambda: any("会话已结束" in t and "丢弃 2 条" in t for _, t in harness.notices))


async def test_send_text_terminal_invalid_clear_reports_dropped_count(make_harness) -> None:
    """send_text's TERMINAL_INVALID path clears the queue; the discarded count
    must be reported instead of vanishing silently (§9)."""
    adapter = FakeGhosttyTerminalAdapter()
    harness = make_harness(phase=SessionPhase.PROCESSING, adapter=adapter)
    await _activate(harness)
    await harness.queue.enqueue("session-1", text="q1", binding_id=harness.binding.binding_id)

    adapter.validate_error = InjectionOutcome.NOT_FOUND
    assert await harness.service.send_text(user_id=42, text="late") is SendOutcome.TERMINAL_INVALID
    await _wait_until(lambda: any("配对的终端已失效" in t and "丢弃 1 条" in t for _, t in harness.notices))


async def test_invalidate_notifies_every_target_owner(make_harness) -> None:
    """Several users may target the same session; every owner's dropped queue
    entry must be reported to them, not only the first cleared target."""
    harness = make_harness(phase=SessionPhase.PROCESSING)
    await _activate(harness)  # user 42 targets session-1
    await harness.mode_store.set_target(
        user_id=43,
        session_id="session-1",
        binding_id=harness.binding.binding_id,
    )
    await harness.queue.enqueue("session-1", text="q1", binding_id=harness.binding.binding_id)
    await harness.service.invalidate_binding("session-1", reason="test")
    await _wait_until(lambda: {uid for uid, _t in harness.notices} >= {42, 43})
    assert all("丢弃 1 条" in t for _uid, t in harness.notices)


async def test_drain_indeterminate_drops_only_current_and_pauses_remainder(make_harness) -> None:
    """An indeterminate injection drops only the dequeued head (it may have been
    pasted without Enter). The remainder must NOT be injected right away — the
    next paste would concatenate onto the unsubmitted line — and only a real
    session publish releases it."""
    adapter = FakeGhosttyTerminalAdapter()
    adapter.inject_outcomes.append(InjectionOutcome.INDETERMINATE)
    harness = make_harness(phase=SessionPhase.PROCESSING, adapter=adapter, drain_wait=0.05)
    await _activate(harness)
    assert await harness.service.send_text(user_id=42, text="first") is SendOutcome.QUEUED
    assert await harness.service.send_text(user_id=42, text="second") is SendOutcome.QUEUED

    state = harness.session_store.get("session-1")
    assert state is not None
    state.phase = SessionPhase.IDLE
    harness.session_store.save(state)  # publish → drain injects "first" → INDETERMINATE

    await _wait_until(lambda: [t for _term, t in harness.adapter.inject_calls] == ["first"])
    assert len(harness.notices) == 1
    uid, text = harness.notices[0]
    assert uid == 42
    assert "已丢弃" in text
    assert "indeterminate" in text
    assert "其余 1 条已暂缓" in text
    assert "残留文本" in text
    # Give the settle wait several drain_wait windows to misbehave: the paused
    # drain must not inject "second" onto the possibly-unsubmitted "first".
    await asyncio.sleep(0.2)
    assert [t for _term, t in harness.adapter.inject_calls] == ["first"]
    assert await harness.queue.peek_size("session-1") == 1

    # A real session publish (Claude produced output ⇒ the input line was
    # submitted or cleared) releases the paused drain.
    harness.session_store.save(state)
    await _wait_until(lambda: [t for _term, t in harness.adapter.inject_calls] == ["first", "second"])


async def test_drain_prunes_expired_heads_and_notifies(make_harness) -> None:
    clock = {"t": 100.0}
    harness = make_harness(phase=SessionPhase.PROCESSING, monotonic=lambda: clock["t"], drain_wait=0.05)
    await _activate(harness)
    assert await harness.service.send_text(user_id=42, text="stale") is SendOutcome.QUEUED
    clock["t"] += 61.0  # past ttl_sec=60

    state = harness.session_store.get("session-1")
    assert state is not None
    state.phase = SessionPhase.IDLE
    harness.session_store.save(state)

    await _wait_until(lambda: any("等待超时" in t for _, t in harness.notices))
    assert await harness.queue.peek_size("session-1") == 0


async def test_drain_entry_expiring_during_validation_notifies(make_harness) -> None:
    """An entry whose TTL lapses during the drain's validation awaits is
    silently dropped by dequeue — it must still surface as a notice."""
    clock = {"t": 100.0}
    adapter = FakeGhosttyTerminalAdapter()
    harness = make_harness(phase=SessionPhase.PROCESSING, adapter=adapter, monotonic=lambda: clock["t"], drain_wait=0.05)
    await _activate(harness)
    assert await harness.service.send_text(user_id=42, text="lapse") is SendOutcome.QUEUED
    # Hold validate_terminal open so the TTL crosses while the drain awaits.
    # Created after send_text: its own preflight also calls validate_terminal.
    adapter.validate_entered = asyncio.Event()
    adapter.validate_release = asyncio.Event()

    state = harness.session_store.get("session-1")
    assert state is not None
    state.phase = SessionPhase.IDLE
    harness.session_store.save(state)

    await asyncio.wait_for(adapter.validate_entered.wait(), timeout=1)
    clock["t"] += 61.0  # entry expires while the drain is inside validate_terminal
    adapter.validate_release.set()

    await _wait_until(lambda: any("过期或失效" in t for _, t in harness.notices))
    assert await harness.queue.peek_size("session-1") == 0
    assert harness.adapter.inject_calls == []


async def test_leave_and_invalidate_notify_discarded_counts(make_harness) -> None:
    harness = make_harness(phase=SessionPhase.PROCESSING)
    await _activate(harness)
    await harness.service.send_text(user_id=42, text="q1")
    await harness.service.send_text(user_id=42, text="q2")
    assert await harness.service.leave(user_id=42)
    assert any(user == 42 and "丢弃 2 条未发送的排队消息" in text for user, text in harness.notices)

    await _activate(harness)
    harness.notices.clear()
    await harness.queue.enqueue("session-1", text="queued3", binding_id=harness.binding.binding_id)
    await harness.service.invalidate_binding("session-1", reason="test")
    await _wait_until(lambda: any(user == 42 and "丢弃 1 条未发送的排队消息" in text for user, text in harness.notices))


async def test_invalidate_notice_delivery_runs_off_caller(make_harness) -> None:
    """invalidate_binding callers (SessionEnd / reaper) hold the external
    reply-delivery lock; the §9 notice must be delivered in the background so
    retrying Telegram I/O cannot stall the same session's Stop push."""
    harness = make_harness(phase=SessionPhase.PROCESSING)
    await _activate(harness)
    await harness.queue.enqueue("session-1", text="queued", binding_id=harness.binding.binding_id)

    release = asyncio.Event()
    original = harness.service._notify_user

    async def slow_notify(*, user_id: int, text: str) -> bool:
        await release.wait()
        assert original is not None
        return await original(user_id=user_id, text=text)

    harness.service._notify_user = slow_notify
    await harness.service.invalidate_binding("session-1", reason="test")
    # The caller returned while the notice was still in flight.
    assert harness.notices == []
    release.set()
    await _wait_until(lambda: any("丢弃 1 条" in text for _, text in harness.notices))


async def test_stop_drain_cancelling_mid_inject_reports_entry_in_hand(make_harness) -> None:
    """invalidate_binding calls _stop_drain without the input lock, so the
    drain task can be cancelled while a dequeued entry is in hand: that entry
    must be reported (design §9), not silently lost beside the caller's
    queue clear() which can no longer see it."""
    adapter = FakeGhosttyTerminalAdapter()
    adapter.inject_entered = asyncio.Event()
    adapter.inject_release = asyncio.Event()
    harness = make_harness(phase=SessionPhase.PROCESSING, adapter=adapter)
    await _activate(harness)
    assert await harness.service.send_text(user_id=42, text="doomed") is SendOutcome.QUEUED

    state = harness.session_store.get("session-1")
    state.phase = SessionPhase.IDLE
    harness.session_store.save(state)
    await _wait_until(lambda: adapter.inject_entered is not None and adapter.inject_entered.is_set())
    # The drain now holds the dequeued entry inside inject_text; cancel it
    # exactly like invalidate_binding does (without the input lock).
    await harness.service._stop_drain("session-1")

    await _wait_until(lambda: any("退出或切换输入模式" in text for _, text in harness.notices))
    assert await harness.queue.peek_size("session-1") == 0
    assert "session-1" not in harness.service._injecting
    adapter.inject_release.set()
    await harness.service.shutdown()


async def test_drain_cancellation_does_not_lose_pending_drop_notices(make_harness) -> None:
    """A §9 notice still being delivered when the drain task is cancelled must
    survive: delivery runs in its own task, not inside the drain loop."""
    adapter = FakeGhosttyTerminalAdapter()
    adapter.inject_outcomes.append(InjectionOutcome.INDETERMINATE)
    harness = make_harness(phase=SessionPhase.PROCESSING, adapter=adapter)
    await _activate(harness)
    assert await harness.service.send_text(user_id=42, text="one") is SendOutcome.QUEUED

    notify_entered = asyncio.Event()
    release = asyncio.Event()
    original = harness.service._notify_user

    async def slow_notify(*, user_id: int, text: str) -> bool:
        notify_entered.set()
        await release.wait()
        assert original is not None
        return await original(user_id=user_id, text=text)

    harness.service._notify_user = slow_notify
    state = harness.session_store.get("session-1")
    state.phase = SessionPhase.IDLE
    harness.session_store.save(state)
    # The indeterminate-drop notice is now in flight (hanging in slow_notify)
    # while the drain itself has moved on to its settle wait.
    await _wait_until(lambda: notify_entered.is_set())
    await harness.service._stop_drain("session-1")
    release.set()
    await _wait_until(lambda: any("发送结果不确定" in text for _, text in harness.notices))
    await harness.service.shutdown()


async def test_leave_and_invalidate_clear_queue_and_mode(make_harness) -> None:
    harness = make_harness(phase=SessionPhase.PROCESSING)
    await _activate(harness)
    assert await harness.service.send_text(user_id=42, text="queued") is SendOutcome.QUEUED
    assert await harness.service.leave(user_id=42)
    assert await harness.mode_store.get_target(42) is None
    assert await harness.queue.peek_size("session-1") == 0

    await _activate(harness)
    await harness.queue.enqueue("session-1", text="queued2", binding_id=harness.binding.binding_id)
    await harness.service.invalidate_binding("session-1", reason="test")
    assert await harness.mode_store.get_target(42) is None
    assert await harness.queue.peek_size("session-1") == 0


async def test_rebind_aba_clears_old_mode_queue_and_pair_tokens(make_harness) -> None:
    harness = make_harness(phase=SessionPhase.PROCESSING)
    await _activate(harness)
    await harness.queue.enqueue("session-1", text="old", binding_id=harness.binding.binding_id)
    token = await harness.service.register_pair_token(
        user_id=42,
        session_id="session-1",
        expected_binding_id=harness.binding.binding_id,
        terminal_id="term-1",
    )
    assert token is not None

    await harness.service.rebind_aba("session-1", "new-generation")
    assert await harness.mode_store.get_target(42) is None
    assert await harness.queue.peek_size("session-1") == 0
    assert await harness.service.consume_pair_token(token=token, user_id=42) is PairOutcome.TOKEN_INVALID


# ─── shutdown race ─────────────────────────────────────────────────────────


async def test_shutdown_blocks_new_drain_tasks(make_harness) -> None:
    """shutdown must stop _ensure_drain from spawning orphan drain tasks.

    Regression for #7: ``shutdown`` snapshots the drain slots and releases the
    lifecycle lock before gathering cancellations. An in-flight caller entering
    ``_ensure_drain`` in that window used to create a fresh ``_drain_loop`` that
    shutdown's gather never saw — an orphan that kept injecting past teardown.
    After shutdown sets ``_shutting_down``, ``_ensure_drain`` must refuse to
    create a new slot even when no slot exists.
    """
    harness = make_harness(phase=SessionPhase.PROCESSING)
    await _activate(harness)

    # Precondition: no drain slot spins up by itself while the session is busy.
    assert harness.service._drain_slots.get("session-1") is None

    # Before shutdown, _ensure_drain creates a slot as usual.
    await harness.service._ensure_drain("session-1")
    slot = harness.service._drain_slots.get("session-1")
    assert slot is not None and not slot.task.done()

    # Shutdown cancels the existing slot and arms the shutdown flag.
    await harness.service.shutdown()
    assert harness.service._shutting_down is True
    assert harness.service._drain_slots.get("session-1") is None

    # After shutdown, _ensure_drain must not spawn a new orphan task even
    # though no slot currently exists.
    await harness.service._ensure_drain("session-1")
    assert harness.service._drain_slots.get("session-1") is None

    # Idempotent shutdown: a second call stays a no-op (no new slots, no error).
    await harness.service.shutdown()
    assert harness.service._drain_slots.get("session-1") is None


# ─── validate offload (#12) ────────────────────────────────────────────────


async def test_validate_foreground_offloads_to_worker_thread(make_harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """validate_claude_foreground must run off the event-loop thread (#12).

    The probe spawns up to four synchronous ``ps`` subprocesses per call; running
    them synchronously on the loop thread blocks every Hook/Telegram/drain coroutine
    for the fork+exec+wait duration. The service routes the probe through
    ``asyncio.to_thread`` so the loop stays responsive while the system subprocess
    executes.
    """
    harness = make_harness(paired=True)

    from app.services import external_session_input_service as svc_mod

    captured: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    real = asyncio.to_thread

    async def spy_to_thread(func, /, *args, **kwargs):
        captured.append((func, args, kwargs))
        return await real(func, *args, **kwargs)

    monkeypatch.setattr(svc_mod.asyncio, "to_thread", spy_to_thread)

    result = await harness.service._validate_foreground(pid=1234, paired_tty="/dev/ttys005")

    assert result.ok is True
    # Exactly one offload via to_thread, invoking the probe's sync method with
    # the call's kwargs (not positional pid, no other args).
    assert len(captured) == 1
    func, args, kwargs = captured[0]
    # Bound-method identity is not stable across attribute accesses (a fresh
    # bound method object is created each time), so compare by equality instead
    # of ``is``.
    assert func == harness.probe.validate_claude_foreground
    assert args == ()
    assert kwargs == {"pid": 1234, "paired_tty": "/dev/ttys005"}
    assert harness.probe.validation_calls[-1] == (1234, "/dev/ttys005")


async def test_in_flight_timeout_prevents_drain_deadlock_when_no_further_stop_arrives(make_harness) -> None:
    adapter = FakeGhosttyTerminalAdapter()
    adapter.inject_entered = asyncio.Event()
    adapter.inject_release = asyncio.Event()
    # 使用极短的 in_flight_ttl (0.05s) 测试自动超时解除
    # drain_wait 必须远大于 TTL：超时任务自己要叫醒 drain，不能靠 publish 超时重试。
    harness = make_harness(adapter=adapter, drain_wait=30, in_flight_ttl=0.05)
    await _activate(harness)

    first_task = asyncio.create_task(harness.service.send_text(user_id=42, text="first"))
    await asyncio.wait_for(adapter.inject_entered.wait(), timeout=1)
    # 在注入期间到达 Stop 事件（被 _injecting 分支忽略）
    await harness.service.notify_hook_event(session_id="session-1", event_kind="Stop")
    adapter.inject_release.set()
    assert await first_task is SendOutcome.SENT

    adapter.inject_entered = None
    adapter.inject_release = None

    # 第二条消息入队，进入 in-flight 排队保护
    assert await harness.service.send_text(user_id=42, text="second") is SendOutcome.QUEUED
    assert adapter.inject_calls == [("term-1", "first")]

    # 注意：此时没有任何后续的 Hook Stop 事件到达！
    # 旧实现会永久死锁在这个状态；新实现经过 in_flight_ttl 超时自动释放并成功 drain 注入！
    await _wait_until(lambda: len(adapter.inject_calls) == 2, timeout=1.0)
    assert adapter.inject_calls[-1] == ("term-1", "second")


async def test_stale_in_flight_timer_does_not_clear_newer_guard(make_harness) -> None:
    harness = make_harness(in_flight_ttl=30, drain_wait=30)
    await _activate(harness)
    service = harness.service

    await service._mark_in_flight("session-1")
    stale_generation = service._in_flight_generation["session-1"]
    await service._mark_in_flight("session-1")
    await service._release_in_flight_if_current("session-1", stale_generation)

    assert "session-1" in service._in_flight
    assert await service.send_text(user_id=42, text="queued") is SendOutcome.QUEUED


async def test_in_flight_timeout_cancellation_propagates(make_harness) -> None:
    harness = make_harness(in_flight_ttl=30)
    await _activate(harness)
    service = harness.service
    await service._mark_in_flight("session-1")
    task = service._in_flight_timers["session-1"]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert "session-1" in service._in_flight


async def test_stop_while_arming_in_flight_does_not_drop_guard(make_harness) -> None:
    """A Stop that becomes runnable only while arming waits on the lifecycle lock
    is the same stale completion `_injecting` exists to ignore."""
    harness = make_harness(in_flight_ttl=30, drain_wait=30)
    await _activate(harness)
    service = harness.service

    await service._lifecycle_lock.acquire()
    send_task = asyncio.create_task(service.send_text(user_id=42, text="first"))
    try:
        await _wait_until(
            lambda: "session-1" in service._in_flight and "session-1" not in service._in_flight_timers,
        )
        await service.notify_hook_event(session_id="session-1", event_kind="Stop")
        assert "session-1" in service._in_flight
    finally:
        service._lifecycle_lock.release()

    assert await send_task is SendOutcome.SENT
    assert "session-1" in service._in_flight
    assert "session-1" in service._in_flight_timers
    assert await service.send_text(user_id=42, text="second") is SendOutcome.QUEUED


async def test_cancel_during_in_flight_arming_still_arms_timer(make_harness) -> None:
    harness = make_harness(in_flight_ttl=30, drain_wait=30)
    await _activate(harness)
    service = harness.service

    await service._lifecycle_lock.acquire()
    send_task = asyncio.create_task(service.send_text(user_id=42, text="first"))
    try:
        await _wait_until(
            lambda: "session-1" in service._in_flight and "session-1" not in service._in_flight_timers,
        )
        send_task.cancel()
        await asyncio.sleep(0)
    finally:
        service._lifecycle_lock.release()

    with pytest.raises(asyncio.CancelledError):
        await send_task

    assert "session-1" not in service._injecting
    assert "session-1" in service._in_flight
    assert "session-1" in service._in_flight_timers


async def test_send_text_cancellation_cleans_up_injecting_flag(make_harness) -> None:
    adapter = FakeGhosttyTerminalAdapter()
    adapter.inject_entered = asyncio.Event()
    adapter.inject_release = asyncio.Event()
    harness = make_harness(adapter=adapter)
    await _activate(harness)

    task = asyncio.create_task(harness.service.send_text(user_id=42, text="cancelled"))
    await asyncio.wait_for(adapter.inject_entered.wait(), timeout=1)
    assert "session-1" in harness.service._injecting

    # 取消正在注入的任务
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # 验证 _injecting 已被清理，不会永久残留屏蔽后续 Hook 事件
    assert "session-1" not in harness.service._injecting
    adapter.inject_release.set()
