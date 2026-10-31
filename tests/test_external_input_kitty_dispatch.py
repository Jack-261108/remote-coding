"""ExternalSessionInputService dispatch across multiple terminal backends.

The service must route every operation through the backend that owns the
target's ``kind``: a kitty pairing must never reach the Ghostty adapter, and
candidate listing must merge both backends with kitty's PID signal used only
for ordering.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.external_terminal_models import ExternalTerminal, TerminalOutcome
from app.services.external_session_input_service import (
    ExternalSessionInputService,
    PairOutcome,
    SendOutcome,
)
from tests.fakes.external_input_harness import InputHarness, build_input_harness
from tests.fakes.ghostty import FakeGhosttyTerminalAdapter
from tests.fakes.kitty import FakeKittyTerminalAdapter


@pytest.fixture
async def make_harness(tmp_path: Path):
    services: list[ExternalSessionInputService] = []
    counter = 0

    def _make(
        *,
        paired: bool = True,
        paired_kind: str = "kitty",
        ghostty: FakeGhosttyTerminalAdapter | None = None,
        kitty: FakeKittyTerminalAdapter | None = None,
    ) -> tuple[InputHarness, FakeGhosttyTerminalAdapter, FakeKittyTerminalAdapter]:
        nonlocal counter
        counter += 1
        ghostty = ghostty or FakeGhosttyTerminalAdapter()
        kitty = kitty or FakeKittyTerminalAdapter()
        harness = build_input_harness(
            tmp_path,
            name=f"k-{counter}",
            paired=paired,
            paired_kind=paired_kind,  # type: ignore[arg-type]
            paired_terminal_id="101" if paired_kind == "kitty" else "term-1",
            adapter=ghostty,
            extra_backends={"kitty": kitty},
        )
        services.append(harness.service)
        return harness, ghostty, kitty

    yield _make

    for service in services:
        await service.shutdown()


async def _activate(harness: InputHarness) -> None:
    await harness.mode_store.set_target(
        user_id=42,
        session_id=harness.binding.session_id,
        binding_id=harness.binding.binding_id,
    )


# ─── candidate listing ──────────────────────────────────────────────


async def test_pair_candidates_merges_both_backends(make_harness) -> None:
    harness, ghostty, kitty = make_harness(paired=False)

    outcome, candidates = await harness.service.pair_candidates(user_id=42, session_id="session-1")

    assert outcome is PairOutcome.NEEDS_PAIRING
    assert candidates is not None
    assert {(t.kind, t.terminal_id) for t in candidates.terminals} == {("ghostty", "term-1"), ("kitty", "101")}
    assert candidates.binding_pid == 1234


async def test_pair_candidates_orders_kitty_foreground_pid_first(make_harness) -> None:
    kitty = FakeKittyTerminalAdapter(
        terminals=[
            ExternalTerminal(kind="kitty", terminal_id="201", name="shell", cwd="/project", pid=999),
            ExternalTerminal(
                kind="kitty",
                terminal_id="202",
                name="claude",
                cwd="/elsewhere",
                pid=555,
                foreground_pids=(1234,),
            ),
            ExternalTerminal(kind="kitty", terminal_id="203", name="other", cwd="/project", pid=777),
        ]
    )
    harness, _ghostty, _kitty = make_harness(paired=False, kitty=kitty)

    outcome, candidates = await harness.service.pair_candidates(user_id=42, session_id="session-1")

    assert outcome is PairOutcome.NEEDS_PAIRING
    assert candidates is not None
    kitty_order = [t.terminal_id for t in candidates.terminals if t.kind == "kitty"]
    # foreground pid hit first, then cwd hit, then the rest; no auto-selection.
    assert kitty_order == ["202", "201", "203"]


async def test_pair_candidates_ranks_own_pid_above_cwd(make_harness) -> None:
    kitty = FakeKittyTerminalAdapter(
        terminals=[
            ExternalTerminal(kind="kitty", terminal_id="301", name="a", cwd="/project", pid=1),
            ExternalTerminal(kind="kitty", terminal_id="302", name="b", cwd="/other", pid=1234),
        ]
    )
    harness, _ghostty, _kitty = make_harness(paired=False, kitty=kitty)

    _, candidates = await harness.service.pair_candidates(user_id=42, session_id="session-1")

    assert candidates is not None
    assert [t.terminal_id for t in candidates.terminals if t.kind == "kitty"] == ["302", "301"]


async def test_pair_candidates_reports_adapter_unavailable_only_when_all_fail(make_harness) -> None:
    harness, ghostty, kitty = make_harness(paired=False)
    ghostty.list_error = "tcc denied"
    kitty.list_error = "socket gone"

    outcome, candidates = await harness.service.pair_candidates(user_id=42, session_id="session-1")

    assert outcome is PairOutcome.ADAPTER_UNAVAILABLE
    assert candidates is None


async def test_pair_candidates_tolerates_one_backend_failing(make_harness) -> None:
    harness, ghostty, kitty = make_harness(paired=False)
    ghostty.list_error = "tcc denied"

    outcome, candidates = await harness.service.pair_candidates(user_id=42, session_id="session-1")

    assert outcome is PairOutcome.NEEDS_PAIRING
    assert candidates is not None
    assert [t.kind for t in candidates.terminals] == ["kitty"]


async def test_pair_candidates_no_terminals_when_backends_list_empty(make_harness) -> None:
    harness, _ghostty, _kitty = make_harness(
        paired=False,
        ghostty=FakeGhosttyTerminalAdapter(terminals=[]),
        kitty=FakeKittyTerminalAdapter(terminals=[]),
    )

    outcome, candidates = await harness.service.pair_candidates(user_id=42, session_id="session-1")

    assert outcome is PairOutcome.NO_TERMINALS
    assert candidates is None


# ─── pairing ───────────────────────────────────────────────────────


async def test_register_pair_token_unknown_kind_is_refused(make_harness) -> None:
    harness, _ghostty, _kitty = make_harness(paired=False)

    token = await harness.service.register_pair_token(
        user_id=42,
        session_id="session-1",
        expected_binding_id=harness.binding.binding_id,
        terminal_kind="iterm",  # type: ignore[arg-type]
        terminal_id="1",
    )

    assert token is None


async def test_consume_pair_token_pairs_kitty_and_routes_to_kitty_backend(make_harness) -> None:
    harness, ghostty, kitty = make_harness(paired=False)
    token = await harness.service.register_pair_token(
        user_id=42,
        session_id="session-1",
        expected_binding_id=harness.binding.binding_id,
        terminal_kind="kitty",
        terminal_id="101",
    )
    assert token is not None

    outcome = await harness.service.consume_pair_token(token=token, user_id=42)

    assert outcome is PairOutcome.PAIRED
    binding = harness.binding_store.get_binding("session-1")
    assert binding is not None and binding.terminal_target is not None
    assert binding.terminal_target.kind == "kitty"
    assert binding.terminal_target.terminal_id == "101"
    assert kitty.validate_calls == ["101"]
    assert ghostty.validate_calls == []


# ─── send ──────────────────────────────────────────────────────────


async def test_send_routes_to_kitty_backend_only(make_harness) -> None:
    harness, ghostty, kitty = make_harness(paired=True, paired_kind="kitty")
    await _activate(harness)

    outcome = await harness.service.send_text(user_id=42, text="hello")

    assert outcome is SendOutcome.SENT
    assert kitty.inject_calls == [("101", "hello")]
    assert ghostty.inject_calls == []
    assert ghostty.validate_calls == []


async def test_send_with_ghostty_target_never_touches_kitty(make_harness) -> None:
    harness, ghostty, kitty = make_harness(paired=True, paired_kind="ghostty")
    await _activate(harness)

    outcome = await harness.service.send_text(user_id=42, text="hi")

    assert outcome is SendOutcome.SENT
    assert ghostty.inject_calls == [("term-1", "hi")]
    assert kitty.inject_calls == []


async def test_send_reports_terminal_invalid_when_kitty_window_closed(make_harness) -> None:
    harness, _ghostty, kitty = make_harness(paired=True, paired_kind="kitty")
    await _activate(harness)
    kitty.validate_error = TerminalOutcome.NOT_FOUND

    outcome = await harness.service.send_text(user_id=42, text="hello")

    assert outcome is SendOutcome.TERMINAL_INVALID
    binding = harness.binding_store.get_binding("session-1")
    assert binding is not None and binding.terminal_target is None
    assert await harness.mode_store.get_target(42) is None


async def test_send_reports_adapter_unavailable_when_kitty_off(make_harness) -> None:
    harness, _ghostty, kitty = make_harness(paired=True, paired_kind="kitty")
    await _activate(harness)
    kitty.available = False

    assert await harness.service.send_text(user_id=42, text="hello") is SendOutcome.ADAPTER_UNAVAILABLE


async def test_send_indeterminate_outcome_is_propagated(make_harness) -> None:
    harness, _ghostty, kitty = make_harness(paired=True, paired_kind="kitty")
    await _activate(harness)
    kitty.inject_outcomes.append(TerminalOutcome.INDETERMINATE)

    assert await harness.service.send_text(user_id=42, text="hello") is SendOutcome.INJECTION_INDETERMINATE


# ─── activation ────────────────────────────────────────────────────


async def test_activate_select_uses_kitty_backend(make_harness) -> None:
    harness, ghostty, kitty = make_harness(paired=True, paired_kind="kitty")

    outcome = await harness.service.activate_select(user_id=42, session_id="session-1")

    assert outcome is PairOutcome.ACTIVATED
    assert kitty.validate_calls == ["101"]
    assert ghostty.validate_calls == []
