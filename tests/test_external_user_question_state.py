from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.domain.user_question_models import (
    ExternalTerminalQuestionTarget,
    ExternalTmuxQuestionTarget,
    ExternalUserQuestionPhase,
    UserQuestionPrompt,
)
from app.services.external_user_question_state import (
    ExternalQuestionPendingAmbiguous,
    ExternalQuestionPendingNone,
    ExternalQuestionPendingUnique,
    ExternalUserQuestionState,
    PendingExternalUserQuestion,
)


def _prompt(tool_use_id: str) -> tuple[UserQuestionPrompt, ...]:
    return (
        UserQuestionPrompt(
            tool_use_id=tool_use_id,
            question_index=0,
            total_questions=1,
            question="Choose",
        ),
    )


def _terminal_target(*, paired_at: datetime | None = None) -> ExternalTerminalQuestionTarget:
    return ExternalTerminalQuestionTarget(
        binding_id="binding-1",
        terminal_id="terminal-1",
        paired_tty="/dev/ttys005",
        paired_at=paired_at or datetime.now(UTC),
        kind="ghostty",
    )


def _pending(
    tool_use_id: str,
    *,
    user_id: int = 42,
    session_id: str = "session-1",
    target=None,
) -> PendingExternalUserQuestion:
    return PendingExternalUserQuestion(
        tool_use_id=tool_use_id,
        session_id=session_id,
        user_id=user_id,
        prompts=_prompt(tool_use_id),
        target=target or _terminal_target(),
    )


def test_store_returns_immutable_snapshot_for_tmux_and_ghostty() -> None:
    state = ExternalUserQuestionState()
    state.store(_pending("tool-ghostty"))
    state.store(
        _pending(
            "tool-tmux",
            target=ExternalTmuxQuestionTarget(pane_id="%1", tmux_bin="tmux-custom"),
        )
    )

    ghostty = state.get("tool-ghostty")
    tmux = state.get("tool-tmux")

    assert ghostty is not None and ghostty.target.kind == "ghostty"
    assert tmux is not None and tmux.target == ExternalTmuxQuestionTarget(pane_id="%1", tmux_bin="tmux-custom")


def test_resolve_unique_active_for_user_is_fail_closed_on_ambiguity() -> None:
    state = ExternalUserQuestionState()
    assert isinstance(state.resolve_unique_active_for_user(42), ExternalQuestionPendingNone)

    state.store(_pending("tool-1"))
    unique = state.resolve_unique_active_for_user(42)
    assert isinstance(unique, ExternalQuestionPendingUnique)
    assert unique.pending.tool_use_id == "tool-1"

    state.store(_pending("tool-2", session_id="session-2"))
    ambiguous = state.resolve_unique_active_for_user(42)
    assert isinstance(ambiguous, ExternalQuestionPendingAmbiguous)
    assert ambiguous.count == 2


def test_resolve_unique_active_filters_by_group_across_mixed_targets() -> None:
    """A user with one terminal and one tmux pending is not ambiguous per-group.

    ``resolve_unique_active_for_user(group="external_terminal")`` must return
    the single terminal question (not an ambiguous count of 2), so the
    free-text router — which always queries ``group="external_terminal"`` —
    cannot be tripped by a concurrent external tmux question on the same user.
    """
    state = ExternalUserQuestionState()
    state.store(_pending("tool-ghostty"))

    tmux_target = ExternalTmuxQuestionTarget(pane_id="%1", tmux_bin="tmux")
    state.store(
        PendingExternalUserQuestion(
            tool_use_id="tool-tmux",
            session_id="session-2",
            user_id=42,
            prompts=_prompt("tool-tmux"),
            target=tmux_target,
        )
    )

    terminal = state.resolve_unique_active_for_user(42, group="external_terminal")
    assert isinstance(terminal, ExternalQuestionPendingUnique)
    assert terminal.pending.tool_use_id == "tool-ghostty"

    tmux = state.resolve_unique_active_for_user(42, group="tmux")
    assert isinstance(tmux, ExternalQuestionPendingUnique)
    assert tmux.pending.tool_use_id == "tool-tmux"


def test_phase_transitions_require_expected_target_and_order() -> None:
    state = ExternalUserQuestionState()
    target = _terminal_target()
    state.store(_pending("tool-1", target=target))

    assert not state.mark_completed(tool_use_id="tool-1", expected_target=target)
    assert not state.mark_terminal_action_applied(
        tool_use_id="tool-1",
        expected_target=_terminal_target(paired_at=target.paired_at + timedelta(seconds=1)),
    )
    assert state.mark_terminal_action_applied(tool_use_id="tool-1", expected_target=target)
    assert state.get_active("tool-1") is None
    assert state.mark_completed(tool_use_id="tool-1", expected_target=target)
    assert state.get("tool-1").phase is ExternalUserQuestionPhase.COMPLETED  # type: ignore[union-attr]


def test_mark_indeterminate_blocks_active_question() -> None:
    state = ExternalUserQuestionState()
    target = _terminal_target()
    state.store(_pending("tool-1", target=target))

    assert state.mark_indeterminate(tool_use_id="tool-1", expected_target=target, reason="timeout")
    snapshot = state.get("tool-1")
    assert snapshot is not None
    assert snapshot.phase is ExternalUserQuestionPhase.INDETERMINATE
    assert snapshot.failure_reason == "timeout"
    assert state.get_active("tool-1") is None


def test_mark_indeterminate_from_terminal_action_applied() -> None:
    """``TERMINAL_ACTION_APPLIED`` may still collapse to ``INDETERMINATE``.

    The final-question path pushes the record to ``TERMINAL_ACTION_APPLIED``
    (transport ``submit_after=True``) before the Hook ``allow``; if that allow
    fails (``write_failed``), ``question_indeterminate`` must still be able to
    mark the record. This pairs with ``test_mark_indeterminate_blocks_active_question``
    and covers the transition at the state-API layer, independent of transport.
    """
    state = ExternalUserQuestionState()
    target = _terminal_target()
    state.store(_pending("tool-1", target=target))
    assert state.mark_terminal_action_applied(tool_use_id="tool-1", expected_target=target)

    marked = state.mark_indeterminate(tool_use_id="tool-1", expected_target=target, reason="hook_allow_failed")

    assert marked is True
    snapshot = state.get("tool-1")
    assert snapshot is not None
    assert snapshot.phase is ExternalUserQuestionPhase.INDETERMINATE
    assert snapshot.failure_reason == "hook_allow_failed"
    # Once indeterminate the record is no longer active, and cannot transition
    # forward to COMPLETED (the tower)
    assert state.get_active("tool-1") is None
    assert not state.mark_completed(tool_use_id="tool-1", expected_target=target)
    assert snapshot.phase is ExternalUserQuestionPhase.INDETERMINATE


def test_repair_with_same_terminal_and_tty_invalidates_old_paired_at() -> None:
    state = ExternalUserQuestionState()
    paired_at = datetime.now(UTC)
    target = _terminal_target(paired_at=paired_at)
    state.store(_pending("tool-1", target=target))

    assert (
        state.invalidate_terminal_target(
            session_id="session-1",
            binding_id="binding-1",
            kind="ghostty",
            terminal_id="terminal-1",
            paired_tty="/dev/ttys005",
            paired_at=paired_at + timedelta(seconds=1),
        )
        == 0
    )
    assert state.get("tool-1") is not None
    assert (
        state.invalidate_terminal_target(
            session_id="session-1",
            binding_id="binding-1",
            kind="ghostty",
            terminal_id="terminal-1",
            paired_tty="/dev/ttys005",
            paired_at=paired_at,
        )
        == 1
    )
    assert state.get("tool-1") is None


def test_ttl_prunes_active_but_keeps_recent_terminal_tombstone() -> None:
    now = datetime(2026, 8, 6, tzinfo=UTC)
    wall = now
    mono = 1000.0
    state = ExternalUserQuestionState(
        ttl_sec=10,
        wall_clock=lambda: wall,
        monotonic_clock=lambda: mono,
    )
    target = _terminal_target(paired_at=now)
    state.store(_pending("active", target=target))
    state.store(_pending("done", target=target))
    assert state.mark_terminal_action_applied(tool_use_id="done", expected_target=target)
    assert state.mark_completed(tool_use_id="done", expected_target=target)

    # TTL expiry is monotonic; advance the monotonic clock past the active TTL
    # but stay within the non-active grace window so the terminal tombstone survives.
    mono += 11
    removed = state.prune_stale()

    assert [item.tool_use_id for item in removed] == ["active"]
    assert state.get("done") is not None


def test_ttl_ignores_wall_clock_jumps_and_uses_monotonic_only() -> None:
    """A wall-clock jump (NTP/manual time change) must NOT affect TTL expiry.

    Regression for the twin-clock domain: expiry must read the monotonic anchor,
    never the wall-clock ``created_at``/``updated_at``. A wall clock moved far
    backwards must keep a fresh question alive; moved far forwards must not
    let a just-stored question evade its TTL grace either.
    """
    wall = datetime(2026, 8, 6, tzinfo=UTC)
    mono = 5000.0
    state = ExternalUserQuestionState(
        ttl_sec=10,
        wall_clock=lambda: wall,
        monotonic_clock=lambda: mono,
    )
    target = _terminal_target(paired_at=wall)
    state.store(_pending("fresh", target=target))

    # Wall clock yanked back a year — must not prematurely expire a fresh record.
    wall = wall - timedelta(days=365)
    mono += 1
    assert state.prune_stale() == ()
    assert state.get("fresh") is not None

    # Wall clock yanked forward a year — must not grant a reprieve either; only
    # monotonic advancement past the TTL expires the record.
    wall = wall + timedelta(days=2 * 365)
    mono += 1
    assert state.prune_stale() == ()
    # monotonic past the active TTL → expires regardless of wall clock.
    mono += 10
    assert [item.tool_use_id for item in state.prune_stale()] == ["fresh"]

    state = ExternalUserQuestionState()
    old = _terminal_target()
    new = _terminal_target(paired_at=old.paired_at + timedelta(seconds=1))
    state.store(_pending("tool-1", target=new))

    assert state.remove_if_matches(tool_use_id="tool-1", session_id="session-1", expected_target=old) is None
    assert state.get("tool-1") is not None
    assert state.remove_if_matches(tool_use_id="tool-1", session_id="session-1", expected_target=new) is not None
