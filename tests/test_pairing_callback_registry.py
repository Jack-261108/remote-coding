"""Unit tests for PairingCallbackRegistry.

Verifies the two pairing security invariants (owner binding + binding
generation anchor) plus TTL, single-use consumption, supersede-on-refresh,
session/binding invalidation, and that a concurrent pair of presses for the
same token consumes exactly once. The registry never embeds the full triple
in Telegram callback data — the token is the lookup key, and consume returns
the resolved triple for the caller to re-authorise against the live binding.
"""

from __future__ import annotations

import asyncio

import pytest

from app.services.pairing_callback_registry import (
    PairConsumeAlreadyConsumed,
    PairConsumeNotFound,
    PairConsumeOk,
    PairConsumeResult,
    PairConsumeUnauthorized,
    PairingCallbackRegistry,
    PairingTokenStatus,
)


def _registry(
    *,
    ttl_sec: int = 60,
    token_factory=None,
) -> PairingCallbackRegistry:
    return PairingCallbackRegistry(ttl_sec=ttl_sec, token_factory=token_factory)


async def test_register_then_consume_returns_triple() -> None:
    reg = _registry(token_factory=lambda: "tok-1")
    token = await reg.register_token(user_id=42, session_id="s1", binding_id="gen-A", terminal_kind="ghostty", terminal_id="term-uuid")
    assert token == "tok-1"
    result = await reg.consume(token, user_id=42)
    assert isinstance(result, PairConsumeOk)
    snap = result.snapshot
    assert snap.user_id == 42
    assert snap.session_id == "s1"
    assert snap.binding_id == "gen-A"
    assert snap.terminal_id == "term-uuid"
    assert snap.status is PairingTokenStatus.CONSUMED


async def test_consume_unknown_token_is_not_found() -> None:
    reg = _registry(token_factory=lambda: "tok-1")
    result = await reg.consume("nope", user_id=42)
    assert isinstance(result, PairConsumeNotFound)


async def test_consume_owner_mismatch_is_unauthorized() -> None:
    """A token issued to user 42 cannot be consumed by user 7 — tokens do not
    cross owners even if both are allowed Telegram users."""
    reg = _registry(token_factory=lambda: "tok-1")
    token = await reg.register_token(user_id=42, session_id="s1", binding_id="gen-A", terminal_kind="ghostty", terminal_id="term-uuid")
    result = await reg.consume(token, user_id=7)
    assert isinstance(result, PairConsumeUnauthorized)
    # The failing consume MUST NOT burn the token — owner 42 can still consume.
    second = await reg.consume(token, user_id=42)
    assert isinstance(second, PairConsumeOk)


async def test_consume_is_single_use_replay_yields_already_consumed() -> None:
    """A double-press of the same callback button consumes once; the second
    press sees AlreadyConsumed (no double pairing)."""
    reg = _registry(token_factory=lambda: "tok-1")
    token = await reg.register_token(user_id=42, session_id="s1", binding_id="gen-A", terminal_kind="ghostty", terminal_id="term-uuid")
    first = await reg.consume(token, user_id=42)
    assert isinstance(first, PairConsumeOk)
    second = await reg.consume(token, user_id=42)
    assert isinstance(second, PairConsumeAlreadyConsumed)


async def test_token_expires_after_ttl() -> None:
    """An expired token yields NotFound (not Unauthorized/AlreadyConsumed) so the
    UI can say "pairing expired, refresh"."""
    clock = {"t": 0.0}
    reg = PairingCallbackRegistry(
        ttl_sec=10,
        clock=lambda: clock["t"],
        token_factory=lambda: "tok-1",
    )
    token = await reg.register_token(user_id=42, session_id="s1", binding_id="gen-A", terminal_kind="ghostty", terminal_id="term-uuid")
    clock["t"] += 11.0
    result = await reg.consume(token, 42)
    assert isinstance(result, PairConsumeNotFound), "expired token must not consume"


async def test_invalidate_session_marks_pending_invalidated() -> None:
    reg = _registry(token_factory=lambda: "tok-1")
    token = await reg.register_token(user_id=42, session_id="s1", binding_id="gen-A", terminal_kind="ghostty", terminal_id="term-uuid")
    count = await reg.invalidate_session("s1")
    assert count == 1
    result = await reg.consume(token, user_id=42)
    assert isinstance(result, PairConsumeNotFound), "invalidated token must not consume"


async def test_invalidate_binding_marks_old_generation_tokens() -> None:
    """After unbind+rebind produced a new binding_id, tokens issued under the
    old generation MUST be invalidated so they cannot pair into the new binding
    (ABA barrier at the token layer)."""
    reg = _registry(token_factory=lambda: "tok-1")
    token = await reg.register_token(user_id=42, session_id="s1", binding_id="gen-old", terminal_kind="ghostty", terminal_id="term-uuid")
    count = await reg.invalidate_binding("s1", "gen-new")
    assert count == 1
    result = await reg.consume(token, user_id=42)
    assert isinstance(result, PairConsumeNotFound)
    # A token issued under the CURRENT generation is untouched.
    reg2 = _registry(token_factory=lambda: "tok-2")
    tok2 = await reg2.register_token(user_id=42, session_id="s1", binding_id="gen-new", terminal_kind="ghostty", terminal_id="term-uuid")
    assert await reg2.invalidate_binding("s1", "gen-new") == 0
    assert isinstance(await reg2.consume(tok2, 42), PairConsumeOk)


async def test_register_supersedes_pending_for_same_session_terminal() -> None:
    """Refreshing the candidate list issues a new token; the old token is
    INVALIDATED so a stale button pressed later cannot pair to an old snapshot."""
    seq = iter(["tok-old", "tok-new"])
    reg = _registry(token_factory=lambda: next(seq))
    old = await reg.register_token(user_id=42, session_id="s1", binding_id="gen-A", terminal_kind="ghostty", terminal_id="term-uuid")
    new = await reg.register_token(user_id=42, session_id="s1", binding_id="gen-A", terminal_kind="ghostty", terminal_id="term-uuid")
    assert old == "tok-old" and new == "tok-new"
    assert isinstance(await reg.consume(old, 42), PairConsumeNotFound), "superseded -> NotFound"
    assert isinstance(await reg.consume(new, 42), PairConsumeOk)


async def test_consume_result_variants_exhaustive() -> None:
    """Sanity: the public result union covers exactly these four variants."""
    from typing import get_args

    variants = get_args(PairConsumeResult)
    names = {v.__name__ for v in variants}
    assert names == {"PairConsumeOk", "PairConsumeNotFound", "PairConsumeUnauthorized", "PairConsumeAlreadyConsumed"}


@pytest.mark.asyncio
async def test_concurrent_presses_consume_exactly_once() -> None:
    """Two presses for the same token arriving concurrently: the asyncio.Lock
    serialises them, exactly one succeeds and the other sees AlreadyConsumed."""
    reg = _registry(token_factory=lambda: "tok-1")
    token = await reg.register_token(user_id=42, session_id="s1", binding_id="gen-A", terminal_kind="ghostty", terminal_id="term-uuid")
    results = await asyncio.gather(reg.consume(token, 42), reg.consume(token, 42))
    ok_count = sum(1 for r in results if isinstance(r, PairConsumeOk))
    consumed_count = sum(1 for r in results if isinstance(r, PairConsumeAlreadyConsumed))
    assert ok_count == 1 and consumed_count == 1
