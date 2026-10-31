from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.infra.user_question_callbacks import build_user_question_callback_data, parse_user_question_callback_token
from app.services.user_question_callback_registry import (
    UserQuestionCallbackAction,
    UserQuestionCallbackNotFound,
    UserQuestionCallbackOrigin,
    UserQuestionCallbackRegistry,
    UserQuestionCallbackResolved,
    UserQuestionCallbackUnauthorized,
)


async def test_register_reuses_compound_token_and_resolve_is_repeatable() -> None:
    registry = UserQuestionCallbackRegistry(ttl_sec=60, token_factory=lambda: "opaque-token")
    kwargs = dict(
        owner_user_id=42,
        session_id="session-1",
        tool_use_id="tool-" + "x" * 256,
        question_index=0,
        action=UserQuestionCallbackAction.TOGGLE,
        option_index=1,
        origin=UserQuestionCallbackOrigin.EXTERNAL_TERMINAL,
    )

    token = await registry.register(**kwargs)
    assert await registry.register(**kwargs) == token

    first = await registry.resolve(token, user_id=42)
    second = await registry.resolve(token, user_id=42)
    assert isinstance(first, UserQuestionCallbackResolved)
    assert isinstance(second, UserQuestionCallbackResolved)
    assert first.snapshot.tool_use_id == kwargs["tool_use_id"]
    assert isinstance(await registry.resolve(token, user_id=7), UserQuestionCallbackUnauthorized)


def test_callback_data_contains_only_opaque_token_and_is_within_limit() -> None:
    token = "opaque-token"
    for prefix in ("ask", "ext_uq"):
        data = build_user_question_callback_data(prefix=prefix, token=token)
        assert data == f"{prefix}:{token}"
        assert len(data.encode("utf-8")) <= 64
        assert parse_user_question_callback_token(data, prefix=prefix) == token


async def test_invalidation_by_question_tool_session_and_user() -> None:
    tokens: list[str] = []
    counter = 0

    def token_factory() -> str:
        nonlocal counter
        counter += 1
        return f"token-{counter}"

    registry = UserQuestionCallbackRegistry(ttl_sec=60, token_factory=token_factory)
    for question_index, user_id, session_id, tool_use_id in [
        (0, 42, "s1", "t1"),
        (1, 42, "s1", "t1"),
        (0, 42, "s1", "t2"),
        (0, 7, "s2", "t3"),
    ]:
        tokens.append(
            await registry.register(
                owner_user_id=user_id,
                session_id=session_id,
                tool_use_id=tool_use_id,
                question_index=question_index,
                action=UserQuestionCallbackAction.SELECT,
                option_index=0,
                origin=UserQuestionCallbackOrigin.MANAGED,
            )
        )

    assert await registry.invalidate_tool(session_id="s1", tool_use_id="t1") == 2
    assert await registry.invalidate_session("s1") == 1
    assert await registry.invalidate_user(7) == 1
    for token in tokens[:3]:
        assert isinstance(await registry.resolve(token, user_id=42), UserQuestionCallbackNotFound)


async def test_ttl_uses_monotonic_clock_without_refreshing_reused_token() -> None:
    now = 0.0
    registry = UserQuestionCallbackRegistry(
        ttl_sec=10,
        token_factory=lambda: "token",
        clock=lambda: now,
        wall_clock=lambda: datetime(2026, 8, 6, tzinfo=UTC),
    )
    kwargs = dict(
        owner_user_id=42,
        session_id="session-1",
        tool_use_id="tool-1",
        question_index=0,
        action=UserQuestionCallbackAction.SELECT,
        option_index=0,
        origin=UserQuestionCallbackOrigin.EXTERNAL_TERMINAL,
    )
    token = await registry.register(**kwargs)
    now = 9.0
    assert await registry.register(**kwargs) == token
    now = 10.1
    assert isinstance(await registry.resolve(token, user_id=42), UserQuestionCallbackNotFound)


async def test_invalidated_record_is_reaped_at_deadline_without_grace() -> None:
    """Invalidated tokens get no 5xTTL eviction grace: resolve() only accepts
    ACTIVE records, so keeping them past the deadline just grows ``_records``."""
    now = 0.0
    registry = UserQuestionCallbackRegistry(
        ttl_sec=10,
        token_factory=lambda: "token",
        clock=lambda: now,
        wall_clock=lambda: datetime(2026, 8, 6, tzinfo=UTC),
    )
    kwargs = dict(
        owner_user_id=42,
        session_id="session-1",
        tool_use_id="tool-1",
        question_index=0,
        action=UserQuestionCallbackAction.SELECT,
        option_index=0,
        origin=UserQuestionCallbackOrigin.MANAGED,
    )
    token = await registry.register(**kwargs)
    assert await registry.invalidate_tool(session_id="session-1", tool_use_id="tool-1") == 1

    now = 10.1  # just past the deadline, far inside the base-class 5xTTL grace
    assert await registry.prune_stale() == 1
    assert isinstance(await registry.resolve(token, user_id=42), UserQuestionCallbackNotFound)
    # The compound key was reaped with the record, so re-registering mints a fresh token.
    assert await registry.register(**kwargs) == "token"


@pytest.mark.parametrize(
    ("action", "option_index"),
    [
        (UserQuestionCallbackAction.SELECT, None),
        (UserQuestionCallbackAction.TOGGLE, -1),
        (UserQuestionCallbackAction.SUBMIT, 0),
    ],
)
async def test_register_rejects_invalid_action_shape(action, option_index) -> None:
    registry = UserQuestionCallbackRegistry(ttl_sec=60)
    with pytest.raises(ValueError):
        await registry.register(
            owner_user_id=42,
            session_id="session-1",
            tool_use_id="tool-1",
            question_index=0,
            action=action,
            option_index=option_index,
            origin=UserQuestionCallbackOrigin.MANAGED,
        )
