"""Reusable opaque tokens for Telegram AskUserQuestion callbacks."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from app.services.base_callback_registry import BaseCallbackRegistry


class UserQuestionCallbackAction(StrEnum):
    SELECT = "select"
    TOGGLE = "toggle"
    SUBMIT = "submit"


class UserQuestionCallbackOrigin(StrEnum):
    MANAGED = "managed"
    EXTERNAL_GHOSTTY = "external_ghostty"
    EXTERNAL_TMUX = "external_tmux"


class UserQuestionCallbackStatus(StrEnum):
    ACTIVE = "active"
    INVALIDATED = "invalidated"


@dataclass(slots=True)
class UserQuestionCallbackRecord:
    token: str
    owner_user_id: int
    session_id: str
    tool_use_id: str
    question_index: int
    action: UserQuestionCallbackAction
    option_index: int | None
    origin: UserQuestionCallbackOrigin
    created_at: datetime
    expires_at: datetime
    status: UserQuestionCallbackStatus


@dataclass(frozen=True, slots=True)
class UserQuestionCallbackSnapshot:
    token: str
    owner_user_id: int
    session_id: str
    tool_use_id: str
    question_index: int
    action: UserQuestionCallbackAction
    option_index: int | None
    origin: UserQuestionCallbackOrigin
    created_at: datetime
    expires_at: datetime
    status: UserQuestionCallbackStatus

    @classmethod
    def from_record(cls, record: UserQuestionCallbackRecord) -> UserQuestionCallbackSnapshot:
        return cls(
            token=record.token,
            owner_user_id=record.owner_user_id,
            session_id=record.session_id,
            tool_use_id=record.tool_use_id,
            question_index=record.question_index,
            action=record.action,
            option_index=record.option_index,
            origin=record.origin,
            created_at=record.created_at,
            expires_at=record.expires_at,
            status=record.status,
        )


@dataclass(frozen=True, slots=True)
class UserQuestionCallbackResolved:
    snapshot: UserQuestionCallbackSnapshot


@dataclass(frozen=True, slots=True)
class UserQuestionCallbackUnauthorized:
    pass


@dataclass(frozen=True, slots=True)
class UserQuestionCallbackNotFound:
    pass


UserQuestionCallbackResolveResult = UserQuestionCallbackResolved | UserQuestionCallbackUnauthorized | UserQuestionCallbackNotFound


@dataclass(frozen=True, slots=True)
class QuestionCallbackTokens:
    """Opaque tokens for one question's buttons, keyed by option index.

    ``select_tokens[i]`` is the single-choice callback for option *i*;
    ``toggle_tokens[i]`` is the multi-select toggle for option *i*;
    ``submit_token`` is the multi-select submit. Empty tuples and a null
    ``submit_token`` mark the degenerate case where registration was skipped
    (no registry/session available); callers detect this by checking the
    tuples rather than a separate flag.
    """

    select_tokens: tuple[str, ...] = ()
    toggle_tokens: tuple[str, ...] = ()
    submit_token: str | None = None


UserQuestionCompoundKey = tuple[int, str, str, int, UserQuestionCallbackAction, int | None, UserQuestionCallbackOrigin]


class UserQuestionCallbackRegistry(BaseCallbackRegistry[UserQuestionCallbackRecord, UserQuestionCallbackStatus, UserQuestionCompoundKey]):
    """TTL registry whose tokens can be resolved repeatedly until invalidated."""

    # No post-deadline grace for invalidated tokens: unlike resolved
    # permission/pairing records they serve no "already answered" observation
    # window (resolve() only accepts ACTIVE), so reap them at their deadline
    # like the pre-BaseCallbackRegistry implementation did.
    _NON_PENDING_EVICT_GRACE_MULTIPLIER = 0

    def __init__(
        self,
        *,
        ttl_sec: int,
        token_factory: Callable[[], str] | None = None,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(
            ttl_sec=ttl_sec,
            token_factory=token_factory or (lambda: secrets.token_urlsafe(12)),
            clock=clock,
            wall_clock=wall_clock,
        )

    def _pending_status(self) -> UserQuestionCallbackStatus:
        return UserQuestionCallbackStatus.ACTIVE

    async def register_question_tokens(
        self,
        *,
        owner_user_id: int,
        session_id: str,
        tool_use_id: str,
        question_index: int,
        option_count: int,
        multi_select: bool,
        origin: UserQuestionCallbackOrigin,
    ) -> QuestionCallbackTokens:
        """Register opaque tokens for every button of one question prompt at once."""
        if not session_id or not tool_use_id or option_count <= 0:
            return QuestionCallbackTokens()
        if multi_select:
            toggle_tokens = [
                await self.register(
                    owner_user_id=owner_user_id,
                    session_id=session_id,
                    tool_use_id=tool_use_id,
                    question_index=question_index,
                    action=UserQuestionCallbackAction.TOGGLE,
                    option_index=index,
                    origin=origin,
                )
                for index in range(option_count)
            ]
            submit_token = await self.register(
                owner_user_id=owner_user_id,
                session_id=session_id,
                tool_use_id=tool_use_id,
                question_index=question_index,
                action=UserQuestionCallbackAction.SUBMIT,
                option_index=None,
                origin=origin,
            )
            return QuestionCallbackTokens(
                toggle_tokens=tuple(toggle_tokens),
                submit_token=submit_token,
            )
        select_tokens = [
            await self.register(
                owner_user_id=owner_user_id,
                session_id=session_id,
                tool_use_id=tool_use_id,
                question_index=question_index,
                action=UserQuestionCallbackAction.SELECT,
                option_index=index,
                origin=origin,
            )
            for index in range(option_count)
        ]
        return QuestionCallbackTokens(
            select_tokens=tuple(select_tokens),
        )

    async def register(
        self,
        *,
        owner_user_id: int,
        session_id: str,
        tool_use_id: str,
        question_index: int,
        action: UserQuestionCallbackAction,
        option_index: int | None,
        origin: UserQuestionCallbackOrigin,
    ) -> str:
        if not session_id or not tool_use_id:
            raise ValueError("session_id and tool_use_id are required")
        if owner_user_id <= 0:
            raise ValueError("owner_user_id must be positive")
        if question_index < 0:
            raise ValueError("question_index must be non-negative")
        if action in {UserQuestionCallbackAction.SELECT, UserQuestionCallbackAction.TOGGLE}:
            if option_index is None or option_index < 0:
                raise ValueError("select/toggle requires a non-negative option_index")
        elif option_index is not None:
            raise ValueError("submit must not include option_index")

        compound_key: UserQuestionCompoundKey = (
            owner_user_id,
            session_id,
            tool_use_id,
            question_index,
            action,
            option_index,
            origin,
        )
        async with self._lock:
            self._evict_stale()
            existing_token = self._compound_index.get(compound_key)
            existing = self._records.get(existing_token) if existing_token is not None else None
            if existing is not None and existing.status is UserQuestionCallbackStatus.ACTIVE and not self._is_expired(existing):
                return existing.token

            token = self._generate_unique_token("user-question callback token")
            monotonic_now = self._clock()
            created_at = self._now_datetime()
            record = UserQuestionCallbackRecord(
                token=token,
                owner_user_id=owner_user_id,
                session_id=session_id,
                tool_use_id=tool_use_id,
                question_index=question_index,
                action=action,
                option_index=option_index,
                origin=origin,
                created_at=created_at,
                expires_at=created_at + timedelta(seconds=self._ttl_sec),
                status=UserQuestionCallbackStatus.ACTIVE,
            )
            self._records[token] = record
            self._ttl_deadlines[token] = monotonic_now + self._ttl_sec
            self._compound_index[compound_key] = token
            return token

    async def resolve(self, token: str, *, user_id: int) -> UserQuestionCallbackResolveResult:
        async with self._lock:
            self._evict_stale()
            record = self._records.get(token)
            if record is None or record.status is not UserQuestionCallbackStatus.ACTIVE or self._is_expired(record):
                return UserQuestionCallbackNotFound()
            if record.owner_user_id != user_id:
                return UserQuestionCallbackUnauthorized()
            return UserQuestionCallbackResolved(UserQuestionCallbackSnapshot.from_record(record))

    async def invalidate_prompt(self, *, session_id: str, tool_use_id: str, question_index: int) -> int:
        return await self._invalidate_matching(
            lambda record: record.session_id == session_id and record.tool_use_id == tool_use_id and record.question_index == question_index
        )

    async def invalidate_tool(self, *, session_id: str, tool_use_id: str) -> int:
        return await self._invalidate_matching(lambda record: record.session_id == session_id and record.tool_use_id == tool_use_id)

    async def invalidate_session(self, session_id: str) -> int:
        return await self._invalidate_matching(lambda record: record.session_id == session_id)

    async def invalidate_user(self, user_id: int) -> int:
        return await self._invalidate_matching(lambda record: record.owner_user_id == user_id)

    async def clear(self) -> int:
        async with self._lock:
            count = len(self._records)
            self._records.clear()
            self._ttl_deadlines.clear()
            self._compound_index.clear()
            return count

    async def prune_stale(self) -> int:
        async with self._lock:
            before = len(self._records)
            self._evict_stale()
            return before - len(self._records)

    async def _invalidate_matching(self, predicate: Callable[[UserQuestionCallbackRecord], bool]) -> int:
        async with self._lock:
            self._evict_stale()
            invalidated_tokens: set[str] = set()
            for record in self._records.values():
                if record.status is UserQuestionCallbackStatus.ACTIVE and predicate(record):
                    record.status = UserQuestionCallbackStatus.INVALIDATED
                    invalidated_tokens.add(record.token)
            if invalidated_tokens:
                for compound_key, token in list(self._compound_index.items()):
                    if token in invalidated_tokens:
                        self._compound_index.pop(compound_key, None)
            return len(invalidated_tokens)
