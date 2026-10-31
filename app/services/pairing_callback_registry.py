"""Short-lived token registry for external terminal pairing callbacks.

Telegram callback data has a 64-byte limit and travels through Telegram's
infra, so we never embed the full ``session_id``, ``binding_id`` and
``terminal_id`` UUIDs in callback data. Instead a pairing candidate carries a
short opaque token (``secrets.token_urlsafe(6)``); the callback resolves the
token back to the full quadruple here (terminal kind included, because a
Ghostty UUID and a kitty window id live in different id spaces).

The registry enforces the two pairing security invariants from the design
(``docs/specs/2026-08-03-external-ghostty-input-design.md`` §4 / Security):

  * **owner binding** — only the user who registered the token may consume it
    (``record.user_id == consumer.user_id``). A token cannot be reused by
    another allowed user.
  * **binding generation anchor** — ``record.binding_id`` is the binding
    generation captured at pairing time. ``consume`` returns it so the caller
    re-checks it against the *current* ``ExternalBinding.binding_id`` before
    writing the target. An unbind+rebind that produced a new ``binding_id``
    thus invalidates a previously-issued pairing token even before it expires
    (the caller refuses the stale generation). ``invalidate_binding`` also
    marks such tokens consumed-to-stale explicitly.

Pattern mirrors ``PermissionCallbackRegistry``: an ``asyncio.Lock`` protects
the records dict, a monotonic clock drives TTL eviction, a wall clock stamps
``created_at``/``expires_at``. Tokens are single-use: a successful consume
flips the record to ``CONSUMED`` and it can never be consumed again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from app.domain.external_terminal_models import TerminalKind
from app.services.base_callback_registry import BaseCallbackRegistry

logger = logging.getLogger(__name__)


class PairingTokenStatus(StrEnum):
    PENDING = "pending"
    CONSUMED = "consumed"
    INVALIDATED = "invalidated"  # superseded by a fresh token, or binding tore down


@dataclass(slots=True)
class PairingTokenRecord:
    token: str
    user_id: int
    session_id: str
    binding_id: str
    terminal_kind: TerminalKind
    terminal_id: str
    created_at: datetime
    expires_at: datetime
    status: PairingTokenStatus


@dataclass(frozen=True, slots=True)
class PairingTokenSnapshot:
    token: str
    user_id: int
    session_id: str
    binding_id: str
    terminal_kind: TerminalKind
    terminal_id: str
    created_at: datetime
    expires_at: datetime
    status: PairingTokenStatus

    @classmethod
    def from_record(cls, record: PairingTokenRecord) -> PairingTokenSnapshot:
        return cls(
            token=record.token,
            user_id=record.user_id,
            session_id=record.session_id,
            binding_id=record.binding_id,
            terminal_kind=record.terminal_kind,
            terminal_id=record.terminal_id,
            created_at=record.created_at,
            expires_at=record.expires_at,
            status=record.status,
        )


# --- consume results (frozen, caller pattern-matches) -----------------------


@dataclass(frozen=True, slots=True)
class PairConsumeOk:
    """The token was consumed; the caller now owns the resolved triple and
    MUST re-check ``binding_id`` against the live binding before pairing."""

    snapshot: PairingTokenSnapshot


@dataclass(frozen=True, slots=True)
class PairConsumeNotFound:
    """Token unknown, expired, or already invalidated/superseded."""


@dataclass(frozen=True, slots=True)
class PairConsumeUnauthorized:
    """The consumer is not the user who registered the token (owner mismatch)."""


@dataclass(frozen=True, slots=True)
class PairConsumeAlreadyConsumed:
    """The token was already consumed (replay of an old callback button)."""


PairConsumeResult = PairConsumeOk | PairConsumeNotFound | PairConsumeUnauthorized | PairConsumeAlreadyConsumed


class PairingCallbackRegistry(BaseCallbackRegistry[PairingTokenRecord, PairingTokenStatus, tuple[str, TerminalKind, str]]):
    """In-memory, TTL-bounded pairing token registry.

    Construct with ``ttl_sec`` (token lifetime), an optional token factory and
    clocks (for tests). All public mutating methods take the shared
    ``asyncio.Lock`` so concurrent callback presses for the same token are
    serialised: exactly one press consumes, the rest see
    ``PairConsumeAlreadyConsumed``. Eviction/expiry mechanics come from
    ``BaseCallbackRegistry``; this subclass adds the pairing domain policy —
    superseding a still-pending token on a fresh register for the same
    ``(session_id, terminal_id)``, single-use consumption and
    binding-generation invalidation.
    """

    def _pending_status(self) -> PairingTokenStatus:
        return PairingTokenStatus.PENDING

    async def register_token(
        self,
        *,
        user_id: int,
        session_id: str,
        binding_id: str,
        terminal_kind: TerminalKind,
        terminal_id: str,
    ) -> str:
        """Issue a fresh pairing token for one displayed terminal candidate.

        Any still-pending token for the same ``(session_id, kind,
        terminal_id)`` is marked INVALIDATED (superseded) so a stale callback
        button pressed later yields ``PairConsumeNotFound`` instead of pairing
        to an old candidate snapshot. The kind is part of the key because
        terminal ids are only unique within one emulator. Returns the new
        token.
        """
        async with self._lock:
            self._evict_stale()
            compound_key = (session_id, terminal_kind, terminal_id)
            previous_token = self._compound_index.get(compound_key)
            previous = self._records.get(previous_token) if previous_token is not None else None
            if previous is not None and previous.status is PairingTokenStatus.PENDING:
                previous.status = PairingTokenStatus.INVALIDATED

            token = self._generate_unique_token("pairing token")
            monotonic_now = self._clock()
            created_at = self._now_datetime()
            record = PairingTokenRecord(
                token=token,
                user_id=user_id,
                session_id=session_id,
                binding_id=binding_id,
                terminal_kind=terminal_kind,
                terminal_id=terminal_id,
                created_at=created_at,
                expires_at=created_at + timedelta(seconds=self._ttl_sec),
                status=PairingTokenStatus.PENDING,
            )

            self._records[token] = record
            self._ttl_deadlines[token] = monotonic_now + self._ttl_sec
            self._compound_index[compound_key] = token
            return token

    async def consume(self, token: str, user_id: int) -> PairConsumeResult:
        """Resolve and burn a pairing token. Single-use on success.

        Order of checks (fail-closed):
          1. unknown / expired / invalidated  -> NotFound
          2. owner mismatch                    -> Unauthorized
          3. already consumed                  -> AlreadyConsumed
        On success the record is flipped to CONSUMED (cannot be reused) and the
        snapshot returned; the caller MUST re-check ``binding_id`` against the
        live binding before writing the target (the token is the ABA hint, the
        binding store is the ABA authority).
        """
        async with self._lock:
            self._evict_stale()
            record = self._records.get(token)
            if record is None or self._is_expired(record) or record.status is PairingTokenStatus.INVALIDATED:
                return PairConsumeNotFound()
            if record.user_id != user_id:
                return PairConsumeUnauthorized()
            if record.status is PairingTokenStatus.CONSUMED:
                return PairConsumeAlreadyConsumed()

            record.status = PairingTokenStatus.CONSUMED
            return PairConsumeOk(PairingTokenSnapshot.from_record(record))

    async def invalidate_session(self, session_id: str) -> int:
        """Invalidate all pending tokens for a session (unbind / SessionEnd).

        Returns the count of newly-invalidated tokens.
        """
        async with self._lock:
            self._evict_stale()
            count = 0
            for record in self._records.values():
                if record.session_id == session_id and record.status is PairingTokenStatus.PENDING:
                    record.status = PairingTokenStatus.INVALIDATED
                    count += 1
            for compound_key, token in list(self._compound_index.items()):
                rec = self._records.get(token)
                if rec is not None and rec.session_id == session_id and rec.status is PairingTokenStatus.INVALIDATED:
                    self._compound_index.pop(compound_key, None)
            return count

    async def invalidate_binding(self, session_id: str, binding_id: str) -> int:
        """Invalidate pending tokens whose generation no longer matches.

        Called after an unbind+rebind that produced a new ``binding_id``: any
        token issued under the old generation must not pair into the new one.
        """
        async with self._lock:
            self._evict_stale()
            count = 0
            for record in self._records.values():
                if record.session_id == session_id and record.binding_id != binding_id and record.status is PairingTokenStatus.PENDING:
                    record.status = PairingTokenStatus.INVALIDATED
                    count += 1
            for compound_key, token in list(self._compound_index.items()):
                rec = self._records.get(token)
                if (
                    rec is not None
                    and rec.session_id == session_id
                    and rec.binding_id != binding_id
                    and rec.status is PairingTokenStatus.INVALIDATED
                ):
                    self._compound_index.pop(compound_key, None)
            return count

    async def get_record(self, token: str) -> PairingTokenSnapshot | None:
        """Inspect a token without consuming it (diagnostics / tests)."""
        async with self._lock:
            record = self._records.get(token)
            if record is None:
                return None
            return PairingTokenSnapshot.from_record(record)
