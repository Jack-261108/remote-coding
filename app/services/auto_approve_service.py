from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from app.domain.session_tombstone import SessionTombstoneStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AutoApproveActivation:
    session_id: str
    user_id: int
    activated_at: datetime


@dataclass(frozen=True, slots=True)
class ActivationSlot:
    session_id: str
    holder_user_id: int
    attempt_id: str


@dataclass(frozen=True, slots=True)
class SlotClaimed:
    attempt_id: str


@dataclass(frozen=True, slots=True)
class SlotConflict:
    holder_user_id: int


@dataclass(frozen=True, slots=True)
class SlotAlreadyClaimedBySameUser:
    attempt_id: str


@dataclass(frozen=True, slots=True)
class SlotActiveOwnerExists:
    owner_user_id: int


SlotClaimResult = SlotClaimed | SlotConflict | SlotAlreadyClaimedBySameUser | SlotActiveOwnerExists


@dataclass(frozen=True, slots=True)
class CommitSlotSucceeded:
    pass


@dataclass(frozen=True, slots=True)
class CommitSlotSessionEnded:
    pass


@dataclass(frozen=True, slots=True)
class CommitSlotMismatch:
    pass


CommitSlotResult = CommitSlotSucceeded | CommitSlotSessionEnded | CommitSlotMismatch


class AutoApproveService:
    """Manages per-session auto-approve state (in-memory only)."""

    def __init__(self, tombstone: SessionTombstoneStore | None = None) -> None:
        self._activations: dict[tuple[int, str], AutoApproveActivation] = {}
        self._slots: dict[str, ActivationSlot] = {}
        self._active_owners: dict[str, int] = {}
        self._tombstone = tombstone or SessionTombstoneStore()
        self._user_locks: dict[int, asyncio.Lock] = {}
        self._deny_epoch: dict[int, int] = {}
        self._service_lock = asyncio.Lock()

    def per_user_lock(self, user_id: int) -> asyncio.Lock:
        """Return the stable per-user lock, creating it lazily."""
        lock = self._user_locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._user_locks[user_id] = lock
        return lock

    def deny_epoch(self, user_id: int) -> int:
        """Return the current /deny epoch for a user."""
        return self._deny_epoch.get(user_id, 0)

    def is_session_ended(self, session_id: str) -> bool:
        """Check if the session has been tombstoned as ended."""
        return self._tombstone.is_ended(session_id)

    def is_active(self, session_id: str | None = None, *, user_id: int | None = None) -> bool:
        """Check if auto-approve is active.

        New callers should pass both user_id and session_id. The positional
        session-only form is retained for existing call sites and checks whether
        the session has any active owner.
        """
        if session_id is None:
            raise TypeError("session_id is required")
        if user_id is None:
            owner_user_id = self._active_owners.get(session_id)
            return owner_user_id is not None and (owner_user_id, session_id) in self._activations
        return (user_id, session_id) in self._activations

    def is_enabled(self, session_id: str) -> bool:
        """Compatibility alias for is_active(session_id)."""
        return self.is_active(session_id)

    def get_active_user_for_session(self, session_id: str) -> int | None:
        """Return the active auto-approve owner for the session, if any."""
        owner_user_id = self._active_owners.get(session_id)
        if owner_user_id is None or (owner_user_id, session_id) not in self._activations:
            return None
        return owner_user_id

    def get_active_session_for_user(self, user_id: int, session_id: str) -> bool:
        """Check if the given session_id has auto-approve active for the user."""
        return self.is_active(session_id=session_id, user_id=user_id)

    async def activate(self, session_id: str, *, user_id: int) -> None:
        """Compatibility wrapper for legacy callers."""
        async with self._service_lock:
            if self._tombstone.is_ended(session_id):
                logger.warning(
                    "auto-approve activation ignored for ended session",
                    extra={"session_id": session_id, "user_id": user_id},
                )
                return
            self._activate(user_id=user_id, session_id=session_id)

    # Alias: enable == activate
    enable = activate

    async def deactivate(self, session_id: str) -> bool:
        """Compatibility wrapper disabling any active owner for a session."""
        async with self._service_lock:
            keys = [key for key in self._activations if key[1] == session_id]
            deactivated = False
            for user_id, _ in keys:
                deactivated = self._deactivate(user_id=user_id, session_id=session_id) or deactivated
            return deactivated

    # Alias: disable == deactivate
    disable = deactivate

    async def clear_session(
        self,
        session_id: str,
        *,
        mark_ended: bool = True,
    ) -> None:
        """Clear state for a session and optionally mark it as ended."""
        async with self._service_lock:
            self._deactivate_all_for_session_locked(
                session_id,
                mark_ended=mark_ended,
            )

    async def try_claim_slot(self, *, session_id: str, user_id: int) -> SlotClaimResult:
        """Try to reserve the session activation slot for a user."""
        async with self._service_lock:
            owner_user_id = self._active_owners.get(session_id)
            if owner_user_id is not None and self.is_active(session_id=session_id, user_id=owner_user_id):
                if owner_user_id == user_id:
                    return SlotActiveOwnerExists(owner_user_id=user_id)
                return SlotConflict(holder_user_id=owner_user_id)

            slot = self._slots.get(session_id)
            if slot is not None:
                if slot.holder_user_id == user_id:
                    return SlotAlreadyClaimedBySameUser(attempt_id=slot.attempt_id)
                return SlotConflict(holder_user_id=slot.holder_user_id)

            attempt_id = uuid.uuid4().hex
            self._slots[session_id] = ActivationSlot(session_id=session_id, holder_user_id=user_id, attempt_id=attempt_id)
            return SlotClaimed(attempt_id=attempt_id)

    async def commit_slot(self, *, session_id: str, user_id: int, attempt_id: str) -> bool:
        """Commit a held activation slot if both holder and attempt match."""
        async with self._service_lock:
            return self._commit_slot_locked(session_id=session_id, user_id=user_id, attempt_id=attempt_id)

    async def commit_slot_if_session_alive(self, *, session_id: str, user_id: int, attempt_id: str) -> CommitSlotResult:
        """Commit a slot only if the session was not ended before commit."""
        async with self._service_lock:
            if self._tombstone.is_ended(session_id):
                return CommitSlotSessionEnded()

            slot = self._slots.get(session_id)
            if slot is None or slot.holder_user_id != user_id or slot.attempt_id != attempt_id:
                self._log_slot_commit_mismatch(session_id=session_id, user_id=user_id, attempt_id=attempt_id, slot=slot)
                return CommitSlotMismatch()

            self._activate(user_id=user_id, session_id=session_id)
            self._active_owners[session_id] = user_id
            self._slots.pop(session_id, None)
            return CommitSlotSucceeded()

    async def activate_if_session_alive(self, *, user_id: int, session_id: str) -> bool:
        """Activate auto-approve unless the session has already ended."""
        async with self._service_lock:
            if self._tombstone.is_ended(session_id):
                return False
            self._activate(user_id=user_id, session_id=session_id)
            return True

    async def deactivate_and_release_for_user_session(self, *, user_id: int, session_id: str) -> bool:
        """Deactivate one user/session activation and release that user's slot for the session."""
        async with self._service_lock:
            deactivated = self._deactivate(user_id=user_id, session_id=session_id)
            slot = self._slots.get(session_id)
            if slot is not None and slot.holder_user_id == user_id:
                self._slots.pop(session_id, None)
            return deactivated

    async def deactivate_all_for_user(self, user_id: int) -> int:
        """Panic-stop all auto-approve state for a user and advance the deny epoch.

        Lock ordering: per_user_lock must be acquired before _service_lock
        to avoid deadlock. This is the only method that nests both locks.
        """
        async with self.per_user_lock(user_id):
            async with self._service_lock:
                self._release_all_slots_for_user_locked(user_id)
                keys = [key for key in self._activations if key[0] == user_id]
                for _, session_id in keys:
                    self._deactivate(user_id=user_id, session_id=session_id)
                self._deny_epoch[user_id] = self._deny_epoch.get(user_id, 0) + 1
                return len(keys)

    async def deactivate_all_for_session(self, session_id: str) -> int:
        """Deactivate all users for a session, release its slot, and tombstone it as ended."""
        async with self._service_lock:
            return self._deactivate_all_for_session_locked(session_id)

    async def release_slot(self, *, session_id: str, user_id: int, attempt_id: str) -> bool:
        """Release a held activation slot if both holder and attempt match."""
        async with self._service_lock:
            return self._release_slot_locked(session_id=session_id, user_id=user_id, attempt_id=attempt_id)

    async def release_all_slots_for_user(self, user_id: int) -> int:
        """Release every activation slot currently held by the user."""
        async with self._service_lock:
            return self._release_all_slots_for_user_locked(user_id)

    async def release_all_slots_for_session(self, session_id: str) -> int:
        """Release the activation slot for a session, if present."""
        async with self._service_lock:
            if session_id not in self._slots:
                return 0
            self._slots.pop(session_id, None)
            return 1

    def _activate(self, *, user_id: int, session_id: str) -> None:
        self._slots.pop(session_id, None)
        previous_owner_user_id = self._active_owners.get(session_id)
        if previous_owner_user_id is not None and previous_owner_user_id != user_id:
            self._activations.pop((previous_owner_user_id, session_id), None)

        self._activations[(user_id, session_id)] = AutoApproveActivation(
            session_id=session_id,
            user_id=user_id,
            activated_at=datetime.now(UTC),
        )
        self._active_owners[session_id] = user_id
        logger.info("Auto-approve activated for session %s by user %d", session_id, user_id)

    def _deactivate(self, *, user_id: int, session_id: str) -> bool:
        activation = self._activations.pop((user_id, session_id), None)
        if self._active_owners.get(session_id) == user_id:
            self._active_owners.pop(session_id, None)
        if activation is not None:
            logger.info("Auto-approve deactivated for session %s", session_id)
            return True
        return False

    def _commit_slot_locked(self, *, session_id: str, user_id: int, attempt_id: str) -> bool:
        slot = self._slots.get(session_id)
        if slot is None or slot.holder_user_id != user_id or slot.attempt_id != attempt_id:
            self._log_slot_commit_mismatch(session_id=session_id, user_id=user_id, attempt_id=attempt_id, slot=slot)
            return False

        self._activate(user_id=user_id, session_id=session_id)
        self._active_owners[session_id] = user_id
        self._slots.pop(session_id, None)
        return True

    def _release_slot_locked(self, *, session_id: str, user_id: int, attempt_id: str) -> bool:
        slot = self._slots.get(session_id)
        if slot is None or slot.holder_user_id != user_id or slot.attempt_id != attempt_id:
            return False
        self._slots.pop(session_id, None)
        return True

    def _release_all_slots_for_user_locked(self, user_id: int) -> int:
        session_ids = [session_id for session_id, slot in self._slots.items() if slot.holder_user_id == user_id]
        for session_id in session_ids:
            self._slots.pop(session_id, None)
        return len(session_ids)

    def _deactivate_all_for_session_locked(
        self,
        session_id: str,
        *,
        mark_ended: bool = True,
    ) -> int:
        affected_user_ids: set[int] = set()
        owner_user_id = self._active_owners.get(session_id)
        if owner_user_id is not None:
            affected_user_ids.add(owner_user_id)

        slot = self._slots.pop(session_id, None)
        if slot is not None:
            affected_user_ids.add(slot.holder_user_id)

        keys = [key for key in self._activations if key[1] == session_id]
        for user_id, _ in keys:
            affected_user_ids.add(user_id)
            self._deactivate(user_id=user_id, session_id=session_id)

        self._active_owners.pop(session_id, None)
        if mark_ended:
            self._tombstone.mark_ended(session_id)

        for user_id in affected_user_ids:
            self._cleanup_user_lock_if_idle_locked(user_id)

        return len(keys)

    def _cleanup_user_lock_if_idle_locked(self, user_id: int) -> None:
        lock = self._user_locks.get(user_id)
        if lock is not None and lock.locked():
            return
        if any(key[0] == user_id for key in self._activations):
            return
        if any(slot.holder_user_id == user_id for slot in self._slots.values()):
            return
        self._user_locks.pop(user_id, None)

    def _log_slot_commit_mismatch(self, *, session_id: str, user_id: int, attempt_id: str, slot: ActivationSlot | None) -> None:
        logger.warning(
            "auto-approve slot commit mismatch",
            extra={
                "session_id": session_id,
                "user_id": user_id,
                "attempt_id": attempt_id,
                "holder_user_id": slot.holder_user_id if slot is not None else None,
                "holder_attempt_id": slot.attempt_id if slot is not None else None,
            },
        )
