from __future__ import annotations

import asyncio

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule, run_state_machine_as_test

from app.services.auto_approve_service import AutoApproveService, SlotAlreadyClaimedBySameUser, SlotClaimed


class AutoApproveSlotStateMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.service = AutoApproveService()
        self._stale_attempts: dict[str, str] = {}

    def teardown(self) -> None:
        self.loop.close()

    def _call(self, awaitable):  # noqa: ANN001
        return self.loop.run_until_complete(awaitable)

    def _session(self, index: int) -> str:
        return f"session-{index}"

    def _snapshot(self):  # noqa: ANN202
        return (
            dict(self.service._activations),
            dict(self.service._slots),
            dict(self.service._active_owners),
            self.service._tombstone.ended_ids(),
            dict(self.service._deny_epoch),
        )

    def _attempt_for(self, session_id: str, selector: int) -> str:
        slot = self.service._slots.get(session_id)
        if slot is None:
            return f"missing-{selector}"
        if selector % 4 == 0:
            return slot.attempt_id
        stale_attempt_id = self._stale_attempts.get(session_id)
        if selector % 4 == 1 and stale_attempt_id is not None and stale_attempt_id != slot.attempt_id:
            return stale_attempt_id
        return f"wrong-{selector}-{slot.attempt_id}"

    def _remember_stale(self, session_id: str, attempt_id: str) -> None:
        self._stale_attempts[session_id] = attempt_id

    @invariant()
    def state_is_consistent(self) -> None:
        assert len(self.service._slots) == len({slot.session_id for slot in self.service._slots.values()})
        for session_id, slot in self.service._slots.items():
            assert slot.session_id == session_id
            assert self.service._slots[session_id] is slot

        active_sessions: set[str] = set()
        for (user_id, session_id), activation in self.service._activations.items():
            assert activation.user_id == user_id
            assert activation.session_id == session_id
            assert session_id not in active_sessions
            active_sessions.add(session_id)

        for session_id, owner_user_id in self.service._active_owners.items():
            assert (owner_user_id, session_id) in self.service._activations
            assert self.service.get_active_user_for_session(session_id) == owner_user_id

    @rule(session_index=st.integers(min_value=0, max_value=2), user_id=st.integers(min_value=1, max_value=3))
    def try_claim_slot(self, session_index: int, user_id: int) -> None:
        session_id = self._session(session_index)

        result = self._call(self.service.try_claim_slot(session_id=session_id, user_id=user_id))

        if isinstance(result, SlotClaimed):
            slot = self.service._slots[session_id]
            assert slot.holder_user_id == user_id
            assert slot.attempt_id == result.attempt_id
        elif isinstance(result, SlotAlreadyClaimedBySameUser):
            assert self.service._slots[session_id].attempt_id == result.attempt_id

    @rule(
        session_index=st.integers(min_value=0, max_value=2),
        user_id=st.integers(min_value=1, max_value=3),
        attempt_selector=st.integers(min_value=0, max_value=8),
    )
    def commit_slot(self, session_index: int, user_id: int, attempt_selector: int) -> None:
        session_id = self._session(session_index)
        attempt_id = self._attempt_for(session_id, attempt_selector)
        slot_before = self.service._slots.get(session_id)
        before = self._snapshot()
        matches = slot_before is not None and slot_before.holder_user_id == user_id and slot_before.attempt_id == attempt_id

        result = self._call(self.service.commit_slot(session_id=session_id, user_id=user_id, attempt_id=attempt_id))

        if matches:
            assert result is True
            assert session_id not in self.service._slots
            assert self.service.is_active(session_id=session_id, user_id=user_id)
            assert self.service.get_active_user_for_session(session_id) == user_id
            self._remember_stale(session_id, attempt_id)
        else:
            assert result is False
            assert self._snapshot() == before

    @rule(
        session_index=st.integers(min_value=0, max_value=2),
        user_id=st.integers(min_value=1, max_value=3),
        attempt_selector=st.integers(min_value=0, max_value=8),
    )
    def release_slot(self, session_index: int, user_id: int, attempt_selector: int) -> None:
        session_id = self._session(session_index)
        attempt_id = self._attempt_for(session_id, attempt_selector)
        slot_before = self.service._slots.get(session_id)
        before = self._snapshot()
        matches = slot_before is not None and slot_before.holder_user_id == user_id and slot_before.attempt_id == attempt_id

        result = self._call(self.service.release_slot(session_id=session_id, user_id=user_id, attempt_id=attempt_id))

        if matches:
            assert result is True
            assert session_id not in self.service._slots
            self._remember_stale(session_id, attempt_id)
        else:
            assert result is False
            assert self._snapshot() == before

    @rule(user_id=st.integers(min_value=1, max_value=3))
    def release_all_slots_for_user(self, user_id: int) -> None:
        slots_before = [slot for slot in self.service._slots.values() if slot.holder_user_id == user_id]
        activations_before = dict(self.service._activations)
        active_owners_before = dict(self.service._active_owners)

        result = self._call(self.service.release_all_slots_for_user(user_id))

        assert result == len(slots_before)
        assert all(slot.holder_user_id != user_id for slot in self.service._slots.values())
        assert self.service._activations == activations_before
        assert self.service._active_owners == active_owners_before
        for slot in slots_before:
            self._remember_stale(slot.session_id, slot.attempt_id)

    @rule(session_index=st.integers(min_value=0, max_value=2))
    def release_all_slots_for_session(self, session_index: int) -> None:
        session_id = self._session(session_index)
        slot_before = self.service._slots.get(session_id)

        result = self._call(self.service.release_all_slots_for_session(session_id))

        assert result == int(slot_before is not None)
        assert session_id not in self.service._slots
        if slot_before is not None:
            self._remember_stale(slot_before.session_id, slot_before.attempt_id)

    @rule(user_id=st.integers(min_value=1, max_value=3))
    def deactivate_all_for_user(self, user_id: int) -> None:
        active_sessions_before = [session_id for active_user_id, session_id in self.service._activations if active_user_id == user_id]
        slots_before = [slot for slot in self.service._slots.values() if slot.holder_user_id == user_id]
        epoch_before = self.service.deny_epoch(user_id)

        result = self._call(self.service.deactivate_all_for_user(user_id))

        assert result == len(active_sessions_before)
        assert self.service.deny_epoch(user_id) == epoch_before + 1
        assert all(not self.service.is_active(session_id=self._session(index), user_id=user_id) for index in range(3))
        assert all(slot.holder_user_id != user_id for slot in self.service._slots.values())

        for slot in slots_before:
            before = self._snapshot()
            commit_result = self._call(
                self.service.commit_slot(session_id=slot.session_id, user_id=slot.holder_user_id, attempt_id=slot.attempt_id)
            )
            assert commit_result is False
            assert self._snapshot() == before
            self._remember_stale(slot.session_id, slot.attempt_id)


def test_auto_approve_slot_aba_state_machine() -> None:
    run_state_machine_as_test(AutoApproveSlotStateMachine, settings=settings(max_examples=60, stateful_step_count=30, deadline=None))


@pytest.mark.asyncio
async def test_stale_slot_cannot_commit_after_direct_activate_deactivate() -> None:
    service = AutoApproveService()

    claimed = await service.try_claim_slot(session_id="session-1", user_id=1)
    assert isinstance(claimed, SlotClaimed)

    await service.activate("session-1", user_id=2)
    assert service.is_active("session-1", user_id=2)

    assert await service.deactivate("session-1") is True

    assert await service.commit_slot(session_id="session-1", user_id=1, attempt_id=claimed.attempt_id) is False
    assert not service.is_active("session-1", user_id=1)


@pytest.mark.asyncio
async def test_user_lock_cleanup_keeps_locked_idle_lock() -> None:
    service = AutoApproveService()
    lock = service.per_user_lock(1)

    async with lock:
        await service.activate("session-1", user_id=1)
        assert await service.deactivate_all_for_session("session-1") == 1
        assert service._user_locks.get(1) is lock

    async with service._service_lock:
        service._cleanup_user_lock_if_idle_locked(1)

    assert 1 not in service._user_locks


@pytest.mark.asyncio
async def test_user_lock_cleanup_preserves_deny_epoch() -> None:
    service = AutoApproveService()
    await service.deactivate_all_for_user(1)
    assert service.deny_epoch(1) == 1

    await service.activate("session-1", user_id=1)
    assert await service.deactivate_all_for_session("session-1") == 1

    assert service.deny_epoch(1) == 1


@pytest.mark.asyncio
async def test_auto_approve_public_mutators_wait_for_service_lock() -> None:
    service = AutoApproveService()

    async with service._service_lock:
        activate_task = asyncio.create_task(service.activate("session-1", user_id=1))
        await asyncio.sleep(0)
        assert not service.is_active("session-1", user_id=1)

    await activate_task
    assert service.is_active("session-1", user_id=1)

    async with service._service_lock:
        deactivate_task = asyncio.create_task(service.deactivate("session-1"))
        await asyncio.sleep(0)
        assert service.is_active("session-1", user_id=1)

    assert await deactivate_task is True
    assert not service.is_active("session-1", user_id=1)

    await service.activate("session-1", user_id=1)
    async with service._service_lock:
        clear_task = asyncio.create_task(service.clear_session("session-1"))
        await asyncio.sleep(0)
        assert service.is_active("session-1", user_id=1)

    assert await clear_task is None
    assert not service.is_active("session-1", user_id=1)
    assert service.is_session_ended("session-1")


@pytest.mark.asyncio
async def test_auto_approve_public_alias_mutators_are_async() -> None:
    service = AutoApproveService()

    await service.enable("session-1", user_id=1)
    assert service.is_active("session-1", user_id=1)

    assert await service.disable("session-1") is True
    assert not service.is_active("session-1", user_id=1)
