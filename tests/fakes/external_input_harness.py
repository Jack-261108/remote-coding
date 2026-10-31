"""Shared assembly harness for ExternalSessionInputService tests.

Extracted from test_external_session_input_service.py's ``make_harness`` so the
handler-level tests can reuse the same wiring. Builds a real binding store on
disk (tmp), an optional Ghostty pairing target, session store with one seeded
session, and the service with fakes for adapter/probe and a notices recorder.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.adapters.storage.file_session_store import FileSessionStore
from app.domain.external_session_models import ExternalBinding, ExternalTerminalTarget
from app.domain.external_terminal_models import TerminalKind
from app.domain.models import utc_now
from app.domain.session_models import SessionPhase
from app.infra.lock_registry import RefCountedLockRegistry
from app.services.external_binding_store import ExternalBindingStore
from app.services.external_input_mode_state import ExternalInputTargetStore
from app.services.external_input_queue import ExternalInputQueue
from app.services.external_session_input_service import ExternalSessionInputService
from app.services.external_user_question_state import ExternalUserQuestionState
from app.services.pairing_callback_registry import PairingCallbackRegistry
from app.services.session_store import SessionStore
from tests.fakes.ghostty import FakeGhosttyTerminalAdapter
from tests.fakes.process_probe import FakeLocalProcessProbe


@dataclass
class InputHarness:
    service: ExternalSessionInputService
    binding_store: ExternalBindingStore
    session_store: SessionStore
    mode_store: ExternalInputTargetStore
    queue: ExternalInputQueue
    pairing: PairingCallbackRegistry
    adapter: FakeGhosttyTerminalAdapter
    probe: FakeLocalProcessProbe
    external_uq_state: ExternalUserQuestionState
    binding: ExternalBinding
    notices: list[tuple[int, str]] = field(default_factory=list)


def build_input_harness(
    tmp_path: Path,
    *,
    name: str,
    session_id: str = "session-1",
    user_id: int = 42,
    phase: SessionPhase = SessionPhase.IDLE,
    paired: bool = True,
    enabled: bool = True,
    adapter: FakeGhosttyTerminalAdapter | None = None,
    extra_backends: dict[TerminalKind, object] | None = None,
    paired_kind: TerminalKind = "ghostty",
    paired_terminal_id: str = "term-1",
    probe: FakeLocalProcessProbe | None = None,
    queue_max_size: int = 5,
    drain_wait: float = 0.05,
    in_flight_ttl: float = 2.0,
    monotonic: Callable[[], float] | None = None,
) -> InputHarness:
    """Assemble one harness under ``tmp_path / name``; caller picks a unique name."""
    root = tmp_path / name
    binding_store = ExternalBindingStore(root / "binding")
    binding = ExternalBinding(
        session_id=session_id,
        user_id=user_id,
        cwd="/project",
        bound_at=utc_now(),
        jsonl_path=None,
        binding_id=f"binding-{name}",
        pid=1234,
        tty="/dev/ttys005",
    )
    if paired:
        binding.terminal_target = ExternalTerminalTarget(
            kind=paired_kind,
            terminal_id=paired_terminal_id,
            paired_tty="/dev/ttys005",
            paired_at=utc_now(),
            binding_id=binding.binding_id,
            name="claude — project",
            cwd="/project",
        )
    binding_store.save_binding(binding)

    session_store = SessionStore(FileSessionStore(str(root / "state")))
    state = session_store.get_or_create(
        session_id=session_id,
        user_id=user_id,
        workdir="/project",
        claude_session_id=session_id,
    )
    state.phase = phase
    session_store.save(state)

    resolved_adapter = adapter or FakeGhosttyTerminalAdapter()
    resolved_probe = probe or FakeLocalProcessProbe()
    pairing = PairingCallbackRegistry(ttl_sec=60)
    external_uq_state = ExternalUserQuestionState(ttl_sec=60)
    mode_store = ExternalInputTargetStore()
    queue = ExternalInputQueue(max_size=queue_max_size, ttl_sec=60, monotonic=monotonic)
    locks = RefCountedLockRegistry(
        ttl_sec=60,
        cleanup_interval_sec=60,
        cleanup_batch_size=50,
    )
    notices: list[tuple[int, str]] = []

    async def _notify_spy(*, user_id: int, text: str) -> bool:
        notices.append((user_id, text))
        return True

    backends: dict[TerminalKind, object] = {"ghostty": resolved_adapter}
    backends.update(extra_backends or {})

    service = ExternalSessionInputService(
        enabled=enabled,
        binding_store=binding_store,
        session_store=session_store,
        backends=backends,  # type: ignore[arg-type]
        process_probe=resolved_probe,  # type: ignore[arg-type]
        pairing_registry=pairing,
        input_mode_store=mode_store,
        input_queue=queue,
        input_locks=locks,
        external_user_question_state=external_uq_state,
        drain_publish_wait_timeout_sec=drain_wait,
        in_flight_ttl_sec=in_flight_ttl,
        notify_user=_notify_spy,
    )
    return InputHarness(
        service=service,
        binding_store=binding_store,
        session_store=session_store,
        mode_store=mode_store,
        queue=queue,
        pairing=pairing,
        adapter=resolved_adapter,
        probe=resolved_probe,
        external_uq_state=external_uq_state,
        binding=binding,
        notices=notices,
    )
