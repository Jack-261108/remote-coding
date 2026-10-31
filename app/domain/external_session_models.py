from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from app.domain.external_terminal_models import TerminalKind


class SessionOrigin(StrEnum):
    TMUX = "tmux"
    EXTERNAL = "external"


@dataclass
class OwnershipResult:
    owner_user_id: int | None
    origin: SessionOrigin
    ownership_state: Literal["owned", "bound", "unbound"]
    binding_id: str | None = None


@dataclass
class UnboundExternalSession:
    session_id: str
    cwd: str
    pid: int | None
    first_seen: datetime
    last_seen: datetime
    event_count: int
    title: str | None = None
    tty: str | None = None


@dataclass
class ExternalTerminalTarget:
    """Stable terminal selected for injecting input to a bound session.

    ``terminal_id`` is the only addressing field — a Ghostty surface UUID or a
    kitty window id as a decimal string. ``kind`` disambiguates the backend
    because ids are only unique within one terminal emulator.
    ``paired_tty`` is the trust anchor used on every send to verify the bound
    Claude process still runs in the same PTY as at pairing time.
    ``binding_id`` records the binding generation at pairing time and acts as
    an ABA barrier: an unbind+rebind that produced a new binding_id invalidates
    a previously-paired target. ``name``/``cwd`` snapshots are display-only and
    must never participate in addressing.
    """

    kind: TerminalKind
    terminal_id: str
    paired_tty: str
    paired_at: datetime
    binding_id: str
    name: str | None = None
    cwd: str | None = None


@dataclass
class ExternalBinding:
    session_id: str
    user_id: int
    cwd: str
    bound_at: datetime
    jsonl_path: str | None
    binding_id: str = field(default_factory=lambda: uuid4().hex)
    pid: int | None = None
    title: str | None = None
    last_activity_at_init: InitVar[datetime | None] = None
    last_activity_at: datetime = field(init=False)
    ended_at: datetime | None = None
    last_pushed_reply_turn_id: str | None = None
    reply_cursor_initialized: bool = False
    tty: str | None = None
    terminal_target: ExternalTerminalTarget | None = None

    def __post_init__(self, last_activity_at_init: datetime | None) -> None:
        # Default activity timestamp to bind time so existing callers that don't
        # pass `last_activity_at` get a sensible non-None value. The stored
        # attribute is always a `datetime`, never None.
        self.last_activity_at = last_activity_at_init if last_activity_at_init is not None else self.bound_at
        if self.last_pushed_reply_turn_id is not None:
            self.reply_cursor_initialized = True


@dataclass
class UnboundPermissionState:
    session_id: str
    tool_use_id: str
    notified_user_ids: list[int]
    responded: bool
    responded_by: int | None
    created_at: datetime


@dataclass
class BindResult:
    success: bool
    message: str
    session_id: str | None = None
    jsonl_path: Path | None = None
    conversation_available: bool = False
