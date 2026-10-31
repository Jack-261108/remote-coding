"""Terminal-agnostic models for external session input injection.

These types are shared by every external terminal backend (Ghostty, kitty).
They deliberately carry no backend-specific behaviour: an adapter produces
``ExternalTerminal`` snapshots and reports a ``TerminalOutcome`` string.

Design: ``docs/specs/2026-10-29-external-kitty-input-design.md`` §1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TerminalKind = Literal["ghostty", "kitty"]


class TerminalOutcome:
    """Stable outcome constants returned by every backend.

    String values are part of the persisted/logged surface — keep them stable.
    The service compares against ``TerminalOutcome.*`` names, never literals.
    """

    OK = "ok"
    NOT_FOUND = "not_found"  # validate_terminal found zero matches
    NOT_UNIQUE = "not_unique"  # matched more than one terminal

    # Ghostty (AppleScript) specific.
    APPLESCRIPT_DISABLED = "applescript_disabled"
    GHOSTTY_NOT_RUNNING = "ghostty_not_running"
    NON_DARWIN = "non_darwin"
    OSASCRIPT_MISSING = "osascript_missing"
    TCC_DENIED = "tcc_denied"

    # kitty (remote control) specific.
    KITTY_NOT_RUNNING = "kitty_not_running"  # socket unreachable
    KITTY_REMOTE_CONTROL_DISABLED = "kitty_remote_control_disabled"  # no listen_on configured
    KITTY_BIN_MISSING = "kitty_bin_missing"

    # Shared.
    TIMEOUT = "timeout"
    INDETERMINATE = "indeterminate"  # partial action applied — never retry
    OS_ERROR = "os_error"


@dataclass(frozen=True, slots=True)
class ExternalTerminal:
    """One addressable terminal surface/tab of a given kind.

    ``terminal_id`` is the ONLY addressing field: a Ghostty surface UUID, or a
    kitty window id rendered as a decimal string. ``name``/``cwd`` are
    display-only snapshots and must never participate in addressing.

    ``pid``/``foreground_pids`` are only populated by backends that expose
    process identity (kitty). They feed candidate *ordering* in the pairing UI
    and never auto-select a target.
    """

    kind: TerminalKind
    terminal_id: str
    name: str | None
    cwd: str | None
    pid: int | None = None
    foreground_pids: tuple[int, ...] = ()
