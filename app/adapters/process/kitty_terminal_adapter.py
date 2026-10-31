"""kitty remote-control adapter for external session input injection.

This adapter is the ONLY component that talks to kitty. It mirrors
``GhosttyTerminalAdapter`` and implements the same
``ExternalTerminalBackendProtocol``, so ``ExternalSessionInputService`` can
drive either terminal emulator through one code path.

Design: ``docs/specs/2026-10-29-external-kitty-input-design.md`` §3.

Transport contract:

  * kitty has no AppleScript dictionary. The only control channel is
    ``kitten @`` remote control, which requires ``allow_remote_control`` plus a
    ``listen_on`` socket. Because this process runs OUTSIDE kitty, the address
    must be passed explicitly as ``--to unix:<socket>``; the ``KITTY_LISTEN_ON``
    environment variable only reaches kitty's own children.
  * ``kitten @ ls`` returns a JSON tree of OS windows -> tabs -> windows. Each
    window carries ``id``/``title``/``cwd``/``pid``/``foreground_processes``.
    The window ``id`` (an int, monotonically increasing and never reused) is
    the only addressing field.
  * ``kitten @ send-text`` interprets Python escapes when the payload arrives
    as argv (``\\e``, ``\\u21fa`` would be rewritten). The payload is therefore
    always piped through ``--stdin``, which kitty forwards as raw bytes.

Known limitation (do not paper over it): ``send-text`` and ``send-key`` never
report match failures — they exit 0 even when no window matched. A window that
closes in the narrow interval between ``validate_terminal`` and the injection
therefore loses the message silently. The service narrows that window by
re-validating immediately before every injection; there is no transport-level
signal to detect it.

Automatic tests monkeypatch ``asyncio.create_subprocess_exec``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
from collections.abc import Sequence

from app.domain.external_terminal_models import ExternalTerminal, TerminalKind, TerminalOutcome

logger = logging.getLogger(__name__)

_SYSTEM_KITTEN = "/Applications/kitty.app/Contents/MacOS/kitten"


def _is_darwin(platform: str | None) -> bool:
    return sys.platform == "darwin" if platform is None else platform == "darwin"


class KittyTerminalAdapter:
    """Encapsulates all ``kitten @`` remote-control subprocess calls.

    Construct with ``enable_remote_control=False``, or without a resolvable
    listen address, to render the adapter unavailable: every method then
    returns a clear reason instead of spawning subprocesses. Socket
    reachability is only discoverable at call time, so ``is_available`` stays
    optimistic and the first call classifies the real failure.
    """

    kind: TerminalKind = "kitty"

    def __init__(
        self,
        *,
        kitten_bin: str = "kitten",
        listen_on: str = "",
        timeout_sec: float = 5.0,
        key_delay_sec: float = 0.05,
        platform_name: str | None = None,
        enable_remote_control: bool = True,
        env_listen_on: str | None = None,
    ) -> None:
        self._kitten_bin = kitten_bin
        self._listen_on = listen_on.strip()
        self._timeout_sec = timeout_sec
        self._key_delay_sec = key_delay_sec
        self._platform_name = platform_name
        self._enable_remote_control = enable_remote_control
        # Injectable for tests; defaults to the live environment.
        self._env_listen_on = env_listen_on

    # ─── availability ────────────────────────────────────────────────

    def _resolved_listen_on(self) -> str | None:
        """Return the socket address, falling back to ``KITTY_LISTEN_ON``.

        The explicit setting wins. The environment variable is a convenience
        for users who launch the gateway from inside a kitty window.
        """
        if self._listen_on:
            return self._listen_on
        env_value = self._env_listen_on if self._env_listen_on is not None else os.environ.get("KITTY_LISTEN_ON", "")
        return env_value.strip() or None

    def _resolved_kitten_bin(self) -> str | None:
        """Resolve ``kitten`` without requiring it on a restricted PATH.

        GUI launchers on macOS often hand the process a PATH that omits
        ``/opt/homebrew/bin``. Only the default name falls back to the app
        bundle's absolute path; custom binaries are used as given.
        """
        if shutil.which(self._kitten_bin) is not None:
            return self._kitten_bin
        if self._kitten_bin == "kitten" and _is_darwin(self._platform_name) and os.access(_SYSTEM_KITTEN, os.X_OK):
            return _SYSTEM_KITTEN
        return None

    def is_available(self) -> bool:
        """Whether a remote-control call can be attempted at all.

        False when remote control is disabled by config, when no listen address
        is configured, or when the ``kitten`` binary cannot be resolved.
        Socket liveness and ``allow_remote_control`` are only knowable at call
        time and are classified from stderr instead.
        """
        if not self._enable_remote_control:
            return False
        if self._resolved_listen_on() is None:
            return False
        return self._resolved_kitten_bin() is not None

    def _unavailable_reason(self) -> str:
        if not self._enable_remote_control:
            return TerminalOutcome.KITTY_REMOTE_CONTROL_DISABLED
        if self._resolved_listen_on() is None:
            return TerminalOutcome.KITTY_REMOTE_CONTROL_DISABLED
        return TerminalOutcome.KITTY_BIN_MISSING

    # ─── public: enumeration ────────────────────────────────────────

    async def list_terminals(self) -> tuple[list[ExternalTerminal] | None, str | None]:
        """Enumerate every kitty window across all OS windows and tabs."""
        if not self.is_available():
            return None, self._unavailable_reason()
        stdout, stderr, returncode = await self._run(["ls"], stdin_data=None)
        if returncode != 0:
            reason = self._classify_failure(stderr or stdout, returncode)
            logger.warning("kitty list_terminals failed: %s", reason, extra={"returncode": returncode})
            return None, reason
        try:
            return _parse_ls_json(stdout), None
        except (ValueError, TypeError, KeyError):
            logger.warning("kitty list_terminals returned unparsable JSON", extra={"returncode": returncode})
            return None, TerminalOutcome.OS_ERROR

    async def validate_terminal(self, terminal_id: str) -> tuple[bool, ExternalTerminal | None, str | None]:
        """Confirm ``terminal_id`` resolves to exactly one live kitty window.

        Matching happens in Python against the full listing (same contract as
        the Ghostty adapter) so the caller also gets the matched snapshot for
        display.
        """
        terminals, reason = await self.list_terminals()
        if terminals is None:
            return False, None, reason
        matches = [t for t in terminals if t.terminal_id == terminal_id]
        if len(matches) != 1:
            return False, None, TerminalOutcome.NOT_FOUND if not matches else TerminalOutcome.NOT_UNIQUE
        return True, matches[0], None

    # ─── public: injection ──────────────────────────────────────────

    async def inject_text(self, terminal_id: str, text: str) -> str:
        """Paste ``text`` into the window then press Enter.

        The text goes through stdin (raw bytes, no escape interpretation). If
        the paste succeeds but Enter fails the result is ``INDETERMINATE``: the
        text may already be sitting in the input line, so the caller must not
        retry.
        """
        if not self.is_available():
            return self._unavailable_reason()
        match = _match_expr(terminal_id)
        stdout, stderr, returncode = await self._run(["send-text", "--match", match, "--stdin"], stdin_data=text)
        if returncode != 0:
            reason = self._classify_inject_failure(stderr or stdout, returncode)
            logger.warning("kitty inject_text failed: %s", reason, extra={"returncode": returncode})
            return reason
        stdout, stderr, returncode = await self._run(["send-key", "--match", match, "enter"], stdin_data=None)
        if returncode != 0:
            reason = self._classify_inject_failure(stderr or stdout, returncode)
            logger.warning(
                "kitty inject_text pasted but Enter failed: %s",
                reason,
                extra={"returncode": returncode},
            )
            return TerminalOutcome.INDETERMINATE
        return TerminalOutcome.OK

    # ─── public: AskUserQuestion actions ────────────────────────────

    async def select_user_question_option(
        self,
        terminal_id: str,
        *,
        option_count: int,
        option_index: int,
        submit_after: bool,
    ) -> str:
        if option_count <= 0:
            raise ValueError("option_count must be positive")
        if option_index < 0 or option_index >= option_count:
            raise ValueError("option_index out of range")
        return await self._apply_user_question_action(
            terminal_id,
            keys=["up"] * (option_count + 1) + ["down"] * option_index + ["enter"],
            final_enter=submit_after,
        )

    async def answer_user_question_with_text(
        self,
        terminal_id: str,
        *,
        option_count: int,
        text: str,
        submit_after: bool,
    ) -> str:
        if option_count < 0:
            raise ValueError("option_count must be non-negative")
        if not text:
            raise ValueError("text must be non-empty")
        return await self._apply_user_question_action(
            terminal_id,
            keys=["up"] * (option_count + 1) + ["down"] * option_count + ["enter"],
            paste_text=text,
            final_enter=submit_after,
        )

    async def advance_user_question_after_multi_select(
        self,
        terminal_id: str,
        *,
        option_count: int,
        final_question: bool,
    ) -> str:
        if option_count <= 0:
            raise ValueError("option_count must be positive")
        return await self._apply_user_question_action(
            terminal_id,
            keys=["up"] * (option_count + 1) + ["right"],
            final_enter=final_question,
        )

    async def _apply_user_question_action(
        self,
        terminal_id: str,
        *,
        keys: Sequence[str],
        final_enter: bool,
        paste_text: str | None = None,
    ) -> str:
        """Send a key sequence, optionally pasting text before the final Enter.

        Mirrors the Ghostty ``_QUESTION_ACTION_SCRIPT``: ``keys`` starts with
        ``up`` × (option_count + 1) so the TUI cursor returns to the form
        controls before any navigation, and a paste is always confirmed with
        its own Enter (exiting the text field) before ``final_enter`` submits
        the form. Every step after the first successful key is INDETERMINATE
        on failure: the TUI has already moved, so the caller cannot retry the
        whole action.
        """
        if not self.is_available():
            return self._unavailable_reason()
        match = _match_expr(terminal_id)
        started = False
        for key in keys:
            outcome = await self._send_key(match, key)
            if outcome != TerminalOutcome.OK:
                return outcome if not started else TerminalOutcome.INDETERMINATE
            started = True
            await asyncio.sleep(self._key_delay_sec)
        if paste_text is not None:
            stdout, stderr, returncode = await self._run(["send-text", "--match", match, "--stdin"], stdin_data=paste_text)
            if returncode != 0:
                reason = self._classify_question_failure(stderr or stdout)
                return reason if not started else TerminalOutcome.INDETERMINATE
            started = True
            await asyncio.sleep(self._key_delay_sec)
            # The paste must be confirmed with Enter even when this is not the
            # final question — otherwise the text stays in the editor and the
            # TUI never advances.
            outcome = await self._send_key(match, "enter")
            if outcome != TerminalOutcome.OK:
                return TerminalOutcome.INDETERMINATE
        if final_enter:
            outcome = await self._send_key(match, "enter")
            if outcome != TerminalOutcome.OK:
                return TerminalOutcome.INDETERMINATE if started else outcome
        return TerminalOutcome.OK

    async def _send_key(self, match: str, key: str) -> str:
        stdout, stderr, returncode = await self._run(["send-key", "--match", match, key], stdin_data=None)
        if returncode == 0:
            return TerminalOutcome.OK
        return self._classify_question_failure(stderr or stdout)

    # ─── internals: subprocess ──────────────────────────────────────

    async def _run(self, args: Sequence[str], *, stdin_data: str | None) -> tuple[str, str, int]:
        """Run ``kitten @ --to <addr> <args...>`` with a hard timeout.

        Returns ``(stdout, stderr, returncode)``. On timeout the child is
        killed and reaped within the same budget, and a sentinel
        ``("", "timeout", -1)`` is returned so callers classify it as TIMEOUT.
        """
        kitten_bin = self._resolved_kitten_bin()
        listen_on = self._resolved_listen_on()
        if kitten_bin is None:
            return "", "kitten missing", 127
        if listen_on is None:
            return "", "no listen_on configured", 127

        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                kitten_bin,
                "@",
                "--to",
                listen_on,
                *args,
                stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(stdin_data.encode() if stdin_data is not None else None),
                self._timeout_sec,
            )
        except TimeoutError:
            if proc is not None:
                await self._kill(proc)
            return "", "timeout", -1
        except asyncio.CancelledError:
            if proc is not None:
                await self._kill(proc)
            raise
        except OSError:
            # FileNotFoundError (binary absent) and PermissionError (custom
            # KITTY_BIN without +x) both mean "kitten cannot be started".
            return "", "kitten spawn failed", 127
        return stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace"), proc.returncode or 0

    async def _kill(self, proc: asyncio.subprocess.Process) -> None:
        """Kill and bounded-reap a wedged ``kitten`` child.

        A stuck remote-control call must not pin the caller's input lock; if
        the reap does not finish within the command budget the OS is left to
        collect it.
        """
        try:
            proc.kill()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), self._timeout_sec)
        except TimeoutError:
            logger.warning(
                "kitty kitten child did not die %ss after kill; abandoning reap",
                self._timeout_sec,
            )
        except Exception:  # pragma: no cover - reap best-effort
            logger.debug("kitty kitten wait after kill swallowed error")

    # ─── internals: failure classification ──────────────────────────

    @staticmethod
    def _classify_failure(message: str, returncode: int) -> str:
        """Classify a ``ls`` failure into a stable reason."""
        text = (message or "").lower()
        if "kitten missing" in text or "kitten spawn failed" in text:
            return TerminalOutcome.KITTY_BIN_MISSING
        if "no listen_on configured" in text:
            return TerminalOutcome.KITTY_REMOTE_CONTROL_DISABLED
        if "failed to connect" in text or "no such file or directory" in text or "connection refused" in text:
            return TerminalOutcome.KITTY_NOT_RUNNING
        if "allow_remote_control" in text or "permission denied" in text or "remote control is not allowed" in text:
            return TerminalOutcome.KITTY_REMOTE_CONTROL_DISABLED
        if "timeout" in text or returncode == -1:
            return TerminalOutcome.TIMEOUT
        return TerminalOutcome.OS_ERROR

    @staticmethod
    def _classify_inject_failure(message: str, returncode: int) -> str:
        """Classify a text-injection failure.

        Pre-flight failures (socket down, remote control off, binary missing)
        map to their retryable reasons. Anything else is treated as
        indeterminate, because kitty gives no signal about whether the bytes
        reached the child.
        """
        text = (message or "").lower()
        if "kitten missing" in text or "kitten spawn failed" in text:
            return TerminalOutcome.KITTY_BIN_MISSING
        if "no listen_on configured" in text:
            return TerminalOutcome.KITTY_REMOTE_CONTROL_DISABLED
        if "failed to connect" in text or "no such file or directory" in text or "connection refused" in text:
            return TerminalOutcome.KITTY_NOT_RUNNING
        if "allow_remote_control" in text or "permission denied" in text or "remote control is not allowed" in text:
            return TerminalOutcome.KITTY_REMOTE_CONTROL_DISABLED
        if "timeout" in text or returncode == -1:
            return TerminalOutcome.TIMEOUT
        return TerminalOutcome.INDETERMINATE

    @staticmethod
    def _classify_question_failure(message: str) -> str:
        """Classify a single key send inside a multi-key question action.

        Unlike text injection, a failed key here may already have moved the
        TUI cursor, so unknown failures are indeterminate.
        """
        text = (message or "").lower()
        if "kitten missing" in text or "kitten spawn failed" in text:
            return TerminalOutcome.KITTY_BIN_MISSING
        if "no listen_on configured" in text:
            return TerminalOutcome.KITTY_REMOTE_CONTROL_DISABLED
        if "failed to connect" in text or "no such file or directory" in text or "connection refused" in text:
            return TerminalOutcome.KITTY_NOT_RUNNING
        if "allow_remote_control" in text or "permission denied" in text or "remote control is not allowed" in text:
            return TerminalOutcome.KITTY_REMOTE_CONTROL_DISABLED
        return TerminalOutcome.INDETERMINATE


def _match_expr(terminal_id: str) -> str:
    """Build a kitty window match expression from a validated id.

    The service only ever passes an id that came back from ``ls``, so this is
    a plain decimal string; reject anything else rather than letting an
    arbitrary expression reach kitty's match parser.
    """
    if not terminal_id.isdigit():
        raise ValueError(f"invalid kitty window id: {terminal_id!r}")
    return f"id:{terminal_id}"


def _parse_ls_json(raw: str) -> list[ExternalTerminal]:
    """Flatten ``kitten @ ls`` JSON into one entry per window.

    Tolerates missing/typed-wrong fields by skipping them rather than failing
    the whole listing: a single odd window must not hide the others from the
    pairing UI.
    """
    data = json.loads(raw)
    if not isinstance(data, list):
        raise TypeError("kitty ls output must be a JSON list")
    terminals: list[ExternalTerminal] = []
    for os_window in data:
        if not isinstance(os_window, dict):
            continue
        for tab in os_window.get("tabs") or ():
            if not isinstance(tab, dict):
                continue
            for window in tab.get("windows") or ():
                if not isinstance(window, dict):
                    continue
                window_id = window.get("id")
                if not isinstance(window_id, int):
                    continue
                foreground = window.get("foreground_processes")
                foreground_pids: tuple[int, ...] = ()
                if isinstance(foreground, list):
                    foreground_pids = tuple(
                        item["pid"] for item in foreground if isinstance(item, dict) and isinstance(item.get("pid"), int)
                    )
                pid = window.get("pid")
                cwd = window.get("cwd")
                title = window.get("title")
                terminals.append(
                    ExternalTerminal(
                        kind="kitty",
                        terminal_id=str(window_id),
                        name=title if isinstance(title, str) and title else None,
                        cwd=cwd if isinstance(cwd, str) and cwd else None,
                        pid=pid if isinstance(pid, int) else None,
                        foreground_pids=foreground_pids,
                    )
                )
    return terminals
