"""Ghostty AppleScript adapter for external session input injection.

This adapter is the ONLY component that talks to Ghostty via AppleScript. It
provides three capabilities used by ``ExternalSessionInputService``:

  * ``list_terminals`` — enumerate every Ghostty terminal surface (across all
    windows/tabs) for the pairing candidate display. Returns ``terminal.id``
    (a stable surface UUID) plus display-only title/cwd.
  * ``validate_terminal`` — confirm a previously-paired ``terminal_id`` UUID
    still resolves to exactly one live surface.
  * ``inject_text`` — paste ``text`` into the terminal identified by a full
    UUID, then send Enter, in a single ``osascript`` call.

Security contract (see docs/specs/2026-08-03-external-ghostty-input-design.md):

  * The AppleScript source is a fixed constant. The terminal UUID and the user
    text are passed as ``osascript`` argv (``-- <uuid> <text>``) and NEVER
    interpolated into the script source. This is the same separation used by
    ``TmuxRunner.reveal_terminal`` (``tmux_runner.py:1042-1076``): the payload
    rides argv while the script body stays a literal.
  * No ``shell=True``, no ``do script``, no ``activate``/``focus``/``front
    window``. The caller already validated the target by UUID; the adapter
    never guesses a terminal from cwd/title/focus.
  * Failures are fail-closed: a non-unique or missing UUID, a TCC denial, a
    timeout or any OS error yields a non-OK outcome and the caller refuses to
    send. ``INDETERMINATE`` (text may be pasted but Enter failed) is reported
    distinctly and the adapter never retries.

The adapter does NOT depend on a GUI or TCC grant to import — those are only
needed at call time. Automatic tests monkeypatch ``asyncio.create_subprocess_exec``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
from collections.abc import Iterable

from app.domain.external_terminal_models import ExternalTerminal, TerminalKind, TerminalOutcome

logger = logging.getLogger(__name__)

_SYSTEM_OSASCRIPT = "/usr/bin/osascript"


# Fixed AppleScript: enumerate every terminal's id/name/cwd as tab/newline
# delimited rows. Ghostty exposes ``working directory`` as path text already;
# wrapping it in ``POSIX path of`` raises AppleScript -1700 on current Ghostty.
# We do NOT rely on Ghostty's ``every terminal whose id is …`` counting here —
# we list-and-match in Python so the pairing UI can show the matched snapshot.
_LIST_SCRIPT = r"""
on run argv
    set output to ""
    tell application "Ghostty"
        repeat with w in windows
            repeat with t in tabs of w
                repeat with term in (terminals of t)
                    set output to output & (id of term) & "\t" & (name of term) & "\t" & ((working directory of term) as text) & "\n"
                end repeat
            end repeat
        end repeat
    end tell
    return output
end run
"""

# Fixed AppleScript: paste payload into the terminal whose id == targetId,
# requiring exactly one match, then send Enter. targetId and payload arrive
# as argv items 1 and 2 (never as text inside the script source).
_INJECT_SCRIPT = r"""
on run argv
    set targetId to (item 1 of argv)
    set payload to (item 2 of argv)
    tell application "Ghostty"
        set matches to (every terminal whose id is targetId)
        if (count of matches) is not 1 then
            error "terminal not unique"
        end if
        set targetTerminal to item 1 of matches
        input text payload to targetTerminal
        delay 0.1
        send key "enter" to targetTerminal
    end tell
end run
"""

# Fixed AskUserQuestion actions. Dynamic values are argv-only; callers cannot
# supply arbitrary key names or AppleScript. argv fields:
# terminal_id, action, option_count, option_index, final_flag, answer_text.
_QUESTION_ACTION_SCRIPT = r"""
on run argv
    set targetId to (item 1 of argv)
    set actionName to (item 2 of argv)
    set optionCount to (item 3 of argv) as integer
    set optionIndex to (item 4 of argv) as integer
    set finalFlag to (item 5 of argv)
    set answerText to (item 6 of argv)

    if optionCount < 0 then error "invalid option count"
    if actionName is "select" and (optionIndex < 0 or optionIndex >= optionCount) then error "invalid option index"
    if actionName is not "select" and actionName is not "answer_text" and actionName is not "advance_multi" then error "invalid question action"

    tell application "Ghostty"
        set matches to (every terminal whose id is targetId)
        if (count of matches) is not 1 then error "terminal not unique"
        set targetTerminal to item 1 of matches

        repeat (optionCount + 1) times
            send key "arrowUp" to targetTerminal
            delay 0.05
        end repeat

        if actionName is "select" then
            repeat optionIndex times
                send key "arrowDown" to targetTerminal
                delay 0.05
            end repeat
            send key "enter" to targetTerminal
            if finalFlag is "1" then
                delay 0.15
                send key "enter" to targetTerminal
            end if
        else if actionName is "answer_text" then
            repeat optionCount times
                send key "arrowDown" to targetTerminal
                delay 0.05
            end repeat
            send key "enter" to targetTerminal
            delay 0.15
            input text answerText to targetTerminal
            delay 0.1
            send key "enter" to targetTerminal
            if finalFlag is "1" then
                delay 0.15
                send key "enter" to targetTerminal
            end if
        else
            send key "arrowRight" to targetTerminal
            if finalFlag is "1" then
                delay 0.15
                send key "enter" to targetTerminal
            end if
        end if
    end tell
end run
"""


def _is_darwin(platform: str | None) -> bool:
    return sys.platform == "darwin" if platform is None else platform == "darwin"


def _parse_terminals(raw: str) -> list[ExternalTerminal]:
    """Parse the tab/newline-delimited output of ``_LIST_SCRIPT``.

    Each line is ``id\tname\tcwd``; empty lines are skipped. ``name``/``cwd``
    may themselves contain tabs or newlines (rare), but AppleScript
    concatenation here uses our own delimiters so a tab inside a title would
    split into extra columns — we join any surplus columns back into the last
    field conservatively: id is column 0, name is column 1, cwd is the
    remainder joined by tab. Newlines inside titles are not produced by this
    listing because ``return output`` flattens to a single string with our
    explicit ``\n`` separators.
    """
    terminals: list[ExternalTerminal] = []
    for line in raw.split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        terminal_id = parts[0]
        name = parts[1] if len(parts) > 1 else None
        cwd = "\t".join(parts[2:]) if len(parts) > 2 else None
        terminals.append(
            ExternalTerminal(
                kind="ghostty",
                terminal_id=terminal_id,
                name=name if name != "" else None,
                cwd=cwd if cwd != "" else None,
            )
        )
    return terminals


class GhosttyTerminalAdapter:
    """Encapsulates all Ghostty AppleScript subprocess calls.

    Construct with ``enable_applescript=False`` (or when not on macOS, or when
    ``osascript`` is absent) to render the adapter unavailable — every method
    then returns a clear unavailable reason instead of attempting subprocess
    calls. This lets the rest of the system keep the binding and reply-push
    features working when input injection is off.

    Implements ``ExternalTerminalBackendProtocol``; see
    ``docs/specs/2026-10-29-external-kitty-input-design.md``.
    """

    kind: TerminalKind = "ghostty"

    def __init__(
        self,
        *,
        osascript_bin: str = "osascript",
        timeout_sec: float = 5.0,
        platform_name: str | None = None,
        enable_applescript: bool = True,
    ) -> None:
        self._osascript_bin = osascript_bin
        self._timeout_sec = timeout_sec
        self._platform_name = platform_name
        self._enable_applescript = enable_applescript

    def _resolved_osascript_bin(self) -> str | None:
        """Resolve the executable without requiring ``/usr/bin`` on PATH.

        Launch agents and GUI wrappers on macOS may provide a restricted PATH
        that omits ``/usr/bin`` even though the system binary exists there.
        Preserve custom binaries and normal PATH lookup; only the default name
        falls back to the standard macOS absolute path.
        """
        if shutil.which(self._osascript_bin) is not None:
            return self._osascript_bin
        if self._osascript_bin == "osascript" and _is_darwin(self._platform_name) and os.access(_SYSTEM_OSASCRIPT, os.X_OK):
            return _SYSTEM_OSASCRIPT
        return None

    def is_available(self) -> bool:
        """Return whether the adapter can attempt an AppleScript call.

        False on non-darwin, when AppleScript is disabled by config, or when
        ``osascript`` is unavailable via PATH and ``/usr/bin/osascript``.
        TCC/automation permission is only discoverable at call time.
        """
        if not self._enable_applescript:
            return False
        if not _is_darwin(self._platform_name):
            return False
        return self._resolved_osascript_bin() is not None

    def _unavailable_reason(self) -> str:
        if not self._enable_applescript:
            return TerminalOutcome.APPLESCRIPT_DISABLED
        if not _is_darwin(self._platform_name):
            return TerminalOutcome.NON_DARWIN
        return TerminalOutcome.OSASCRIPT_MISSING

    async def list_terminals(self) -> tuple[list[ExternalTerminal] | None, str | None]:
        """Enumerate Ghostty terminals.

        Returns ``(terminals, None)`` on success, or ``(None, reason)`` when the
        adapter is unavailable or Ghostty/AppleScript failed. Distinct failure
        reasons let the service show "Ghostty not running" vs "permission
        denied" without retrying.
        """
        if not self.is_available():
            return None, self._unavailable_reason()
        stdout, stderr, returncode = await self._run_script(_LIST_SCRIPT, [])
        if returncode == 0:
            return _parse_terminals(stdout), None
        reason = self._classify_failure(stderr or stdout, returncode)
        logger.warning("ghostty list_terminals failed: %s", reason, extra={"returncode": returncode})
        return None, reason

    async def validate_terminal(self, terminal_id: str) -> tuple[bool, ExternalTerminal | None, str | None]:
        """Confirm ``terminal_id`` resolves to exactly one live surface.

        Implemented by listing all terminals and matching the full UUID in
        Python (rather than trusting AppleScript's ``every terminal whose
        id`` counting), so we also return the matched snapshot for display.
        Returns ``(False, None, reason)`` on failure/zero-match.
        """
        terminals, reason = await self.list_terminals()
        if terminals is None:
            return False, None, reason
        matches = [t for t in terminals if t.terminal_id == terminal_id]
        if len(matches) != 1:
            return False, None, TerminalOutcome.NOT_FOUND if not matches else TerminalOutcome.NOT_UNIQUE
        return True, matches[0], None

    async def inject_text(self, terminal_id: str, text: str) -> str:
        """Inject ``text`` + Enter into the terminal with ``terminal_id``.

        Returns an ``TerminalOutcome`` constant. Caller must have already
        validated the target; this method does not re-list (an inject-call hit
        on a stale UUID yields ``NOT_UNIQUE``/``NOT_FOUND`` from the script,
        classified below). On ``INDETERMINATE`` the caller must NOT retry.
        """
        if not self.is_available():
            return self._unavailable_reason()
        stdout, stderr, returncode = await self._run_script(_INJECT_SCRIPT, [terminal_id, text])
        if returncode == 0:
            return TerminalOutcome.OK
        reason = self._classify_inject_failure(stderr or stdout, returncode)
        logger.warning("ghostty inject_text failed: %s", reason, extra={"returncode": returncode})
        return reason

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
            action="select",
            option_count=option_count,
            option_index=option_index,
            submit_after=submit_after,
            text="",
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
            action="answer_text",
            option_count=option_count,
            option_index=-1,
            submit_after=submit_after,
            text=text,
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
            action="advance_multi",
            option_count=option_count,
            option_index=-1,
            submit_after=final_question,
            text="",
        )

    async def _apply_user_question_action(
        self,
        terminal_id: str,
        *,
        action: str,
        option_count: int,
        option_index: int,
        submit_after: bool,
        text: str,
    ) -> str:
        if not self.is_available():
            return self._unavailable_reason()
        stdout, stderr, returncode = await self._run_script(
            _QUESTION_ACTION_SCRIPT,
            [
                terminal_id,
                action,
                str(option_count),
                str(option_index),
                "1" if submit_after else "0",
                text,
            ],
        )
        if returncode == 0:
            return TerminalOutcome.OK
        reason = self._classify_question_failure(stderr or stdout, returncode)
        logger.warning(
            "ghostty user-question action failed: %s",
            reason,
            extra={"returncode": returncode, "action": action},
        )
        return reason

    async def _run_script(self, script: str, argv: Iterable[str]) -> tuple[str, str, int]:
        """Run ``osascript -e <script> -- <argv...>`` with a hard timeout.

        Returns ``(stdout, stderr, returncode)``. On timeout the child is
        killed and reaped, and a sentinel empty result with returncode -1 is
        returned so callers classify it as TIMEOUT. A missing ``osascript``
        binary (FileNotFoundError at spawn) yields ``(empty, "osascript
        missing", 127)`` so callers classify it as unavailable/os-error.
        """
        osascript_bin = self._resolved_osascript_bin()
        if osascript_bin is None:
            return "", "osascript missing", 127

        argv_list = list(argv)
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                osascript_bin,
                "-e",
                script,
                "--",
                *argv_list,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), self._timeout_sec)
        except TimeoutError:
            if proc is not None:
                await self._kill(proc)
            return "", "timeout", -1
        except asyncio.CancelledError:
            if proc is not None:
                await self._kill(proc)
            raise
        except FileNotFoundError:
            return "", "osascript missing", 127
        return stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace"), proc.returncode or 0

    async def _kill(self, proc: asyncio.subprocess.Process) -> None:
        try:
            proc.kill()
        except ProcessLookupError:
            return
        # Reap the killed child, but bound the wait so a wedged osascript
        # (uninterruptible 'D' state, or a grandchild holding the stdout/stderr
        # pipe after a stuck TCC prompt) cannot pin the caller's input-lock
        # await context indefinitely. If reaping does not complete within the
        # same budget as the original command timeout, give up and let the OS
        # reap it later rather than stalling every subsequent send for this
        # session behind an unreachable lock.
        try:
            await asyncio.wait_for(proc.wait(), self._timeout_sec)
        except TimeoutError:
            logger.warning(
                "ghostty osascript child did not die %ss after kill; abandoning reap",
                self._timeout_sec,
            )
        except Exception:  # pragma: no cover - reap best-effort
            logger.debug("ghostty osascript wait after kill swallowed error")

    @staticmethod
    def _classify_failure(message: str, returncode: int) -> str:
        """Classify a list_terminals failure text into a stable reason."""
        text = (message or "").lower()
        if "osascript missing" in text:
            return TerminalOutcome.OS_ERROR
        if "not allowed" in text or "automation" in text or "appleevent" in text or "-1743" in text:
            return TerminalOutcome.TCC_DENIED
        if "can't get application" in text or "no such application" in text or "-1728" in text:
            return TerminalOutcome.GHOSTTY_NOT_RUNNING
        if "timed out" in text or "timeout" in text or "-1712" in text:
            return TerminalOutcome.TIMEOUT
        if not message:
            return TerminalOutcome.NOT_FOUND if returncode != 0 else TerminalOutcome.OK
        return TerminalOutcome.OS_ERROR

    @staticmethod
    def _classify_question_failure(message: str, returncode: int) -> str:
        """Classify a multi-key question action; unknown post-start errors are indeterminate."""
        text = (message or "").lower()
        if "osascript missing" in text:
            return TerminalOutcome.OS_ERROR
        if "can't get application" in text or "no such application" in text or "-1728" in text:
            return TerminalOutcome.GHOSTTY_NOT_RUNNING
        if "not unique" in text:
            return TerminalOutcome.NOT_UNIQUE
        if "not allowed" in text or "automation" in text or "appleevent" in text or "-1743" in text:
            return TerminalOutcome.TCC_DENIED
        if "invalid option" in text or "invalid question action" in text:
            return TerminalOutcome.OS_ERROR
        if returncode == -1 or "timed out" in text or "timeout" in text or "-1712" in text:
            return TerminalOutcome.INDETERMINATE
        return TerminalOutcome.INDETERMINATE

    @staticmethod
    def _classify_inject_failure(message: str, returncode: int) -> str:
        """Classify an inject_text failure.

        A failure AFTER ``input text`` could mean the text was pasted but
        ``send key`` failed — we conservatively mark those INDETERMINATE so
        the caller never retries (no duplicate input). Pure pre-flight
        failures (target not unique, TCC, Ghostty down) map to their
        non-indeterminate reasons.
        """
        text = (message or "").lower()
        # A missing osascript binary is a known availability failure, never
        # indeterminate (the caller would otherwise think the text is pasted).
        if "osascript missing" in text:
            return TerminalOutcome.OS_ERROR
        # Ghostty-down must be checked before the generic "can't get" branch
        # (Can't get application "Ghostty" also contains "can't get").
        if "can't get application" in text or "no such application" in text or "-1728" in text:
            return TerminalOutcome.GHOSTTY_NOT_RUNNING
        if "not unique" in text:
            return TerminalOutcome.NOT_UNIQUE
        if "can't get" in text:
            # Pre-flight: nothing was pasted.
            return TerminalOutcome.NOT_FOUND
        if "not allowed" in text or "automation" in text or "appleevent" in text or "-1743" in text:
            return TerminalOutcome.TCC_DENIED
        if "timed out" in text or "timeout" in text or "-1712" in text:
            return TerminalOutcome.TIMEOUT
        if message:
            # Ambiguous post-paste failure: treat as indeterminate, do not retry.
            logger.info("ghostty inject classified indeterminate: rc=%s msg=%r", returncode, message)
            return TerminalOutcome.INDETERMINATE
        return TerminalOutcome.OS_ERROR
