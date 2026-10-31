"""kitty adapter test double used by external-input dispatch tests."""

from __future__ import annotations

from collections import deque

from app.domain.external_terminal_models import ExternalTerminal, TerminalOutcome


class FakeKittyTerminalAdapter:
    """Same surface as ``FakeGhosttyTerminalAdapter`` but reports ``kind="kitty"``.

    Only the members the service actually calls are implemented, so a test can
    assert the service never routes a ghostty target here (and vice versa).
    """

    kind = "kitty"

    def __init__(
        self,
        *,
        available: bool = True,
        terminals: list[ExternalTerminal] | None = None,
    ) -> None:
        self.available = available
        self.terminals = (
            terminals
            if terminals is not None
            else [
                ExternalTerminal(
                    kind="kitty",
                    terminal_id="101",
                    name="claude — project",
                    cwd="/project",
                    pid=1234,
                    foreground_pids=(1234,),
                )
            ]
        )
        self.list_error: str | None = None
        self.validate_error: str | None = None
        self.inject_error: str | None = None
        self.inject_outcomes: deque[str] = deque()
        self.inject_calls: list[tuple[str, str]] = []
        self.question_calls: list[tuple[str, str]] = []
        self.validate_calls: list[str] = []

    def is_available(self) -> bool:
        return self.available

    async def list_terminals(self) -> tuple[list[ExternalTerminal] | None, str | None]:
        if not self.available:
            return None, "unavailable"
        if self.list_error is not None:
            return None, self.list_error
        return list(self.terminals), None

    async def validate_terminal(self, terminal_id: str) -> tuple[bool, ExternalTerminal | None, str | None]:
        self.validate_calls.append(terminal_id)
        if not self.available:
            return False, None, "unavailable"
        if self.validate_error is not None:
            return False, None, self.validate_error
        matches = [terminal for terminal in self.terminals if terminal.terminal_id == terminal_id]
        if len(matches) == 1:
            return True, matches[0], None
        if not matches:
            return False, None, TerminalOutcome.NOT_FOUND
        return False, None, TerminalOutcome.NOT_UNIQUE

    async def inject_text(self, terminal_id: str, text: str) -> str:
        self.inject_calls.append((terminal_id, text))
        if self.inject_error is not None:
            return self.inject_error
        return self.inject_outcomes.popleft() if self.inject_outcomes else TerminalOutcome.OK

    async def select_user_question_option(
        self,
        terminal_id: str,
        *,
        option_count: int,
        option_index: int,
        submit_after: bool,
    ) -> str:
        self.question_calls.append((terminal_id, "select"))
        return TerminalOutcome.OK

    async def answer_user_question_with_text(
        self,
        terminal_id: str,
        *,
        option_count: int,
        text: str,
        submit_after: bool,
    ) -> str:
        self.question_calls.append((terminal_id, "answer_text"))
        return TerminalOutcome.OK

    async def advance_user_question_after_multi_select(
        self,
        terminal_id: str,
        *,
        option_count: int,
        final_question: bool,
    ) -> str:
        self.question_calls.append((terminal_id, "advance_multi"))
        return TerminalOutcome.OK
