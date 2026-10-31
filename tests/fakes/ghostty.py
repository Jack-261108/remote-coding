"""Ghostty adapter test double used by external-input service tests."""

from __future__ import annotations

import asyncio
from collections import deque

from app.adapters.process.ghostty_terminal_adapter import ExternalTerminal, TerminalOutcome
from app.domain.external_terminal_models import TerminalKind


class FakeGhosttyTerminalAdapter:
    kind: TerminalKind = "ghostty"

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
                    kind="ghostty",
                    terminal_id="term-1",
                    name="claude — project",
                    cwd="/project",
                )
            ]
        )
        self.list_error: str | None = None
        self.validate_error: str | None = None
        self.inject_error: str | None = None
        self.inject_outcomes: deque[str] = deque()
        self.inject_calls: list[tuple[str, str]] = []
        self.question_outcomes: deque[str] = deque()
        self.question_calls: list[tuple[str, str, int, int, bool, str]] = []
        self.validate_calls: list[str] = []
        self.validate_entered: asyncio.Event | None = None
        self.validate_release: asyncio.Event | None = None
        self.inject_entered: asyncio.Event | None = None
        self.inject_release: asyncio.Event | None = None
        self.question_entered: asyncio.Event | None = None
        self.question_release: asyncio.Event | None = None
        self.active_injections = 0
        self.max_active_injections = 0

    def is_available(self) -> bool:
        return self.available

    async def list_terminals(self) -> tuple[list[ExternalTerminal] | None, str | None]:
        if not self.available:
            return None, "unavailable"
        if self.list_error is not None:
            return None, self.list_error
        return list(self.terminals), None

    async def validate_terminal(
        self,
        terminal_id: str,
    ) -> tuple[bool, ExternalTerminal | None, str | None]:
        self.validate_calls.append(terminal_id)
        if self.validate_entered is not None:
            self.validate_entered.set()
        if self.validate_release is not None:
            await self.validate_release.wait()
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

    async def select_user_question_option(
        self,
        terminal_id: str,
        *,
        option_count: int,
        option_index: int,
        submit_after: bool,
    ) -> str:
        return await self._record_question_action(
            terminal_id,
            "select",
            option_count,
            option_index,
            submit_after,
            "",
        )

    async def answer_user_question_with_text(
        self,
        terminal_id: str,
        *,
        option_count: int,
        text: str,
        submit_after: bool,
    ) -> str:
        return await self._record_question_action(
            terminal_id,
            "answer_text",
            option_count,
            -1,
            submit_after,
            text,
        )

    async def advance_user_question_after_multi_select(
        self,
        terminal_id: str,
        *,
        option_count: int,
        final_question: bool,
    ) -> str:
        return await self._record_question_action(
            terminal_id,
            "advance_multi",
            option_count,
            -1,
            final_question,
            "",
        )

    async def _record_question_action(
        self,
        terminal_id: str,
        action: str,
        option_count: int,
        option_index: int,
        final: bool,
        text: str,
    ) -> str:
        self.question_calls.append((terminal_id, action, option_count, option_index, final, text))
        self.active_injections += 1
        self.max_active_injections = max(self.max_active_injections, self.active_injections)
        try:
            if self.question_entered is not None:
                self.question_entered.set()
            if self.question_release is not None:
                await self.question_release.wait()
            return self.question_outcomes.popleft() if self.question_outcomes else TerminalOutcome.OK
        finally:
            self.active_injections -= 1

    async def inject_text(self, terminal_id: str, text: str) -> str:
        self.inject_calls.append((terminal_id, text))
        self.active_injections += 1
        self.max_active_injections = max(self.max_active_injections, self.active_injections)
        try:
            if self.inject_entered is not None:
                self.inject_entered.set()
            if self.inject_release is not None:
                await self.inject_release.wait()
            if self.inject_error is not None:
                return self.inject_error
            return self.inject_outcomes.popleft() if self.inject_outcomes else TerminalOutcome.OK
        finally:
            self.active_injections -= 1
