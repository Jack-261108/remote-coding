from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from aiogram.enums import ParseMode
from aiogram.types import Message

from app.bot.presenters.structured_reply_presenter import (
    FileToolAggregateStatusOutput,
    SubagentAggregateStatusOutput,
    TaskListStatusOutput,
    ToolStatusOutput,
    build_file_tool_aggregate_status_message,
    build_subagent_aggregate_status_message,
    build_task_list_status_message,
    build_tool_status_message,
    build_tool_task_list_message,
)
from app.bot.presenters.telegram_formatting import (
    TELEGRAM_TEXT_LIMIT,
    render_markdownish_with_html_fallback,
)

logger = logging.getLogger(__name__)


@dataclass
class _TrackedToolMessage:
    message: Message


class ToolMessageManager:
    def __init__(self, *, root_message: Message, task_id: str, user_id: int, provider: str) -> None:
        self._root_message = root_message
        self._task_id = task_id
        self._user_id = user_id
        self._provider = provider
        self._messages: dict[str, _TrackedToolMessage] = {}
        self._lock = asyncio.Lock()

    async def handle(
        self,
        output: ToolStatusOutput | SubagentAggregateStatusOutput | TaskListStatusOutput | FileToolAggregateStatusOutput,
    ) -> None:
        resend_on_edit_failure = True
        if isinstance(output, SubagentAggregateStatusOutput):
            message_key = output.message_key
            text = build_subagent_aggregate_status_message(output)
            resend_on_edit_failure = False
        elif isinstance(output, FileToolAggregateStatusOutput):
            message_key = output.message_key
            text = build_file_tool_aggregate_status_message(output)
            resend_on_edit_failure = False
        elif isinstance(output, TaskListStatusOutput):
            message_key = output.message_key
            text = build_task_list_status_message(output)
            resend_on_edit_failure = False
        elif output.subagent_tools:
            message_key = output.tool_use_id
            text = build_tool_task_list_message(output)
        else:
            message_key = output.tool_use_id
            text = build_tool_status_message(
                tool_name=output.tool_name,
                tool_input=output.tool_input,
                status=output.status,
            )

        async with self._lock:
            existing = self._messages.get(message_key)
            if existing is None:
                await self._send_and_track(message_key, text)
                return

            edited = await self._edit(existing.message, text, tool_use_id=message_key)
            if edited or not resend_on_edit_failure:
                return
            await self._send_and_track(message_key, text)

    async def edit_with_keyboard(self, *, tool_use_id: str, text: str, reply_markup) -> tuple[bool, Message | None]:
        """Edit an existing tracked tool message and attach a reply_markup.

        Returns (True, message) if the edit succeeded; (False, None) if the tool was not tracked
        or the edit failed (caller should send a new message instead).
        """
        async with self._lock:
            existing = self._messages.get(tool_use_id)
            if existing is None:
                return False, None
            try:
                rendered, parse_mode = self._render(text)
                await existing.message.edit_text(rendered, parse_mode=parse_mode, reply_markup=reply_markup)
                return True, existing.message
            except Exception as exc:
                if _is_message_not_modified(exc):
                    # Text unchanged, but we still need to attach the keyboard
                    try:
                        await existing.message.edit_reply_markup(reply_markup=reply_markup)
                        return True, existing.message
                    except Exception:
                        logger.exception(
                            "telegram tool message reply_markup edit failed",
                            extra={
                                "task_id": self._task_id,
                                "user_id": self._user_id,
                                "provider": self._provider,
                                "tool_use_id": tool_use_id,
                            },
                        )
                        return False, None
                logger.exception(
                    "telegram tool message edit_with_keyboard failed",
                    extra={
                        "task_id": self._task_id,
                        "user_id": self._user_id,
                        "provider": self._provider,
                        "tool_use_id": tool_use_id,
                    },
                )
                return False, None

    async def _send_and_track(self, tool_use_id: str, text: str) -> None:
        sent = await self._send(text, tool_use_id=tool_use_id)
        if sent is not None:
            self._messages[tool_use_id] = _TrackedToolMessage(message=sent)

    async def _send(self, text: str, *, tool_use_id: str) -> Message | None:
        try:
            rendered, parse_mode = self._render(text)
            return await self._root_message.answer(rendered, parse_mode=parse_mode)
        except Exception:
            logger.exception(
                "telegram tool message send failed",
                extra={
                    "task_id": self._task_id,
                    "user_id": self._user_id,
                    "provider": self._provider,
                    "tool_use_id": tool_use_id,
                },
            )
            return None

    async def _edit(self, message: Message, text: str, *, tool_use_id: str) -> bool:
        try:
            rendered, parse_mode = self._render(text)
            await message.edit_text(rendered, parse_mode=parse_mode, reply_markup=None)
            return True
        except Exception as exc:
            if _is_message_not_modified(exc):
                return True
            logger.exception(
                "telegram tool message edit failed",
                extra={
                    "task_id": self._task_id,
                    "user_id": self._user_id,
                    "provider": self._provider,
                    "tool_use_id": tool_use_id,
                },
            )
            return False

    def _render(self, text: str) -> tuple[str, ParseMode | None]:
        chunks, parse_mode_name = render_markdownish_with_html_fallback(text, TELEGRAM_TEXT_LIMIT)
        parse_mode = ParseMode.HTML if parse_mode_name == "HTML" else None
        return (chunks[0] if chunks else "", parse_mode)


def _is_message_not_modified(exc: Exception) -> bool:
    return "message is not modified" in str(exc).lower()
