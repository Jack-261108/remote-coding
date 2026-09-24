from __future__ import annotations

import asyncio

import pytest
from aiogram.enums import ParseMode

from app.bot.presenters.structured_reply_presenter import (
    FileToolAggregateStatusOutput,
    SubagentAggregateStatusOutput,
    SubagentToolStatusOutput,
    TaskListItemStatusOutput,
    TaskListStatusOutput,
    ToolStatusOutput,
)
from app.bot.presenters.tool_message_manager import ToolMessageManager
from app.domain.session_models import ToolStatus


class DummyTelegramMessage:
    def __init__(self, text: str, parse_mode=None) -> None:
        self.text = text
        self.parse_mode = parse_mode
        self.edits: list[str] = []
        self.edit_parse_modes: list[ParseMode | None] = []
        self.fail_next_edit = False
        self.not_modified_next_edit = False

    async def edit_text(self, text: str, parse_mode=None, reply_markup=None) -> DummyTelegramMessage:
        if self.not_modified_next_edit:
            self.not_modified_next_edit = False
            raise RuntimeError("message is not modified")
        if self.fail_next_edit:
            self.fail_next_edit = False
            raise RuntimeError("edit failed")
        self.text = text
        self.edits.append(text)
        self.edit_parse_modes.append(parse_mode)
        return self


class DummyRootMessage:
    def __init__(self) -> None:
        self.sent: list[DummyTelegramMessage] = []
        self.fail_next_answer = False

    async def answer(self, text: str, parse_mode=None) -> DummyTelegramMessage:
        if self.fail_next_answer:
            self.fail_next_answer = False
            raise RuntimeError("send failed")
        message = DummyTelegramMessage(text, parse_mode=parse_mode)
        self.sent.append(message)
        return message


def _output(status: ToolStatus | str, *, tool_use_id: str = "tool-1") -> ToolStatusOutput:
    status_value = status.value if isinstance(status, ToolStatus) else status
    return ToolStatusOutput(
        tool_use_id=tool_use_id,
        tool_name="Bash",
        tool_input={"command": "pytest -q"},
        status=status_value,
    )


def _task_output(
    *,
    subagent_statuses: tuple[tuple[str, str, dict, ToolStatus], ...],
    task_status: ToolStatus = ToolStatus.RUNNING,
) -> ToolStatusOutput:
    return ToolStatusOutput(
        tool_use_id="task-1",
        tool_name="Task",
        tool_input={"description": "修复测试失败"},
        status=task_status.value,
        subagent_tools=tuple(
            SubagentToolStatusOutput(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                tool_input=tool_input,
                status=status.value,
            )
            for tool_use_id, tool_name, tool_input, status in subagent_statuses
        ),
    )


def _aggregate_output(*, second_status: ToolStatus = ToolStatus.RUNNING) -> SubagentAggregateStatusOutput:
    return SubagentAggregateStatusOutput(
        message_key="subagent-aggregate",
        containers=(
            ToolStatusOutput(
                tool_use_id="agent-1",
                tool_name="Agent",
                tool_input={"description": "项目架构扫描"},
                status=ToolStatus.SUCCESS.value,
                subagent_tools=(
                    SubagentToolStatusOutput(
                        tool_use_id="read-1",
                        tool_name="Read",
                        tool_input={"file_path": "app/foo.py"},
                        status=ToolStatus.SUCCESS.value,
                    ),
                ),
            ),
            ToolStatusOutput(
                tool_use_id="agent-2",
                tool_name="Agent",
                tool_input={"description": "测试质量扫描"},
                status=second_status.value,
                subagent_tools=(
                    SubagentToolStatusOutput(
                        tool_use_id="grep-1",
                        tool_name="Grep",
                        tool_input={"pattern": "pytest"},
                        status=second_status.value,
                    ),
                ),
            ),
        ),
    )


def _task_list_status_output(*, second_status: str = "pending") -> TaskListStatusOutput:
    return TaskListStatusOutput(
        message_key="task-list",
        items=(
            TaskListItemStatusOutput(
                task_id="1",
                subject="梳理项目结构",
                status="completed",
                active_form="梳理项目结构",
            ),
            TaskListItemStatusOutput(
                task_id="2",
                subject="评估当前改动",
                status=second_status,
                active_form="评估当前改动",
            ),
        ),
    )


def _file_tool_aggregate_output(*, read_status: ToolStatus = ToolStatus.RUNNING) -> FileToolAggregateStatusOutput:
    return FileToolAggregateStatusOutput(
        message_key="file-tool-aggregate",
        tools=(
            ToolStatusOutput(
                tool_use_id="grep-1",
                tool_name="Grep",
                tool_input={"pattern": "SessionStore"},
                status=ToolStatus.SUCCESS.value,
            ),
            ToolStatusOutput(
                tool_use_id="read-1",
                tool_name="Read",
                tool_input={"file_path": "app/services/session_store.py"},
                status=read_status.value,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_tool_message_manager_sends_first_status_message() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_output(ToolStatus.RUNNING))

    assert len(root.sent) == 1
    assert "🔄 执行中" in root.sent[0].text
    assert "工具: Bash" in root.sent[0].text
    assert "命令: pytest -q" in root.sent[0].text
    assert root.sent[0].parse_mode == ParseMode.HTML


@pytest.mark.asyncio
async def test_tool_message_manager_edits_existing_message_to_success() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_output(ToolStatus.RUNNING))
    original_sent_text = root.sent[0].text
    await manager.handle(_output(ToolStatus.SUCCESS))

    assert len(root.sent) == 1
    assert "🔄 执行中" in original_sent_text
    assert "✅ 执行完成" in root.sent[0].text
    assert root.sent[0].edits
    assert "✅ 执行完成" in root.sent[0].edits[-1]


@pytest.mark.asyncio
async def test_tool_message_manager_keeps_error_status_message() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_output(ToolStatus.RUNNING))
    original_sent_text = root.sent[0].text
    await manager.handle(_output(ToolStatus.ERROR))

    assert len(root.sent) == 1
    assert "🔄 执行中" in original_sent_text
    assert "❌ 执行失败" in root.sent[0].text
    assert "❌ 执行失败" in root.sent[0].edits[-1]


@pytest.mark.asyncio
async def test_tool_message_manager_sends_success_when_no_existing_message() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_output(ToolStatus.SUCCESS))

    assert len(root.sent) == 1
    assert "执行完成" in root.sent[0].text


@pytest.mark.asyncio
async def test_tool_message_manager_re_sends_when_edit_fails() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_output(ToolStatus.RUNNING))
    root.sent[0].fail_next_edit = True
    await manager.handle(_output(ToolStatus.INTERRUPTED))

    assert len(root.sent) == 2
    assert "已中断" in root.sent[1].text


@pytest.mark.asyncio
async def test_tool_message_manager_sends_subagent_aggregate_message() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_aggregate_output())

    assert len(root.sent) == 1
    assert "🔄 1/2 agents running" in root.sent[0].text
    assert "项目架构扫描 · 1 tool uses · Done" in root.sent[0].text
    assert "测试质量扫描 · 1 tool uses · Running" in root.sent[0].text


@pytest.mark.asyncio
async def test_tool_message_manager_edits_subagent_aggregate_message() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_aggregate_output(second_status=ToolStatus.RUNNING))
    await manager.handle(_aggregate_output(second_status=ToolStatus.SUCCESS))

    assert len(root.sent) == 1
    assert "✅ 2 agents finished" in root.sent[0].text
    assert root.sent[0].edits
    assert "测试质量扫描 · 1 tool uses · Done" in root.sent[0].edits[-1]


@pytest.mark.asyncio
async def test_tool_message_manager_keeps_subagent_aggregate_when_edit_fails() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_aggregate_output(second_status=ToolStatus.RUNNING))
    root.sent[0].fail_next_edit = True
    await manager.handle(_aggregate_output(second_status=ToolStatus.SUCCESS))

    assert len(root.sent) == 1
    assert "🔄 1/2 agents running" in root.sent[0].text

    await manager.handle(_aggregate_output(second_status=ToolStatus.SUCCESS))

    assert len(root.sent) == 1
    assert "✅ 2 agents finished" in root.sent[0].text


@pytest.mark.asyncio
async def test_tool_message_manager_tracks_aggregate_and_flat_tool_separately() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_aggregate_output(second_status=ToolStatus.RUNNING))
    await manager.handle(_output(ToolStatus.RUNNING, tool_use_id="tool-1"))
    await manager.handle(_aggregate_output(second_status=ToolStatus.SUCCESS))

    assert len(root.sent) == 2
    assert "✅ 2 agents finished" in root.sent[0].text
    assert "工具: Bash" in root.sent[1].text


@pytest.mark.asyncio
async def test_tool_message_manager_edits_file_tool_aggregate_message() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_file_tool_aggregate_output(read_status=ToolStatus.RUNNING))
    await manager.handle(_file_tool_aggregate_output(read_status=ToolStatus.SUCCESS))

    assert len(root.sent) == 1
    assert "✅ 文件检索 · 完成" in root.sent[0].text
    assert root.sent[0].edits
    assert "✅ 2. Read - 完成" in root.sent[0].edits[-1]


@pytest.mark.asyncio
async def test_tool_message_manager_keeps_file_tool_aggregate_when_edit_fails() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_file_tool_aggregate_output(read_status=ToolStatus.RUNNING))
    root.sent[0].fail_next_edit = True
    await manager.handle(_file_tool_aggregate_output(read_status=ToolStatus.SUCCESS))

    assert len(root.sent) == 1
    assert "🔄 文件检索 · 执行中" in root.sent[0].text

    await manager.handle(_file_tool_aggregate_output(read_status=ToolStatus.SUCCESS))

    assert len(root.sent) == 1
    assert "✅ 文件检索 · 完成" in root.sent[0].text


@pytest.mark.asyncio
async def test_tool_message_manager_sends_claude_task_list_message() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_task_list_status_output(second_status="in_progress"))

    assert len(root.sent) == 1
    assert "任务列表" in root.sent[0].text
    assert "当前: 🔄 2. 评估当前改动" in root.sent[0].text
    assert "1. 梳理项目结构 - 完成" in root.sent[0].text
    assert "=&gt; 🔄 2. 评估当前改动 - 执行中" in root.sent[0].text


@pytest.mark.asyncio
async def test_tool_message_manager_serializes_concurrent_claude_task_list_updates() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await asyncio.gather(
        manager.handle(_task_list_status_output(second_status="in_progress")),
        manager.handle(_task_list_status_output(second_status="completed")),
    )

    assert len(root.sent) == 1


@pytest.mark.asyncio
async def test_tool_message_manager_ignores_not_modified_edit_without_re_sending() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_task_list_status_output(second_status="in_progress"))
    root.sent[0].not_modified_next_edit = True
    await manager.handle(_task_list_status_output(second_status="in_progress"))

    assert len(root.sent) == 1


@pytest.mark.asyncio
async def test_tool_message_manager_keeps_claude_task_list_when_edit_fails() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_task_list_status_output(second_status="in_progress"))
    root.sent[0].fail_next_edit = True
    await manager.handle(_task_list_status_output(second_status="completed"))

    assert len(root.sent) == 1
    assert "当前: 🔄 2. 评估当前改动" in root.sent[0].text

    await manager.handle(_task_list_status_output(second_status="completed"))

    assert len(root.sent) == 1
    assert "当前: 无（全部完成）" in root.sent[0].text


@pytest.mark.asyncio
async def test_tool_message_manager_edits_claude_task_list_message() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_task_list_status_output(second_status="in_progress"))
    await manager.handle(_task_list_status_output(second_status="completed"))

    assert len(root.sent) == 1
    assert "当前: 无（全部完成）" in root.sent[0].text
    assert root.sent[0].edits
    assert "2. 评估当前改动 - 完成" in root.sent[0].edits[-1]


@pytest.mark.asyncio
async def test_tool_message_manager_tracks_claude_task_list_and_flat_tool_separately() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_task_list_status_output(second_status="in_progress"))
    await manager.handle(_output(ToolStatus.RUNNING, tool_use_id="tool-1"))
    await manager.handle(_task_list_status_output(second_status="completed"))

    assert len(root.sent) == 2
    assert "当前: 无（全部完成）" in root.sent[0].text
    assert "工具: Bash" in root.sent[1].text


@pytest.mark.asyncio
async def test_tool_message_manager_sends_task_list_message() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_task_output(subagent_statuses=(("read-1", "Read", {"file_path": "app/foo.py"}, ToolStatus.RUNNING),)))

    assert len(root.sent) == 1
    assert "任务列表" in root.sent[0].text
    assert "任务: 修复测试失败" in root.sent[0].text
    assert "当前: 🔄 1. Read" in root.sent[0].text
    assert "🔄 1. Read - 执行中 - 文件: app/foo.py" in root.sent[0].text


@pytest.mark.asyncio
async def test_tool_message_manager_edits_task_list_when_current_subagent_changes() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_task_output(subagent_statuses=(("read-1", "Read", {"file_path": "app/foo.py"}, ToolStatus.RUNNING),)))
    await manager.handle(
        _task_output(
            subagent_statuses=(
                ("read-1", "Read", {"file_path": "app/foo.py"}, ToolStatus.SUCCESS),
                ("bash-1", "Bash", {"command": "pytest -q"}, ToolStatus.RUNNING),
            )
        )
    )

    assert len(root.sent) == 1
    assert "当前: 🔄 2. Bash" in root.sent[0].text
    assert root.sent[0].edits
    assert "🔄 2. Bash - 执行中 - 命令: pytest -q" in root.sent[0].edits[-1]


@pytest.mark.asyncio
async def test_tool_message_manager_re_sends_task_list_when_edit_fails() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_task_output(subagent_statuses=(("read-1", "Read", {"file_path": "app/foo.py"}, ToolStatus.RUNNING),)))
    root.sent[0].fail_next_edit = True
    await manager.handle(
        _task_output(
            subagent_statuses=(
                ("read-1", "Read", {"file_path": "app/foo.py"}, ToolStatus.SUCCESS),
                ("bash-1", "Bash", {"command": "pytest -q"}, ToolStatus.RUNNING),
            )
        )
    )

    assert len(root.sent) == 2
    assert "任务列表" in root.sent[1].text
    assert "当前: 🔄 2. Bash" in root.sent[1].text


@pytest.mark.asyncio
async def test_tool_message_manager_does_not_raise_when_send_fails() -> None:
    root = DummyRootMessage()
    root.fail_next_answer = True
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    await manager.handle(_output(ToolStatus.RUNNING))

    assert root.sent == []


@pytest.mark.asyncio
async def test_tool_message_manager_falls_back_to_plain_text_on_oversized_tag() -> None:
    root = DummyRootMessage()
    manager = ToolMessageManager(root_message=root, task_id="task-1", user_id=1, provider="claude_code")

    huge_url = "https://example.com/" + ("a" * 5000)
    huge_text = f"查看文档: [link]({huge_url})"

    text, parse_mode = manager._render(huge_text)
    assert parse_mode is None
    assert len(text) <= 4096
    assert "https://example.com/" in text
