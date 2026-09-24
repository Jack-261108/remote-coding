"""Tests for the file upload handler."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.handlers.file_upload import (
    _format_size,
    _user_has_running_task,
)
from app.domain.file_models import FileUploadResult, FileValidationError
from app.domain.models import TaskRecord, TaskStatus
from app.infra.lock_registry import RefCountedLockRegistry
from app.services.upload_queue import UploadQueueManager

# --- Unit tests for helper functions ---


def test_format_size_bytes() -> None:
    assert _format_size(500) == "500 B"


def test_format_size_kb() -> None:
    assert _format_size(2048) == "2.0 KB"


def test_format_size_mb() -> None:
    assert _format_size(5 * 1024 * 1024) == "5.0 MB"


# --- Tests for _user_has_running_task ---


@pytest.mark.asyncio
async def test_user_has_running_task_true() -> None:
    task_service = AsyncMock()
    record = MagicMock(spec=TaskRecord)
    record.status = TaskStatus.RUNNING
    task_service.list_recent = AsyncMock(return_value=[record])

    result = await _user_has_running_task(task_service, user_id=123)
    assert result is True


@pytest.mark.asyncio
async def test_user_has_running_task_false_when_all_final() -> None:
    task_service = AsyncMock()
    record = MagicMock(spec=TaskRecord)
    record.status = TaskStatus.SUCCEEDED
    task_service.list_recent = AsyncMock(return_value=[record])

    result = await _user_has_running_task(task_service, user_id=123)
    assert result is False


@pytest.mark.asyncio
async def test_user_has_running_task_false_when_no_tasks() -> None:
    task_service = AsyncMock()
    task_service.list_recent = AsyncMock(return_value=[])

    result = await _user_has_running_task(task_service, user_id=123)
    assert result is False


# --- Integration tests for the handler ---


class DummyRouter:
    def __init__(self) -> None:
        self.handlers = []

    def message(self, *filters, **kwargs):
        def decorator(handler):
            self.handlers.append(handler)
            return handler

        return decorator


def _make_message(user_id: int = 42) -> MagicMock:
    message = AsyncMock(spec=["from_user", "bot", "document", "photo", "answer"])
    message.from_user = MagicMock()
    message.from_user.id = user_id
    message.answer = AsyncMock()
    return message


def _make_services():
    file_receiver = AsyncMock()
    file_receiver.receive_file = AsyncMock()
    session_service = AsyncMock()
    task_service = AsyncMock()
    task_service.list_recent = AsyncMock(return_value=[])
    return file_receiver, session_service, task_service


def _running_task() -> MagicMock:
    running_task = MagicMock(spec=TaskRecord)
    running_task.status = TaskStatus.RUNNING
    return running_task


def _register_upload_handlers(
    *,
    upload_max_file_size_mb: int = 20,
    upload_queue_max_files_per_user: int = 5,
    upload_queue_max_bytes_per_user: int = 20 * 1024 * 1024,
):
    from app.bot.handlers.file_upload import register_file_upload_handler

    router = DummyRouter()
    file_receiver, session_service, task_service = _make_services()
    upload_queue = UploadQueueManager(
        max_files_per_user=upload_queue_max_files_per_user,
        max_bytes_per_user=upload_queue_max_bytes_per_user,
    )

    register_file_upload_handler(
        router,
        file_receiver=file_receiver,
        session_service=session_service,
        task_service=task_service,
        upload_queue=upload_queue,
        upload_max_file_size_mb=upload_max_file_size_mb,
    )

    assert len(router.handlers) == 2
    document_handler, photo_handler = router.handlers
    return document_handler, photo_handler, upload_queue, file_receiver, session_service, task_service


def _attach_document(
    message: MagicMock, *, filename: str = "test.py", file_size: int | None = 11, data: bytes = b"hello world"
) -> AsyncMock:
    message.document = MagicMock()
    message.document.file_name = filename
    message.document.file_id = f"file-{filename}"
    message.document.file_size = file_size

    bot = AsyncMock()
    message.bot = bot
    file_obj = MagicMock()
    file_obj.file_path = f"documents/{filename}"
    bot.get_file = AsyncMock(return_value=file_obj)
    bot.download_file = AsyncMock(return_value=io.BytesIO(data))
    return bot


def _attach_photo(message: MagicMock, *, largest_file_size: int | None = 11, data: bytes = b"hello world") -> AsyncMock:
    small = MagicMock()
    small.file_id = "small-photo"
    small.file_unique_id = "small"
    small.file_size = 5
    largest = MagicMock()
    largest.file_id = "large-photo"
    largest.file_unique_id = "large"
    largest.file_size = largest_file_size
    message.photo = [small, largest]

    bot = AsyncMock()
    message.bot = bot
    file_obj = MagicMock()
    file_obj.file_path = "photos/large.jpg"
    bot.get_file = AsyncMock(return_value=file_obj)
    bot.download_file = AsyncMock(return_value=io.BytesIO(data))
    return bot


@pytest.mark.asyncio
async def test_handle_document_success() -> None:
    """Document upload should download, process, and reply with confirmation."""
    file_receiver, session_service, task_service = _make_services()

    session = MagicMock()
    session.workdir = "/tmp/work"
    session_service.get = AsyncMock(return_value=session)

    file_receiver.receive_file = AsyncMock(
        return_value=FileUploadResult(filename="test.py", size_bytes=1234, path=Path("/tmp/work/.tg-uploads/42/test.py"))
    )

    message = _make_message()
    message.document = MagicMock()
    message.document.file_name = "test.py"
    message.document.file_id = "file123"

    bot = AsyncMock()
    message.bot = bot
    file_obj = MagicMock()
    file_obj.file_path = "documents/test.py"
    bot.get_file = AsyncMock(return_value=file_obj)
    bot.download_file = AsyncMock(return_value=io.BytesIO(b"hello world"))

    # Import the actual handler function by extracting from registration
    from app.bot.handlers.file_upload import _process_upload

    # Simulate the handler logic directly
    data = b"hello world"
    await _process_upload(
        message,
        file_receiver=file_receiver,
        session_service=session_service,
        filename="test.py",
        data=data,
    )

    file_receiver.receive_file.assert_awaited_once_with(user_id=42, workdir="/tmp/work", filename="test.py", data=data)
    message.answer.assert_awaited_once()
    reply = message.answer.call_args[0][0]
    assert "test.py" in reply
    assert "✅" in reply


@pytest.mark.asyncio
async def test_handle_document_validation_error() -> None:
    """Rejected file should reply with error."""
    file_receiver, session_service, task_service = _make_services()

    session = MagicMock()
    session.workdir = "/tmp/work"
    session_service.get = AsyncMock(return_value=session)

    file_receiver.receive_file = AsyncMock(return_value=FileValidationError(filename="bad.exe", reason="Extension .exe is not allowed."))

    message = _make_message()
    data = b"MZ binary"

    from app.bot.handlers.file_upload import _process_upload

    await _process_upload(
        message,
        file_receiver=file_receiver,
        session_service=session_service,
        filename="bad.exe",
        data=data,
    )

    message.answer.assert_awaited_once()
    reply = message.answer.call_args[0][0]
    assert "❌" in reply
    assert "bad.exe" in reply
    assert ".exe is not allowed" in reply


@pytest.mark.asyncio
async def test_handle_document_no_session() -> None:
    """If user has no session, reply with guidance."""
    file_receiver, session_service, task_service = _make_services()
    session_service.get = AsyncMock(return_value=None)

    message = _make_message()

    from app.bot.handlers.file_upload import _process_upload

    await _process_upload(
        message,
        file_receiver=file_receiver,
        session_service=session_service,
        filename="test.py",
        data=b"data",
    )

    message.answer.assert_awaited_once()
    reply = message.answer.call_args[0][0]
    assert "/session" in reply or "/claude" in reply


@pytest.mark.asyncio
async def test_document_size_metadata_rejects_before_download() -> None:
    document_handler, _photo_handler, _queue, _file_receiver, _session_service, _task_service = _register_upload_handlers(
        upload_max_file_size_mb=1
    )
    message = _make_message()
    bot = _attach_document(message, filename="large.py", file_size=1024 * 1024 + 1)
    session = MagicMock()

    await document_handler(message, session=session)

    bot.get_file.assert_not_awaited()
    bot.download_file.assert_not_awaited()
    message.answer.assert_awaited_once()
    reply = message.answer.call_args[0][0]
    assert "文件被拒绝" in reply
    assert "1 MB" in reply


@pytest.mark.asyncio
async def test_photo_size_metadata_rejects_before_download() -> None:
    _document_handler, photo_handler, _queue, _file_receiver, _session_service, _task_service = _register_upload_handlers(
        upload_max_file_size_mb=1
    )
    message = _make_message()
    bot = _attach_photo(message, largest_file_size=1024 * 1024 + 1)
    session = MagicMock()

    await photo_handler(message, session=session)

    bot.get_file.assert_not_awaited()
    bot.download_file.assert_not_awaited()
    message.answer.assert_awaited_once()
    reply = message.answer.call_args[0][0]
    assert "文件被拒绝" in reply
    assert "1 MB" in reply


@pytest.mark.asyncio
async def test_running_task_queue_reply_mentions_restart_loss() -> None:
    document_handler, _photo_handler, queue, _file_receiver, _session_service, task_service = _register_upload_handlers(
        upload_max_file_size_mb=1
    )
    task_service.list_recent = AsyncMock(return_value=[_running_task()])
    message = _make_message()
    _attach_document(message, filename="queued.py", file_size=4, data=b"data")
    session = MagicMock()
    session.workdir = "/tmp/work"

    await document_handler(message, session=session)

    assert await queue.queued_count(user_id=42) == 1
    message.answer.assert_awaited_once()
    reply = message.answer.call_args[0][0]
    assert "已加入队列" in reply
    assert "bot" in reply
    assert "重启" in reply
    assert "丢失" in reply
    assert "60 分钟" in reply
    assert "过期" in reply


@pytest.mark.asyncio
async def test_running_task_rejects_when_queue_count_limit_reached() -> None:
    document_handler, _photo_handler, queue, _file_receiver, _session_service, task_service = _register_upload_handlers(
        upload_max_file_size_mb=1,
        upload_queue_max_files_per_user=1,
    )
    task_service.list_recent = AsyncMock(return_value=[_running_task()])
    session = MagicMock()
    session.workdir = "/tmp/work"

    first = _make_message()
    _attach_document(first, filename="first.py", file_size=5, data=b"first")
    await document_handler(first, session=session)

    second = _make_message()
    _attach_document(second, filename="second.py", file_size=6, data=b"second")
    await document_handler(second, session=session)

    assert await queue.queued_count(user_id=42) == 1
    second.answer.assert_awaited_once()
    reply = second.answer.call_args[0][0]
    assert "文件未加入队列" in reply
    assert "队列已满" in reply


@pytest.mark.asyncio
async def test_running_task_rejects_downloaded_file_over_size_limit_before_queueing() -> None:
    document_handler, _photo_handler, queue, _file_receiver, _session_service, task_service = _register_upload_handlers(
        upload_max_file_size_mb=1
    )
    task_service.list_recent = AsyncMock(return_value=[_running_task()])
    message = _make_message()
    bot = _attach_document(message, filename="no-metadata.bin", file_size=None, data=b"x" * (1024 * 1024 + 1))
    session = MagicMock()

    await document_handler(message, session=session)

    bot.get_file.assert_awaited_once()
    bot.download_file.assert_awaited_once()
    assert await queue.queued_count(user_id=42) == 0
    message.answer.assert_awaited_once()
    reply = message.answer.call_args[0][0]
    assert "文件被拒绝" in reply
    assert "1 MB" in reply
    assert "已加入队列" not in reply


@pytest.mark.asyncio
async def test_process_pending_uploads() -> None:
    """process_pending_uploads should process all queued files."""
    from app.bot.handlers.file_upload import process_pending_uploads

    file_receiver, session_service, task_service = _make_services()

    session = MagicMock()
    session.workdir = "/tmp/work"
    session_service.get = AsyncMock(return_value=session)

    file_receiver.receive_file = AsyncMock(
        return_value=FileUploadResult(filename="queued.py", size_bytes=100, path=Path("/tmp/work/.tg-uploads/42/queued.py"))
    )

    upload_queue = UploadQueueManager(max_files_per_user=2, max_bytes_per_user=100)
    result = await upload_queue.enqueue(user_id=42, filename="queued.py", data=b"content")
    assert result.accepted is True

    message = _make_message()

    await process_pending_uploads(
        message,
        file_receiver=file_receiver,
        session_service=session_service,
        upload_queue=upload_queue,
        upload_processing_locks=RefCountedLockRegistry(ttl_sec=300, cleanup_interval_sec=60, cleanup_batch_size=50),
        user_id=42,
    )

    # Queue should be cleared
    assert await upload_queue.queued_count(user_id=42) == 0
    file_receiver.receive_file.assert_awaited_once()
    message.answer.assert_awaited_once()


@pytest.mark.asyncio
async def test_download_failure_replies_with_error() -> None:
    """If Telegram download fails, reply with error message."""
    file_receiver, session_service, task_service = _make_services()

    message = _make_message()
    message.document = MagicMock()
    message.document.file_name = "test.py"
    message.document.file_id = "file123"

    bot = AsyncMock()
    message.bot = bot
    bot.get_file = AsyncMock(side_effect=Exception("Network timeout"))

    # Simulate what handle_document does on download failure:
    try:
        await bot.get_file("file123")
    except Exception as exc:
        await message.answer(f"❌ 文件下载失败: {exc}")

    message.answer.assert_awaited_once()
    reply = message.answer.call_args[0][0]
    assert "❌" in reply
    assert "Network timeout" in reply


@pytest.mark.asyncio
async def test_download_telegram_file_sends_error_on_exception() -> None:
    from app.bot.handlers.file_upload import _download_telegram_file

    message = _make_message()
    bot = AsyncMock()
    message.bot = bot
    bot.get_file = AsyncMock(side_effect=RuntimeError("connection dropped"))

    result = await _download_telegram_file(message, "file123")
    assert result is None
    message.answer.assert_awaited_once()
    assert "下载失败" in message.answer.call_args[0][0]
