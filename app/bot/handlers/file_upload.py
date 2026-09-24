from __future__ import annotations

import asyncio
import logging

from aiogram import F, Router
from aiogram.types import Message

from app.bot.handlers.user_utils import extract_user_id
from app.domain.file_models import FileUploadResult, FileValidationError
from app.domain.models import SessionContext, TaskStatus
from app.infra.lock_registry import RefCountedLockRegistry
from app.services.background_task_registry import BackgroundTaskRegistry
from app.services.file_receiver import FileReceiverService
from app.services.session_service import SessionService
from app.services.task_service import TaskService
from app.services.upload_queue import UploadQueueManager

logger = logging.getLogger(__name__)


def _format_size(size_bytes: int) -> str:
    """Format byte size into human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _max_upload_size_bytes(upload_max_file_size_mb: int) -> int:
    return upload_max_file_size_mb * 1024 * 1024


def _format_ttl(ttl_sec: int) -> str:
    if ttl_sec % 60 == 0:
        return f"{ttl_sec // 60} 分钟"
    return f"{ttl_sec} 秒"


def _metadata_exceeds_limit(file_size: int | None, *, max_size_bytes: int) -> bool:
    return file_size is not None and file_size > max_size_bytes


async def _answer_oversized(message: Message, *, filename: str, size_bytes: int, upload_max_file_size_mb: int) -> None:
    await message.answer(f"❌ 文件被拒绝: {filename}\n原因: 文件大小 {_format_size(size_bytes)} 超过 {upload_max_file_size_mb} MB 限制。")


async def _user_has_running_task(task_service: TaskService, user_id: int, *, exclude_task_id: str | None = None) -> bool:
    """Check if the user has a task currently in RUNNING or PENDING state."""
    tasks = await task_service.list_active(user_id)
    if not isinstance(tasks, list) or not tasks:
        tasks = await task_service.list_recent(user_id, limit=5)
    if not isinstance(tasks, list):
        tasks = []
    return any(
        t.status in (TaskStatus.RUNNING, TaskStatus.PENDING) and (exclude_task_id is None or getattr(t, "task_id", None) != exclude_task_id)
        for t in tasks
    )


async def _download_telegram_file(message: Message, file_id: str) -> bytes | None:
    """从 Telegram 下载文件，失败时发送错误消息并返回 None。"""
    try:
        bot = message.bot
        if bot is None:
            await message.answer("❌ 内部错误: 无法获取 bot 实例。")
            return None
        file_obj = await bot.get_file(file_id)
        if file_obj.file_path is None:
            await message.answer("❌ 下载失败: Telegram 未返回文件路径。")
            return None
        bio = await bot.download_file(file_obj.file_path)
        if bio is None:
            await message.answer("❌ 下载失败: 无法从 Telegram 下载文件。")
            return None
        return bio.read()
    except Exception as exc:
        logger.exception("Failed to download file %s: %s", file_id, exc)
        try:
            await message.answer("❌ 下载失败: 从 Telegram 下载文件时发生异常。")
        except Exception:
            logger.warning("failed to report telegram download error", exc_info=True)
        return None


async def _enqueue_or_process_upload(
    message: Message,
    *,
    file_receiver: FileReceiverService,
    session_service: SessionService,
    task_service: TaskService,
    upload_queue: UploadQueueManager,
    upload_queue_ttl_sec: int,
    user_id: int,
    filename: str,
    data: bytes,
    session: SessionContext | None = None,
) -> None:
    """If the user has a running task, queue the upload; otherwise process it."""
    if await _user_has_running_task(task_service, user_id):
        if session is None:
            session = await session_service.get(user_id)
        if session is None:
            await message.answer("请先使用 /session 或 /claude 创建会话后再上传文件。")
            return
        queued = await upload_queue.enqueue(user_id=user_id, filename=filename, data=data, workdir=session.workdir)
        if not queued.accepted:
            await message.answer(f"❌ 文件未加入队列: {filename}\n原因: {queued.reason}")
            return
        await message.answer(
            f"⏳ 任务运行中，文件 {filename} 已加入队列，将在任务完成后处理。\n"
            f"注意：队列仅保存在内存中，如果 bot 在任务完成前重启，已排队文件会丢失；"
            f"排队文件超过 {_format_ttl(upload_queue_ttl_sec)} 未处理会过期。"
        )
        return

    await _process_upload(
        message,
        file_receiver=file_receiver,
        session_service=session_service,
        filename=filename,
        data=data,
        session=session,
    )


async def _process_upload(
    message: Message,
    *,
    file_receiver: FileReceiverService,
    session_service: SessionService,
    filename: str,
    data: bytes,
    workdir: str | None = None,
    session: SessionContext | None = None,
) -> None:
    """Validate and store a file, then reply with result."""
    user_id = extract_user_id(message)
    if session is None:
        session = await session_service.get(user_id)
    if session is None:
        await message.answer("请先使用 /session 或 /claude 创建会话后再上传文件。")
        return

    selected_workdir = workdir or session.workdir

    result = await file_receiver.receive_file(
        user_id=user_id,
        workdir=selected_workdir,
        filename=filename,
        data=data,
    )

    if isinstance(result, FileUploadResult):
        size_str = _format_size(result.size_bytes)
        await message.answer(f"✅ 文件已接收: {result.filename} ({size_str})")
    elif isinstance(result, FileValidationError):
        await message.answer(f"❌ 文件被拒绝: {result.filename}\n原因: {result.reason}")


async def process_pending_uploads(
    message: Message,
    *,
    file_receiver: FileReceiverService,
    session_service: SessionService,
    upload_queue: UploadQueueManager,
    upload_processing_locks: RefCountedLockRegistry,
    user_id: int,
    task_service: TaskService | None = None,
    completed_task_id: str | None = None,
) -> None:
    """Process any queued uploads for a user after their task completes."""
    async with upload_processing_locks.lock(str(user_id)):
        if task_service is not None and await _user_has_running_task(task_service, user_id, exclude_task_id=completed_task_id):
            logger.info(
                "queued upload processing deferred because another task is active",
                extra={"user_id": user_id, "completed_task_id": completed_task_id},
            )
            return

        pending = await upload_queue.drain(user_id=user_id)
        for item in pending:
            try:
                await _process_upload(
                    message,
                    file_receiver=file_receiver,
                    session_service=session_service,
                    filename=item.filename,
                    data=item.data,
                    workdir=item.workdir,
                )
            except Exception:
                logger.exception("queued upload processing failed", extra={"user_id": user_id, "filename": item.filename})


async def flush_pending_uploads_for_task_start(
    message: Message,
    *,
    file_receiver: FileReceiverService,
    session_service: SessionService,
    upload_queue: UploadQueueManager,
    upload_processing_locks: RefCountedLockRegistry,
    user_id: int,
    task_service: TaskService | None = None,
) -> None:
    """Flush queued uploads before starting a new task for this user."""
    await process_pending_uploads(
        message,
        file_receiver=file_receiver,
        session_service=session_service,
        upload_queue=upload_queue,
        upload_processing_locks=upload_processing_locks,
        user_id=user_id,
        task_service=task_service,
    )


def schedule_pending_upload_processing(
    message: Message,
    *,
    file_receiver: FileReceiverService,
    session_service: SessionService,
    upload_queue: UploadQueueManager,
    upload_background_tasks: BackgroundTaskRegistry,
    upload_processing_locks: RefCountedLockRegistry,
    user_id: int,
    task_service: TaskService | None = None,
    completed_task_id: str | None = None,
) -> asyncio.Task[None]:
    """Schedule queued uploads to be processed in the background."""
    return upload_background_tasks.spawn(
        process_pending_uploads(
            message,
            file_receiver=file_receiver,
            session_service=session_service,
            upload_queue=upload_queue,
            upload_processing_locks=upload_processing_locks,
            user_id=user_id,
            task_service=task_service,
            completed_task_id=completed_task_id,
        )
    )


def register_file_upload_handler(
    router: Router,
    *,
    file_receiver: FileReceiverService,
    session_service: SessionService,
    task_service: TaskService,
    upload_queue: UploadQueueManager,
    upload_max_file_size_mb: int,
    upload_queue_ttl_sec: int = 3600,
) -> None:
    @router.message(F.document)
    async def handle_document(message: Message, session: SessionContext) -> None:
        user_id = extract_user_id(message)
        document = message.document
        if document is None:
            return

        filename = document.file_name or "unnamed_file"
        max_size_bytes = _max_upload_size_bytes(upload_max_file_size_mb)
        file_size = document.file_size
        if _metadata_exceeds_limit(file_size, max_size_bytes=max_size_bytes):
            await _answer_oversized(
                message,
                filename=filename,
                size_bytes=file_size,  # type: ignore[arg-type]
                upload_max_file_size_mb=upload_max_file_size_mb,
            )
            return

        file_data = await _download_telegram_file(message, document.file_id)
        if file_data is None:
            return

        if len(file_data) > max_size_bytes:
            await _answer_oversized(
                message,
                filename=filename,
                size_bytes=len(file_data),
                upload_max_file_size_mb=upload_max_file_size_mb,
            )
            return

        await _enqueue_or_process_upload(
            message,
            file_receiver=file_receiver,
            session_service=session_service,
            task_service=task_service,
            upload_queue=upload_queue,
            upload_queue_ttl_sec=upload_queue_ttl_sec,
            user_id=user_id,
            filename=filename,
            data=file_data,
            session=session,
        )

    @router.message(F.photo)
    async def handle_photo(message: Message, session: SessionContext) -> None:
        user_id = extract_user_id(message)
        if not message.photo:
            return

        # Use the largest resolution photo (last in array)
        photo = message.photo[-1]
        filename = f"photo_{photo.file_unique_id}.jpg"
        max_size_bytes = _max_upload_size_bytes(upload_max_file_size_mb)
        file_size = photo.file_size
        if _metadata_exceeds_limit(file_size, max_size_bytes=max_size_bytes):
            await _answer_oversized(
                message,
                filename=filename,
                size_bytes=file_size,  # type: ignore[arg-type]
                upload_max_file_size_mb=upload_max_file_size_mb,
            )
            return

        file_data = await _download_telegram_file(message, photo.file_id)
        if file_data is None:
            return

        if len(file_data) > max_size_bytes:
            await _answer_oversized(
                message,
                filename=filename,
                size_bytes=len(file_data),
                upload_max_file_size_mb=upload_max_file_size_mb,
            )
            return

        await _enqueue_or_process_upload(
            message,
            file_receiver=file_receiver,
            session_service=session_service,
            task_service=task_service,
            upload_queue=upload_queue,
            upload_queue_ttl_sec=upload_queue_ttl_sec,
            user_id=user_id,
            filename=filename,
            data=file_data,
            session=session,
        )
