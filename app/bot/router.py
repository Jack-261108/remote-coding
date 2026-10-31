from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from aiogram import F, Router
from aiogram.filters import BaseFilter, Command
from aiogram.types import Message

from app.adapters.claude.paths import ClaudePaths
from app.bot.handlers.admin_challenge import maybe_handle_admin_password_text
from app.bot.handlers.command_attach import register_attach_handler
from app.bot.handlers.command_cancel import register_cancel_handler
from app.bot.handlers.command_claude import register_claude_handler
from app.bot.handlers.command_cmds import register_cmds_handler
from app.bot.handlers.command_exit import register_exit_handler
from app.bot.handlers.command_export import register_export_handler
from app.bot.handlers.command_list import register_list_handler
from app.bot.handlers.command_permission import register_permission_handlers
from app.bot.handlers.command_resume import register_resume_handler
from app.bot.handlers.command_run import register_run_handler, run_prompt_and_stream
from app.bot.handlers.command_session import register_session_handler
from app.bot.handlers.command_status import register_status_handler
from app.bot.handlers.command_user_question import (
    _acknowledge_and_send_next_prompt,
    maybe_handle_pending_user_question_text,
    register_user_question_handlers,
)
from app.bot.handlers.external_permission import register_external_permission_handler
from app.bot.handlers.external_session import register_external_session_handler
from app.bot.handlers.file_upload import (
    flush_pending_uploads_for_task_start,
    register_file_upload_handler,
    schedule_pending_upload_processing,
)
from app.bot.handlers.session_actions import (
    register_external_text_handlers,
    register_pair_consume_handler,
    register_session_action_handlers,
)
from app.bot.middleware.callback_validator import CallbackValidatorMiddleware
from app.bot.middleware.error_handling import ErrorHandlingMiddleware
from app.bot.middleware.session_guard import SessionGuardMiddleware
from app.bot.presenters.chunk_sender import ChunkSender
from app.config.settings import Settings
from app.domain.models import SessionContext
from app.infra.lock_registry import RefCountedLockRegistry
from app.services.background_task_registry import BackgroundTaskRegistry
from app.services.diff_generator import DiffGeneratorService
from app.services.external_session_binder import ExternalSessionBinder
from app.services.external_session_discovery import ExternalSessionDiscoveryService
from app.services.external_user_question_state import (
    ExternalQuestionPendingAmbiguous,
    ExternalQuestionPendingUnique,
)
from app.services.file_receiver import FileReceiverService
from app.services.result_exporter import ResultExporterService
from app.services.session_registry import SessionRegistryService
from app.services.session_scanner import SessionScanner
from app.services.session_service import SessionService
from app.services.session_store import SessionStore
from app.services.status_display import StatusDisplayService
from app.services.task_service import TaskService
from app.services.upload_queue import UploadQueueManager
from app.services.user_question_callback_registry import UserQuestionCallbackRegistry

if TYPE_CHECKING:
    from app.adapters.claude.hook_socket_server import HookSocketServer
    from app.services.admin_password_service import AdminPasswordService
    from app.services.external_binding_reaper import ExternalBindingReaper
    from app.services.external_session_input_service import ExternalSessionInputService
    from app.services.external_user_question_state import ExternalUserQuestionState
    from app.services.permission_gateway import PermissionGateway
    from app.services.unbound_permission_handler import UnboundPermissionHandler

logger = logging.getLogger(__name__)


class PendingAdminPasswordFilter(BaseFilter):
    def __init__(self, admin_password_service: AdminPasswordService | None) -> None:
        self._admin_password_service = admin_password_service

    async def __call__(self, message: Message) -> bool:
        user_id = message.from_user.id if message.from_user else 0
        return bool(
            user_id
            and self._admin_password_service is not None
            and self._admin_password_service.is_enabled
            and self._admin_password_service.has_pending(user_id)
        )


class ExternalInputTargetActiveFilter(BaseFilter):
    """Match only when the user has an active external Ghostty input target
    AND no active managed chat session.

    The two states can coexist (activating an external target does not clear
    ``claude_chat_active``, and ``/claude`` does not clear the target). When
    they do, the managed chat wins for plain text only: this filter returns
    False so plain text falls through to the managed ``chat_text_router``
    instead of being silently injected into the external terminal. Slash
    commands are the exception — ``chat_text_router`` excludes them, so an
    unregistered command that also yielded here would fall through to
    nothing; slashes still route to the external terminal (registered
    commands were already consumed by the earlier command routers).
    Without a target it also returns False. Partitioning by user state
    without raising ``SkipHandler`` (which ErrorHandlingMiddleware would
    intercept).

    Order matters for the hot path: ``has_target`` is an in-memory lookup and
    gates the ``session_service.get`` read (an in-memory cache hit in the
    context store after its cold start), so users without an external target —
    the vast majority of plain-text traffic — never pay for it.
    """

    def __init__(self, input_service: ExternalSessionInputService, session_service: SessionService) -> None:
        self._input_service = input_service
        self._session_service = session_service

    async def __call__(self, message: Message) -> bool:
        user_id = message.from_user.id if message.from_user else 0
        if not user_id:
            return False
        # Cheap in-memory check first: no external target → the session read
        # is never reached.
        if not await self._input_service.has_target(user_id):
            return False
        if (message.text or "").startswith("/"):
            # Slash text never reaches ``chat_text_router`` (it excludes
            # leading slashes), so yielding on coexistence would silently drop
            # an unregistered command. Route it to the terminal — registered
            # commands were consumed earlier — and skip the session read.
            return True
        session = await self._session_service.get(user_id)
        if session is not None and session.claude_chat_active:
            return False
        return True


class ExternalQuestionActiveFilter(BaseFilter):
    """Match plain text only when there is at least one active external Ghostty
    question AND no managed pending question.

    Routing here (before the ordinary external/text routers) lets the user
    answer a Ghostty ``AskUserQuestion`` by typing free text as ``Other``.
    An ambiguous state (several active questions) also passes: the handler
    re-resolves and tells the user to answer via buttons. Returning False on
    ambiguity would let the text fall through to the external-terminal router
    and be injected into a terminal that is waiting on a question. Managed
    pending questions are checked first so this never hijacks a managed tmux
    session. ``True`` consumes the message; ``False`` lets it fall through.
    """

    def __init__(
        self,
        external_uq_state: ExternalUserQuestionState | None,
        task_service: TaskService,
    ) -> None:
        self._external_uq_state = external_uq_state
        self._task_service = task_service

    async def __call__(self, message: Message) -> bool:
        if self._external_uq_state is None:
            return False
        user_id = message.from_user.id if message.from_user else 0
        if not user_id:
            return False
        if message.text is None or message.text.startswith("/"):
            return False
        managed_pending = await self._task_service.get_pending_user_questions(user_id)
        if managed_pending:
            return False
        resolution = self._external_uq_state.resolve_unique_active_for_user(user_id, group="external_terminal")
        return isinstance(resolution, (ExternalQuestionPendingUnique, ExternalQuestionPendingAmbiguous))


async def answer_external_user_question_text(
    message: Message,
    *,
    external_uq_state: ExternalUserQuestionState,
    task_service: TaskService,
) -> None:
    """Consume a free-text message as an external Ghostty ``AskUserQuestion`` Other answer.

    Mirrors the routing gate in ``ExternalQuestionActiveFilter``. The filter only
    gates routing; a managed AskUserQuestion may appear in the filter→handler
    window, so managed pending is re-checked here and wins (fail closed) before
    the text is consumed as a Ghostty Other answer.

    Residual race window: between the filter's ``get_pending_user_questions``
    and this handler's re-check, a managed pending could appear AND be consumed
    by a concurrent handler, leaving this re-check seeing an empty managed set.
    In that case the message would be consumed as a Ghostty Other answer despite a
    managed question having existed. This is not a flaw the filter can close: the
    two ``get_pending_user_questions`` reads are non-locking snapshots. The
    invariant that holds it together is the per-user lock in
    ``UserQuestionService.answer_pending_user_question_text``: it serialises the
    resolve+submit of *this* user's free-text answers, so two answers for one user
    cannot both pass the re-check simultaneously. The remaining edge needs a same-user
    managed question to be born, answered, and cleared in the exact window between the
    two reads — narrow enough to be acceptable, and any double-consumption still
    reaches exactly-one owner (per-user lock downstream). Document this so future
    "tighten the re-check with a lock" ideas know the lock already exists one layer
    down, and is the right place to harden rather than the filter.
    """
    user_id = message.from_user.id if message.from_user else 0
    text = (message.text or "").strip()
    if not user_id or not text:
        await message.answer("请发送非空文字作为回答")
        return
    if await task_service.get_pending_user_questions(user_id):
        await message.answer("已有待处理的选择题，请先用其按钮回答")
        return
    resolution = external_uq_state.resolve_unique_active_for_user(user_id, group="external_terminal")
    if isinstance(resolution, ExternalQuestionPendingAmbiguous):
        await message.answer("存在多个待处理的外部问题，请用按钮精确回答")
        return
    if not isinstance(resolution, ExternalQuestionPendingUnique):
        await message.answer("该问题刚过期，请稍后重试或使用按钮")
        return
    ok, response_text, next_prompt = await task_service.answer_pending_user_question_text(user_id=user_id, text=text)
    # ``response_text`` is already a complete user-facing sentence from the service
    # (success: "已记录选择: ..." / failure: "问题已过期或目标已变化" etc.). Forward it
    # verbatim in both branches so the service is the single source of copy — do not
    # prefix failures here, which would double-wrap the service's own sentence.
    await message.answer(response_text)
    if ok and next_prompt is not None:
        # Render the next prompt with a fresh inline keyboard (tokens + buttons),
        # mirroring the managed button path. The plain-text card the router
        # previously emitted left multi/option intermediate prompts answerable
        # only via the "Other" free-text fallback, even when buttons were meant
        # to be the UX.
        await _acknowledge_and_send_next_prompt(
            message=message,
            task_service=task_service,
            user_id=user_id,
            next_prompt=next_prompt,
        )


def _register_middleware(
    router: Router,
    session_service: SessionService,
    *,
    admin_password_service: AdminPasswordService | None = None,
) -> tuple[SessionGuardMiddleware, SessionGuardMiddleware]:
    """Register global middleware (error handling, session guards). Returns guard instances."""
    error_handling_middleware = ErrorHandlingMiddleware()
    router.message.middleware(error_handling_middleware)
    router.callback_query.middleware(error_handling_middleware)

    guard_basic = SessionGuardMiddleware(session_service, require_active=False)
    guard_active = SessionGuardMiddleware(session_service, require_active=True)
    return guard_basic, guard_active


def _register_optional_handlers(
    router: Router,
    *,
    guard_basic: SessionGuardMiddleware,
    guard_active: SessionGuardMiddleware,
    session_callbacks: CallbackValidatorMiddleware,
    permission_callbacks: CallbackValidatorMiddleware,
    settings: Settings,
    task_service: TaskService,
    session_service: SessionService,
    registry_service: SessionRegistryService | None,
    file_receiver: FileReceiverService | None,
    upload_queue: UploadQueueManager | None,
    result_exporter: ResultExporterService | None,
    external_discovery: ExternalSessionDiscoveryService | None,
    external_binder: ExternalSessionBinder | None,
    structured_session_store: SessionStore | None,
    hook_socket_server: HookSocketServer | None,
    unbound_permission_handler: UnboundPermissionHandler | None,
    external_uq_state: ExternalUserQuestionState | None,
    permission_gateway: PermissionGateway | None,
    session_scanner: SessionScanner | None,
    claude_paths: ClaudePaths | None,
    liveness_enabled: bool,
    external_binding_reaper: ExternalBindingReaper | None,
    title_resolver: Callable[[str, str], str | None] | None,
    dead_unbound_cleanup: Callable[[str], Awaitable[object]] | None,
    external_session_input_service: ExternalSessionInputService | None,
    external_question_callbacks: CallbackValidatorMiddleware,
    user_question_callback_registry: UserQuestionCallbackRegistry | None,
) -> None:
    """Register optional handlers that depend on service availability."""
    if session_scanner is not None and claude_paths is not None:
        resume_active_router = Router()
        resume_active_router.message.middleware(guard_active)
        resume_active_router.callback_query.middleware(CallbackValidatorMiddleware(expected_parts=2, prefix="resume"))
        resume_active_router.callback_query.middleware(guard_active)
        register_resume_handler(
            resume_active_router,
            session_scanner=session_scanner,
            task_service=task_service,
            claude_paths=claude_paths,
        )
        router.include_router(resume_active_router)

    if registry_service is not None:
        list_router = Router()
        list_router.callback_query.middleware(CallbackValidatorMiddleware(expected_parts=(2, 3), prefix="sess"))
        register_list_handler(
            list_router,
            registry_service=registry_service,
            external_discovery=external_discovery,
            external_binder=external_binder,
            liveness_enabled=liveness_enabled,
            reaper=external_binding_reaper,
            title_resolver=title_resolver,
            dead_unbound_cleanup=dead_unbound_cleanup,
        )
        router.include_router(list_router)
        register_attach_handler(router, registry_service=registry_service)

    if external_discovery is not None and external_binder is not None:
        session_action_router = Router()
        session_action_router.callback_query.middleware(session_callbacks)
        register_session_action_handlers(
            session_action_router,
            discovery=external_discovery,
            binder=external_binder,
            registry_service=registry_service,
            external_session_input_service=external_session_input_service,
        )
        router.include_router(session_action_router)

        # Pairing callbacks live on a separate router with their own ``tpair`` prefix; sharing
        # ``sess`` middleware would reject tpair buttons (and vice versa), since
        # CallbackValidatorMiddleware answers+consumes any callback whose prefix does not match.
        if external_session_input_service is not None:
            tpair_router = Router()
            tpair_router.callback_query.middleware(CallbackValidatorMiddleware(expected_parts=2, prefix="tpair"))
            register_pair_consume_handler(tpair_router, input_service=external_session_input_service, session_service=session_service)
            router.include_router(tpair_router)

            # Free-text answer router for external Ghostty AskUserQuestion.
            # Registered BEFORE the ordinary external text router so a typed answer
            # takes the question path instead of being sent into the terminal. The
            # filter requires exactly one active Ghostty question and no managed
            # pending; the handler re-resolves to guard against a stale match.
            if external_uq_state is not None:
                external_question_text_router = Router()
                question_filter = ExternalQuestionActiveFilter(external_uq_state, task_service)

                @external_question_text_router.message(question_filter, F.text & ~F.text.startswith("/"))
                async def answer_external_user_question_with_text(message: Message) -> None:
                    await answer_external_user_question_text(
                        message,
                        external_uq_state=external_uq_state,
                        task_service=task_service,
                    )

                router.include_router(external_question_text_router)

            # External text injection: match only when the user has an active
            # Ghostty input target and no active managed chat session (the
            # filter yields to ``chat_text_router`` on coexistence). Deliberately
            # NOT guarded by ``guard_active`` (that would reject users without a
            # managed session). ``F.text`` does not exclude slash commands, so
            # an unregistered slash like ``/compact`` reaches Claude as text when
            # a target is active; registered slashes are handled by earlier
            # command routers which are included before this one. When the filter
            # returns False the message falls through to ``chat_text_router``
            # (UNHANDLED propagation).
            external_text_router = Router()
            target_filter = ExternalInputTargetActiveFilter(external_session_input_service, session_service)
            register_external_text_handlers(external_text_router, input_service=external_session_input_service, target_filter=target_filter)
            router.include_router(external_text_router)

    if external_discovery is not None and external_binder is not None and structured_session_store is not None:
        register_external_session_handler(
            router,
            discovery=external_discovery,
            binder=external_binder,
            session_store=structured_session_store,
            input_service=external_session_input_service,
        )

    if hook_socket_server is not None and unbound_permission_handler is not None and permission_gateway is not None:
        ext_perm_router = Router()
        ext_perm_router.callback_query.middleware(permission_callbacks)
        ext_uq_router = Router()
        ext_uq_router.callback_query.middleware(external_question_callbacks)
        register_external_permission_handler(
            ext_perm_router,
            uq_router=ext_uq_router,
            hook_socket_server=hook_socket_server,
            unbound_permission_handler=unbound_permission_handler,
            external_uq_state=external_uq_state,
            permission_gateway=permission_gateway,
            user_question_callback_registry=user_question_callback_registry,
        )
        router.include_router(ext_perm_router)
        router.include_router(ext_uq_router)

    if file_receiver is not None and upload_queue is not None:
        upload_guard_router = Router()
        upload_guard_router.message.middleware(guard_basic)
        register_file_upload_handler(
            upload_guard_router,
            file_receiver=file_receiver,
            session_service=session_service,
            task_service=task_service,
            upload_queue=upload_queue,
            upload_max_file_size_mb=settings.upload_max_file_size_mb,
            upload_queue_ttl_sec=settings.upload_queue_ttl_sec,
        )
        router.include_router(upload_guard_router)

    if result_exporter is not None:
        register_export_handler(
            router,
            task_service=task_service,
            result_exporter=result_exporter,
        )


def _create_chat_text_router(
    *,
    guard_active: SessionGuardMiddleware,
    task_service: TaskService,
    session_service: SessionService,
    registry_service: SessionRegistryService | None,
    sender_factory: Callable[[], ChunkSender],
    diff_generator: DiffGeneratorService | None,
    result_exporter: ResultExporterService | None,
    status_display: StatusDisplayService | None,
    queued_upload_scheduler: Callable[[Message, int, str], None] | None,
    pending_upload_finalizer: Callable[[Message, int], Awaitable[None]] | None,
    permission_gateway: PermissionGateway | None,
    stream_background_tasks: BackgroundTaskRegistry,
    structured_reply_pump_interval_sec: float,
    spinner_initial_delay_sec: float,
    spinner_interval_sec: float,
) -> Router:
    """Create a sub-router for Claude chat text messages (requires active session)."""
    chat_text_router = Router()
    chat_text_router.message.middleware(guard_active)

    @chat_text_router.message(F.text & ~F.text.startswith("/"))
    async def command_claude_chat_text(message: Message, session: SessionContext) -> None:
        text = (message.text or "").strip()
        if not text:
            return

        user_id = message.from_user.id if message.from_user else 0
        if await maybe_handle_pending_user_question_text(message=message, task_service=task_service):
            return
        logger.info(
            "claude chat text received",
            extra={
                "user_id": user_id,
                "text_len": len(text),
                "has_session": True,
                "claude_chat_active": session.claude_chat_active,
                "session_provider": session.provider,
                "session_workdir": session.workdir,
                "session_claude_session_id": session.claude_session_id,
            },
        )

        # Auto-reattach: validate tmux session is still alive
        if registry_service is not None and session.terminal_id:
            reattached = await registry_service.validate_or_reattach(user_id)
            if reattached is not None:
                session = reattached

        stream_task = await run_prompt_and_stream(
            message=message,
            task_service=task_service,
            sender_factory=sender_factory,
            user_id=user_id,
            provider="claude_code",
            prompt=text,
            workdir=session.workdir,
            diff_generator=diff_generator,
            result_exporter=result_exporter,
            status_display=status_display,
            queued_upload_scheduler=queued_upload_scheduler,
            pending_upload_finalizer=pending_upload_finalizer,
            permission_gateway=permission_gateway,
            stream_background_tasks=stream_background_tasks,
            structured_reply_pump_interval_sec=structured_reply_pump_interval_sec,
            spinner_initial_delay_sec=spinner_initial_delay_sec,
            spinner_interval_sec=spinner_interval_sec,
        )
        logger.info(
            "claude chat stream spawned",
            extra={
                "user_id": user_id,
                "workdir": session.workdir,
                "claude_session_id": session.claude_session_id,
                "task_created": stream_task is not None,
            },
        )

    return chat_text_router


def create_router(
    *,
    settings: Settings,
    task_service: TaskService,
    session_service: SessionService,
    registry_service: SessionRegistryService | None = None,
    file_receiver: FileReceiverService | None = None,
    upload_queue: UploadQueueManager | None = None,
    upload_background_tasks: BackgroundTaskRegistry | None = None,
    upload_processing_locks: RefCountedLockRegistry | None = None,
    stream_background_tasks: BackgroundTaskRegistry | None = None,
    result_exporter: ResultExporterService | None = None,
    diff_generator: DiffGeneratorService | None = None,
    status_display: StatusDisplayService | None = None,
    external_discovery: ExternalSessionDiscoveryService | None = None,
    external_binder: ExternalSessionBinder | None = None,
    structured_session_store: SessionStore | None = None,
    hook_socket_server: HookSocketServer | None = None,
    unbound_permission_handler: UnboundPermissionHandler | None = None,
    external_uq_state: ExternalUserQuestionState | None = None,
    permission_gateway: PermissionGateway | None = None,
    session_scanner: SessionScanner | None = None,
    claude_paths: ClaudePaths | None = None,
    liveness_enabled: bool = False,
    external_binding_reaper: ExternalBindingReaper | None = None,
    title_resolver: Callable[[str, str], str | None] | None = None,
    dead_unbound_cleanup: Callable[[str], Awaitable[object]] | None = None,
    admin_password_service: AdminPasswordService | None = None,
    external_session_input_service: ExternalSessionInputService | None = None,
    user_question_callback_registry: UserQuestionCallbackRegistry | None = None,
) -> Router:
    router = Router()

    # 注册中间件
    guard_basic, guard_active = _register_middleware(router, session_service, admin_password_service=admin_password_service)

    # 回调数据验证中间件
    session_callbacks = CallbackValidatorMiddleware(expected_parts=3, prefix="sess")
    permission_callbacks = CallbackValidatorMiddleware(expected_parts=3, prefix="ext_perm")
    external_question_callbacks = CallbackValidatorMiddleware(expected_parts=2, prefix="ext_uq")
    user_question_callbacks = CallbackValidatorMiddleware(
        expected_parts=(2, 4, 5),
        prefix="ask",
    )

    # /start 命令
    @router.message(Command("start"))
    async def command_start(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else 0
        session = await session_service.get(user_id)
        session_text = (
            f"session_id: {session.session_id}\n"
            f"provider: {session.provider}\n"
            f"workdir: {session.workdir}\n"
            f"claude_chat_active: {session.claude_chat_active}"
            if session
            else "session: 尚未创建"
        )
        providers = ", ".join(task_service.available_providers())
        await message.answer(
            "欢迎使用 Telegram CLI Gateway\n"
            "命令:\n"
            "/run <provider> <task text>\n"
            "/claude [workdir] (开启 Claude 会话模式)\n"
            "/list (查看活跃会话)\n"
            "/attach <terminal_id> (连接到会话)\n"
            "/detach (断开当前会话)\n"
            "/status [task_id]\n"
            "/cancel <task_id>\n"
            "/session [provider] [workdir]\n"
            "/approve\n"
            "/deny [reason]\n"
            "/exit 或 /quit (退出 Claude 会话并关闭持久终端)\n"
            f"可用 provider: {providers}\n"
            f"{session_text}"
        )

    # Sender 和上传调度工厂
    sender_factory = lambda: ChunkSender(
        chunk_size=settings.chunk_size,
        flush_interval_sec=settings.chunk_flush_interval_sec,
    )

    queued_upload_scheduler = None
    pending_upload_finalizer = None
    if file_receiver is not None and upload_queue is not None:
        # 走 upload 路径时，后台 task registry 与串行处理锁必须由组合根同时注入。
        assert upload_background_tasks is not None and upload_processing_locks is not None

        def _queued_upload_scheduler(message: Message, user_id: int, completed_task_id: str) -> None:
            schedule_pending_upload_processing(
                message,
                file_receiver=file_receiver,
                session_service=session_service,
                upload_queue=upload_queue,
                upload_background_tasks=upload_background_tasks,
                upload_processing_locks=upload_processing_locks,
                user_id=user_id,
                task_service=task_service,
                completed_task_id=completed_task_id,
            )

        async def _pending_upload_finalizer(message: Message, user_id: int) -> None:
            await flush_pending_uploads_for_task_start(
                message,
                file_receiver=file_receiver,
                session_service=session_service,
                upload_queue=upload_queue,
                upload_processing_locks=upload_processing_locks,
                user_id=user_id,
                task_service=task_service,
            )

        queued_upload_scheduler = _queued_upload_scheduler
        pending_upload_finalizer = _pending_upload_finalizer

    # 核心命令处理器
    # /run 与 Claude 聊天自由文本都走后台 watchdog task，stream 后台 registry
    # 必须由组合根注入。
    assert stream_background_tasks is not None
    register_run_handler(
        router,
        task_service=task_service,
        sender_factory=sender_factory,
        diff_generator=diff_generator,
        result_exporter=result_exporter,
        status_display=status_display,
        queued_upload_scheduler=queued_upload_scheduler,
        pending_upload_finalizer=pending_upload_finalizer,
        permission_gateway=permission_gateway,
        stream_background_tasks=stream_background_tasks,
        structured_reply_pump_interval_sec=settings.structured_reply_pump_interval_sec,
        spinner_initial_delay_sec=settings.spinner_initial_delay_sec,
        spinner_interval_sec=settings.spinner_interval_sec,
    )
    register_claude_handler(router, task_service=task_service)
    register_cancel_handler(router, task_service=task_service, admin_password_service=admin_password_service)
    register_status_handler(router, task_service=task_service)
    register_session_handler(
        router, task_service=task_service, session_service=session_service, admin_password_service=admin_password_service
    )
    if permission_gateway is not None:
        permission_router = Router()
        permission_router.callback_query.middleware(CallbackValidatorMiddleware(expected_parts=3, prefix="perm"))
        register_permission_handlers(
            permission_router,
            permission_gateway=permission_gateway,
        )
        router.include_router(permission_router)
    uq_router = Router()
    uq_router.callback_query.middleware(user_question_callbacks)
    register_user_question_handlers(uq_router, task_service=task_service, callback_registry=user_question_callback_registry)
    router.include_router(uq_router)
    register_exit_handler(router, task_service=task_service)
    # 子路由器：需要活跃会话的命令
    cmds_active_router = Router()
    cmds_active_router.message.middleware(guard_active)
    cmds_active_router.callback_query.middleware(CallbackValidatorMiddleware(prefix="clcmd"))
    cmds_active_router.callback_query.middleware(guard_active)
    register_cmds_handler(
        cmds_active_router,
        task_service=task_service,
        permission_gateway=permission_gateway,
        stream_background_tasks=stream_background_tasks,
    )
    router.include_router(cmds_active_router)

    # 可选处理器（依赖服务可用性）
    _register_optional_handlers(
        router,
        guard_basic=guard_basic,
        guard_active=guard_active,
        session_callbacks=session_callbacks,
        permission_callbacks=permission_callbacks,
        settings=settings,
        task_service=task_service,
        session_service=session_service,
        registry_service=registry_service,
        file_receiver=file_receiver,
        upload_queue=upload_queue,
        result_exporter=result_exporter,
        external_discovery=external_discovery,
        external_binder=external_binder,
        structured_session_store=structured_session_store,
        hook_socket_server=hook_socket_server,
        unbound_permission_handler=unbound_permission_handler,
        external_uq_state=external_uq_state,
        permission_gateway=permission_gateway,
        session_scanner=session_scanner,
        claude_paths=claude_paths,
        liveness_enabled=liveness_enabled,
        external_binding_reaper=external_binding_reaper,
        title_resolver=title_resolver,
        dead_unbound_cleanup=dead_unbound_cleanup,
        external_session_input_service=external_session_input_service,
        external_question_callbacks=external_question_callbacks,
        user_question_callback_registry=user_question_callback_registry,
    )

    @router.message(PendingAdminPasswordFilter(admin_password_service), F.text & ~F.text.startswith("/"))
    async def admin_password_text(message: Message) -> None:
        await maybe_handle_admin_password_text(
            message,
            task_service=task_service,
            session_service=session_service,
            admin_password_service=admin_password_service,
        )

    # Claude 聊天文本子路由器
    chat_text_router = _create_chat_text_router(
        guard_active=guard_active,
        task_service=task_service,
        session_service=session_service,
        registry_service=registry_service,
        sender_factory=sender_factory,
        diff_generator=diff_generator,
        result_exporter=result_exporter,
        status_display=status_display,
        queued_upload_scheduler=queued_upload_scheduler,
        pending_upload_finalizer=pending_upload_finalizer,
        permission_gateway=permission_gateway,
        stream_background_tasks=stream_background_tasks,
        structured_reply_pump_interval_sec=settings.structured_reply_pump_interval_sec,
        spinner_initial_delay_sec=settings.spinner_initial_delay_sec,
        spinner_interval_sec=settings.spinner_interval_sec,
    )
    router.include_router(chat_text_router)

    return router
