from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession

from app.adapters.claude.hook_installer import HookInstaller
from app.adapters.claude.hook_socket_server import HookSocketServer
from app.adapters.claude.paths import ClaudePaths
from app.adapters.cli.factory import CLIAdapterFactory
from app.adapters.process.ghostty_terminal_adapter import GhosttyTerminalAdapter
from app.adapters.process.kitty_terminal_adapter import KittyTerminalAdapter
from app.adapters.process.subprocess_runner import SubprocessRunner
from app.adapters.process.tmux_runner import TmuxRunner
from app.adapters.storage.file_session_context_store import FileSessionContextStore
from app.adapters.storage.file_session_store import FileSessionStore
from app.adapters.storage.memory import MemoryTaskStore
from app.adapters.storage.upload_store import UploadStoreAdapter
from app.bootstrap_base import AppContainerBase
from app.bootstrap_mixins import (
    EventDispatchMixin,
    HookHandlingMixin,
    JsonlSyncMixin,
    PeriodicRecheckMixin,
    SessionMatchingMixin,
    SessionRestoreMixin,
    WatcherMixin,
)
from app.bot.adapters.message_sender import AiogramMessageSender
from app.bot.middleware.auth import AuthMiddleware
from app.bot.middleware.rate_limit import RateLimitMiddleware
from app.bot.presenters.permission_message_builder import PermissionMessageBuilder
from app.bot.router import create_router
from app.config.settings import Settings
from app.domain.external_terminal_models import TerminalKind
from app.domain.protocols import ExternalTerminalBackendProtocol
from app.domain.session_tombstone import SessionTombstoneStore
from app.infra.lock_registry import RefCountedLockRegistry
from app.services.admin_password_service import AdminPasswordService
from app.services.auto_approve_service import AutoApproveService
from app.services.background_task_registry import BackgroundTaskRegistry
from app.services.claude_jsonl_parser import ClaudeJSONLParser
from app.services.context_builder import ContextBuilderService
from app.services.diff_generator import DiffGeneratorService
from app.services.external_binding_cleanup_service import ExternalBindingCleanupService
from app.services.external_binding_cleanup_task import ExternalBindingCleanupTask
from app.services.external_binding_reaper import ExternalBindingReaper
from app.services.external_binding_store import ExternalBindingStore
from app.services.external_input_mode_state import ExternalInputTargetStore
from app.services.external_input_queue import ExternalInputQueue
from app.services.external_reply_delivery_pump import ExternalReplyDeliveryPump
from app.services.external_session_binder import ExternalSessionBinder
from app.services.external_session_discovery import ExternalSessionDiscoveryService
from app.services.external_session_input_service import ExternalSessionInputService
from app.services.external_session_push_notifier import ExternalSessionPushNotifier
from app.services.file_receiver import FileReceiverService
from app.services.file_sender import FileSenderService
from app.services.janitor_task import JanitorTask
from app.services.jsonl_file_watcher import JSONLFileWatcher
from app.services.local_process_probe import LocalProcessProbe
from app.services.pairing_callback_registry import PairingCallbackRegistry
from app.services.periodic_janitor import PeriodicJanitor
from app.services.permission_callback_registry import PermissionCallbackRegistry
from app.services.permission_gateway import PermissionGateway
from app.services.result_exporter import ResultExporterService
from app.services.risk_evaluator import RiskEvaluator
from app.services.session_ownership_resolver import SessionOwnershipResolver
from app.services.session_registry import SessionRegistryService
from app.services.session_scanner import SessionScanner
from app.services.session_service import SessionService
from app.services.session_store import SessionStore
from app.services.session_supervisor import SessionSupervisor
from app.services.status_display import StatusDisplayService
from app.services.task_service import TaskService
from app.services.unbound_permission_handler import UnboundPermissionHandler
from app.services.upload_cleanup import UploadCleanupService
from app.services.upload_queue import UploadQueueManager

logger = logging.getLogger(__name__)


class AppContainer(
    JsonlSyncMixin,
    HookHandlingMixin,
    SessionMatchingMixin,
    WatcherMixin,
    PeriodicRecheckMixin,
    SessionRestoreMixin,
    EventDispatchMixin,
    AppContainerBase,
):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._init_bot()
        self._init_storage_and_runners()
        self._init_file_services()
        self._init_core_services()
        self._init_session_and_task_services()
        self._init_external_services()
        self._init_infrastructure()

    def _init_bot(self) -> None:
        """Initialize Telegram bot and dispatcher."""
        settings = self.settings
        session_kwargs: dict[str, object] = {"timeout": settings.tg_request_timeout_sec}
        if settings.tg_proxy_url:
            session_kwargs["proxy"] = settings.tg_proxy_url

        try:
            session = AiohttpSession(**session_kwargs)  # type: ignore[arg-type]
        except RuntimeError as exc:
            if settings.tg_proxy_url and "aiohttp-socks" in str(exc):
                raise RuntimeError(
                    "检测到 TG_PROXY_URL，但缺少 aiohttp-socks。请执行: python3 -m pip install aiohttp-socks，或移除 TG_PROXY_URL"
                ) from exc
            raise

        self.bot = Bot(token=settings.tg_bot_token, session=session)
        self.dispatcher = Dispatcher()

    def _init_storage_and_runners(self) -> None:
        """Initialize storage backends, hooks, and CLI runners."""
        settings = self.settings
        self.task_store = MemoryTaskStore(
            max_records=settings.task_store_max_records,
            ttl_hours=settings.task_store_ttl_hours,
        )
        self.runner = SubprocessRunner()
        self.claude_paths = ClaudePaths.resolve(settings.claude_config_dir)
        self.hook_installer = HookInstaller(
            paths=self.claude_paths,
            socket_path=settings.claude_hook_socket_path,
            claude_bin=settings.claude_cli_bin,
        )
        self.hook_socket_server = HookSocketServer(
            settings.claude_hook_socket_path,
            allowed_workdirs=settings.allowed_workdirs,
            max_message_bytes=settings.claude_hook_max_message_bytes,
            pending_permission_ttl_sec=settings.claude_hook_pending_permission_ttl_sec,
            max_pending_permissions=settings.claude_hook_max_pending_permissions,
        )
        self.permission_callback_registry = PermissionCallbackRegistry(
            ttl_sec=settings.claude_hook_pending_permission_ttl_sec,
        )
        self.tombstone_store = SessionTombstoneStore(ttl_seconds=settings.tombstone_ttl_sec)
        self.auto_approve_service = AutoApproveService(tombstone=self.tombstone_store)
        self.admin_password_service = AdminPasswordService(settings.admin_password or "")
        self.permission_message_builder = PermissionMessageBuilder()
        self.file_session_store = FileSessionStore(settings.tmux_data_dir)
        self.session_context_store = FileSessionContextStore(self.file_session_store)
        self.claude_jsonl_parser = ClaudeJSONLParser(self.claude_paths)
        self.structured_session_store = SessionStore(self.file_session_store)
        self.jsonl_file_watcher = JSONLFileWatcher(
            projects_dir=self.claude_paths.projects_dir,
            on_change=lambda session_id, cwd: self.session_supervisor.schedule_jsonl_sync(session_id, cwd),
            enabled=settings.jsonl_file_watcher_enabled,
        )
        self.session_supervisor = SessionSupervisor(
            session_store=self.structured_session_store,
            claude_jsonl_parser=self.claude_jsonl_parser,
            on_jsonl_sync=self.sync_claude_session,
            on_dispatch_event=self._dispatch_session_event,
            poll_interval_sec=settings.session_supervisor_poll_interval_sec,
            idle_poll_interval_sec=settings.session_supervisor_idle_poll_interval_sec,
            debounce_sec=settings.claude_jsonl_sync_debounce_ms / 1000,
            jsonl_file_watcher=self.jsonl_file_watcher,
        )
        self.tmux_runner = TmuxRunner(
            tmux_bin=settings.tmux_bin,
            data_dir=settings.tmux_data_dir,
            poll_interval_sec=settings.tmux_poll_interval_sec,
            enter_delay_sec=settings.tmux_enter_delay_sec,
            partial_flush_sec=settings.tmux_partial_flush_sec,
            interactive_completion_grace_sec=settings.tmux_completion_grace_sec,
            claude_cli_bin=settings.claude_cli_bin,
            file_store=self.file_session_store,
            session_store=self.structured_session_store,
            session_lock_ttl_sec=settings.session_lock_ttl_sec,
            lock_cleanup_interval_sec=settings.lock_cleanup_interval_sec,
            lock_cleanup_batch_size=settings.lock_cleanup_batch_size,
        )
        self.cli_factory = CLIAdapterFactory(
            settings=settings,
            runner=self.runner,
            tmux_runner=self.tmux_runner,
        )

    def _init_file_services(self) -> None:
        """Initialize file upload, receive, and queue services."""
        settings = self.settings
        self.upload_store = UploadStoreAdapter(base_dir=settings.default_workdir, cleanup_roots=settings.allowed_workdirs)
        self.file_receiver = FileReceiverService(
            upload_store=self.upload_store,
            allowed_extensions=set(settings.allowed_file_extensions),
            max_file_size_bytes=settings.upload_max_file_size_mb * 1024 * 1024,
        )
        self.upload_queue = UploadQueueManager(
            max_files_per_user=settings.upload_queue_max_files_per_user,
            max_bytes_per_user=settings.effective_upload_queue_max_bytes_per_user,
            ttl_sec=settings.upload_queue_ttl_sec,
            cleanup_interval_sec=settings.upload_queue_cleanup_interval_sec,
        )

    def _init_core_services(self) -> None:
        """Initialize message sender, exporters, and diff generator."""
        settings = self.settings
        self.message_sender = AiogramMessageSender(self.bot)
        self.file_sender = FileSenderService(
            message_sender=self.message_sender,
            enabled=settings.auto_file_send_enabled,
            extensions=set(settings.auto_file_send_extensions),
            image_extensions={".png", ".jpg", ".jpeg", ".gif", ".webp"},
        )
        self.context_builder = ContextBuilderService(upload_store=self.upload_store)
        self.result_exporter = ResultExporterService(settings=settings)
        self.diff_generator = DiffGeneratorService()
        self.status_display = StatusDisplayService(bot=self.bot)
        self.upload_cleanup = UploadCleanupService(
            upload_store=self.upload_store,
            interval_minutes=settings.upload_cleanup_interval_min,
            max_age_hours=settings.upload_expiry_hours,
        )

    def _init_session_and_task_services(self) -> None:
        """Initialize session service, task service, and session registry."""
        settings = self.settings
        claude_session_capable_providers = frozenset(
            p for p in self.cli_factory.available_providers() if self.cli_factory.capabilities(p).session_state
        )
        self.session_service = SessionService(
            store=self.session_context_store,
            claude_session_capable_providers=claude_session_capable_providers,
        )
        self.task_service = TaskService(
            settings=settings,
            task_store=self.task_store,
            session_service=self.session_service,
            cli_factory=self.cli_factory,
            semaphore=asyncio.Semaphore(settings.max_concurrent_tasks),
            structured_session_store=self.structured_session_store,
            hook_socket_server=self.hook_socket_server,
            context_builder=self.context_builder,
            auto_approve_service=self.auto_approve_service,
        )
        self.session_registry = SessionRegistryService(
            session_service=self.session_service,
            lookup=self.structured_session_store._lookup,
            tmux_runner=self.tmux_runner,
            repository=self.structured_session_store._repository,
            auto_approve_service=self.auto_approve_service,
            health_check_interval_sec=settings.session_health_check_interval_sec,
        )

    def _init_external_services(self) -> None:
        """Initialize external session discovery, binding, permission, and notification services."""
        settings = self.settings
        # External input uses an independent per-session lock registry (design §6):
        # it is NOT part of the reply-delivery → session-event lock order, and must
        # stay out of _init_infrastructure so ExternalSessionInputService can be built
        # here (this method runs before _init_infrastructure).
        self._input_locks = RefCountedLockRegistry(
            ttl_sec=settings.session_lock_ttl_sec,
            cleanup_interval_sec=settings.lock_cleanup_interval_sec,
            cleanup_batch_size=settings.lock_cleanup_batch_size,
        )
        self.external_binding_store = ExternalBindingStore(
            data_dir=Path(settings.tmux_data_dir),
        )
        self.external_discovery = ExternalSessionDiscoveryService(
            stale_timeout_sec=settings.external_session_stale_timeout_sec,
            title_resolver=lambda sid, cwd: self.claude_jsonl_parser.extract_session_title(session_id=sid, cwd=cwd),
            tombstone=self.tombstone_store,
        )
        self.ownership_resolver = SessionOwnershipResolver(
            session_service=self.session_service,
            binding_store=self.external_binding_store,
        )
        # Built before the binder so bind-time tty backfill (pid → controlling
        # tty) has a probe to consult; also reused by the external input service.
        self.local_process_probe = LocalProcessProbe()
        self.external_binder = ExternalSessionBinder(
            discovery=self.external_discovery,
            binding_store=self.external_binding_store,
            projects_dir=Path("~/.claude/projects").expanduser(),
            sync_callback=self._sync_and_baseline_external_reply,
            save_callback=self._save_external_binding,
            remove_callback=self._unbind_external_binding,
            process_probe=self.local_process_probe,
        )
        self.unbound_permission_handler = UnboundPermissionHandler(
            message_sender=self.message_sender,
            hook_socket_server=self.hook_socket_server,
            allowed_user_ids=settings.effective_unbound_permission_notify_user_id_set,
            permission_ttl_sec=settings.claude_hook_pending_permission_ttl_sec,
            title_resolver=lambda sid, cwd: self.claude_jsonl_parser.extract_session_title(session_id=sid, cwd=cwd),
            notify_user_ids_resolver=self._resolve_unbound_permission_notify_user_ids,
        )
        self.risk_evaluator = RiskEvaluator(
            enabled=settings.risk_eval_enabled,
            dangerous_commands=settings.risk_eval_dangerous_commands,
            dangerous_paths=settings.risk_eval_dangerous_paths,
            protected_paths=settings.risk_eval_protected_paths,
            auto_approve_max_risk=settings.risk_eval_auto_approve_max_risk,
        )
        self.permission_gateway = PermissionGateway(
            registry=self.permission_callback_registry,
            auto_approve_service=self.auto_approve_service,
            task_service=self.task_service,
            hook_socket_server=self.hook_socket_server,
            unbound_responder=self.unbound_permission_handler,
            settings=settings,
            message_sender=self.message_sender,
            message_builder=self.permission_message_builder,
            risk_evaluator=self.risk_evaluator,
        )
        self.unbound_permission_handler.set_permission_gateway(self.permission_gateway)

        # External user question state for PTY injection
        from app.services.external_user_question_state import ExternalUserQuestionState

        # Same TTL source as the callback registry below: the two stores must
        # agree (see the registry comment) so a live button never outlives — or
        # is pruned ahead of — its pending question.
        self.external_uq_state = ExternalUserQuestionState(
            ttl_sec=settings.user_question_callback_ttl_sec,
        )
        # Opaque token registry shared by Telegram AskUserQuestion callbacks (managed
        # tmux + external Ghostty + external tmux) so identity never travels in
        # callback_data. TTL seconds match the external pending-question TTL so a live
        # button never resolves against an already-pruned pending question; both stores
        # judge expiry on a monotonic clock (immune to wall-clock jumps) — the pending
        # store keeps wall-clock timestamps only for snapshot display.
        from app.services.user_question_callback_registry import UserQuestionCallbackRegistry

        self.user_question_callback_registry = UserQuestionCallbackRegistry(
            ttl_sec=settings.user_question_callback_ttl_sec,
        )
        self.push_notifier = ExternalSessionPushNotifier(
            message_sender=self.message_sender,
            binding_store=self.external_binding_store,
            permission_gateway=self.permission_gateway,
            retry_count=settings.push_notification_retry_count,
            external_uq_state=self.external_uq_state,
            user_question_callback_registry=self.user_question_callback_registry,
        )

        self.external_binding_reaper = ExternalBindingReaper(
            binding_store=self.external_binding_store,
            auto_approve_service=self.auto_approve_service,
            hook_socket_server=self.hook_socket_server,
            permission_callback_registry=self.permission_callback_registry,
            unbound_permission_handler=self.unbound_permission_handler,
            external_uq_state=self.external_uq_state,
            user_question_callback_registry=self.user_question_callback_registry,
            external_discovery=self.external_discovery,
            tombstone=self.tombstone_store,
            remove_callback=self._remove_external_binding,
        )

        self.external_binding_cleanup_service = ExternalBindingCleanupService(
            binding_store=self.external_binding_store,
            hook_socket_server=self.hook_socket_server,
            reaper=self.external_binding_reaper,
            liveness_enabled=settings.external_binding_pid_liveness_enabled,
            ttl=timedelta(hours=settings.external_binding_idle_ttl_hours),
            interval_sec=settings.session_health_check_interval_sec,
        )

        # External terminal session input (design specs/2026-08-03-external-ghostty-input-design.md §4-9,
        # extended to kitty by specs/2026-10-29-external-kitty-input-design.md).
        # Built unconditionally so enabled=False short-circuits internally and the rest of the
        # binding/permission/reply system keeps working; the service methods are no-ops when off.
        self.ghostty_adapter = GhosttyTerminalAdapter(
            enable_applescript=settings.ghostty_applescript_enabled,
        )
        self.kitty_adapter = KittyTerminalAdapter(
            kitten_bin=settings.kitty_bin,
            listen_on=settings.kitty_listen_on,
            timeout_sec=settings.kitty_timeout_sec,
            key_delay_sec=settings.kitty_key_delay_sec,
            enable_remote_control=settings.kitty_input_enabled,
        )
        # The kitty backend is only registered when enabled; a missing backend
        # makes every kitty-targeted operation fail closed.
        backends: dict[TerminalKind, ExternalTerminalBackendProtocol] = {"ghostty": self.ghostty_adapter}
        if settings.kitty_input_enabled:
            backends["kitty"] = self.kitty_adapter
        self.pairing_callback_registry = PairingCallbackRegistry(
            ttl_sec=settings.external_pairing_token_ttl_sec,
        )
        self.external_input_mode_store = ExternalInputTargetStore()
        self.external_input_queue = ExternalInputQueue(
            max_size=settings.external_input_queue_max_size,
            ttl_sec=settings.external_input_queue_ttl_sec,
        )
        self.external_session_input_service = ExternalSessionInputService(
            enabled=settings.external_input_enabled,
            binding_store=self.external_binding_store,
            session_store=self.structured_session_store,
            backends=backends,
            process_probe=self.local_process_probe,
            pairing_registry=self.pairing_callback_registry,
            input_mode_store=self.external_input_mode_store,
            input_queue=self.external_input_queue,
            input_locks=self._input_locks,
            external_user_question_state=self.external_uq_state,
            user_question_callback_registry=self.user_question_callback_registry,
            drain_publish_wait_timeout_sec=settings.external_drain_publish_wait_timeout_sec,
            # Queue-drop notices (design §9) ride the same push channel.
            notify_user=self.push_notifier.notify_info,
        )

        # Hook external question transport/state/registry into the managed
        # UserQuestionService *after* all external services are built, to avoid
        # reordering the bootstrap dependency ring (design §6 / §10).
        self.task_service.configure_external(
            external_uq_state=self.external_uq_state,
            external_question_transport=self.external_session_input_service,
            callback_registry=self.user_question_callback_registry,
        )

    async def _resolve_unbound_permission_notify_user_ids(self) -> set[int]:
        configured_user_ids = self.settings.unbound_permission_notify_user_id_set
        if configured_user_ids:
            return configured_user_ids

        user_ids = {session.user_id for session in await self.session_service.list_all()}
        user_ids.update(binding.user_id for binding in self.external_binding_store.list_all())
        if not user_ids:
            logger.warning(
                "unbound permission notification has no recipients; set UNBOUND_PERMISSION_NOTIFY_USER_IDS when TG_ALLOWED_USER_IDS=*"
            )
        return user_ids

    def _session_lock_registry(self) -> RefCountedLockRegistry:
        """Build a per-session-id lock registry with the configured TTL/policy."""
        settings = self.settings
        return RefCountedLockRegistry(
            ttl_sec=settings.session_lock_ttl_sec,
            cleanup_interval_sec=settings.lock_cleanup_interval_sec,
            cleanup_batch_size=settings.lock_cleanup_batch_size,
        )

    def _init_infrastructure(self) -> None:
        """Initialize lock registries, background tasks, and janitor."""
        settings = self.settings
        self._jsonl_sync_locks = self._session_lock_registry()
        self._session_event_locks = self._session_lock_registry()
        self._external_reply_delivery_locks = self._session_lock_registry()
        self._background_tasks = BackgroundTaskRegistry(label="bootstrap")
        # upload 队列 drain 的后台 task 与按用户串行处理锁——由组合根构造并
        # 注入 file_upload handler，停机时随 _stop_background_tasks 一并 cancel。
        # 此前这两者在 handler 模块顶层创建，脱离容器停机序列（CLAUDE.md 禁止
        # handler 直接创建后台任务），现回归组合根装配。
        self._upload_background_tasks = BackgroundTaskRegistry(label="upload")
        self._upload_processing_locks = RefCountedLockRegistry(
            ttl_sec=settings.upload_processing_lock_ttl_sec,
            cleanup_interval_sec=settings.lock_cleanup_interval_sec,
            cleanup_batch_size=settings.lock_cleanup_batch_size,
        )
        # /run、Claude 聊天自由文本、/cmds 回调的后台 watchdog task——此前是
        # command_run 模块顶层的裸 set，停机时不会被 cancel_all，已脱管。现由
        # 组合根构造并注入，停机时一并 cancel。
        self._stream_background_tasks = BackgroundTaskRegistry(label="stream")
        self.external_reply_delivery_pump = ExternalReplyDeliveryPump(
            session_store=self.structured_session_store,
            binding_store=self.external_binding_store,
            background_tasks=self._background_tasks,
            sync_callback=self.sync_claude_session,
            drain_callback=self._drain_bound_assistant_replies,
            finalize_callback=self._finalize_bound_external_session,
        )
        self._janitor = PeriodicJanitor()
        self._external_binding_cleanup_task = ExternalBindingCleanupTask(
            cleanup_service=self.external_binding_cleanup_service,
            interval_seconds=self.settings.session_health_check_interval_sec,
        )
        self._janitor_task = JanitorTask(
            janitor=self._janitor,
            interval_seconds=5.0,
        )
        self._pending_dead_unbound_cleanup_ids: dict[str, int] = {}  # session_id -> retry_count
        self._dead_unbound_cleanup_max_retries = 5
        self._started = False
        self._stopping = False

    async def _cleanup_dead_unbound_external_session(self, session_id: str) -> bool:
        """Invalidate pending state for a dead-pruned unbound external session."""

        async def invalidate_external_uq_state() -> int:
            return self.external_uq_state.invalidate_session(session_id)

        cleanup_steps: tuple[tuple[str, Callable[[], Awaitable[object]]], ...] = (
            ("auto approve service", lambda: self.auto_approve_service.clear_session(session_id)),
            ("permission callback registry", lambda: self.permission_callback_registry.invalidate_session(session_id)),
            ("unbound permission handler", lambda: self.unbound_permission_handler.invalidate_session(session_id)),
            ("external user question state", invalidate_external_uq_state),
            (
                "user question callback registry",
                lambda: self.user_question_callback_registry.invalidate_session(session_id),
            ),
            ("hook pending permissions", lambda: self.hook_socket_server.cancel_pending_permissions(session_id=session_id)),
        )
        success = True
        for label, cleanup in cleanup_steps:
            try:
                await cleanup()
            except Exception:
                success = False
                logger.exception("dead unbound external session cleanup failed", extra={"session_id": session_id, "step": label})
        if success:
            self._pending_dead_unbound_cleanup_ids.pop(session_id, None)
        else:
            retry_count = self._pending_dead_unbound_cleanup_ids.get(session_id, 0) + 1
            if retry_count >= self._dead_unbound_cleanup_max_retries:
                logger.warning(
                    "dead unbound external session cleanup exceeded max retries, giving up",
                    extra={"session_id": session_id, "retry_count": retry_count},
                )
                self._pending_dead_unbound_cleanup_ids.pop(session_id, None)
            else:
                self._pending_dead_unbound_cleanup_ids[session_id] = retry_count
        return success

    async def _prune_unbound_external_sessions(self) -> None:
        """Prune in-memory unbound external session discovery entries."""
        dead_ids: list[str] = []
        try:
            dead_ids = self.external_discovery.prune_dead()
        except Exception:
            logger.exception("external discovery dead-prune failed")
        for session_id in dead_ids:
            if session_id not in self._pending_dead_unbound_cleanup_ids:
                self._pending_dead_unbound_cleanup_ids[session_id] = 0
        for session_id in sorted(self._pending_dead_unbound_cleanup_ids):
            await self._cleanup_dead_unbound_external_session(session_id)
        self.external_discovery.prune_stale()

    async def start(self) -> None:
        if self._started:
            return
        self._stopping = False
        self.external_reply_delivery_pump.reopen()
        self.session_supervisor.reopen()
        self.hook_socket_server.pause_ingress()
        try:
            await self.hook_socket_server.start(
                self._handle_hook_event,
                self._handle_permission_failure,
                self._handle_permission_resolved,
            )

            # Register command menu (best-effort)
            try:
                from app.bot.commands import BOT_COMMANDS

                await self.bot.set_my_commands(BOT_COMMANDS)
            except Exception as exc:
                logger.warning("Failed to register bot commands: %s", exc)
            if self.settings.claude_install_hooks:
                self.hook_installer.install()
            self.jsonl_file_watcher.start()
            if self.settings.claude_tmux_mode:
                await self.session_registry.reconcile_terminal_lifecycle()
            await self._restore_session_bindings()

            # Initial cleanup passes (before restoring external delivery tasks or starting periodic loops)
            await self.external_binding_cleanup_service.run_cleanup()
            await self._restore_external_reply_delivery_pumps()
            await self.upload_cleanup.run_cleanup()
            self.hook_socket_server.resume_ingress()

            # Register periodic jobs
            self._janitor.register(
                "upload_queue_cleanup",
                self.settings.upload_queue_cleanup_interval_sec,
                self.upload_queue.prune_expired,
            )
            self._janitor.register(
                "upload_file_cleanup",
                self.settings.upload_cleanup_interval_min * 60,
                self.upload_cleanup.run_cleanup,
            )
            self._janitor.register(
                "external_discovery_cleanup",
                self.settings.session_health_check_interval_sec,
                self._prune_unbound_external_sessions,
            )
            self._janitor.register(
                "session_health_check",
                self.settings.session_health_check_interval_sec,
                self.session_registry.reconcile_terminal_lifecycle,
            )

            async def _cleanup_stale_sessions() -> None:
                await asyncio.to_thread(
                    self.file_session_store.cleanup_stale_sessions,
                    self.settings.session_cleanup_max_age_hours,
                )

            self._janitor.register(
                "periodic_recheck",
                self.settings.claude_periodic_recheck_ms / 1000,
                self._recheck_active_claude_sessions,
            )
            self._janitor.register(
                "session_cleanup",
                self.settings.session_cleanup_interval_sec,
                _cleanup_stale_sessions,
            )
            self._external_binding_cleanup_task.start()
            self._janitor_task.start()
            self._started = True
        except BaseException:
            try:
                await self.stop()
            except Exception:
                logger.exception("cleanup after startup failure failed")
            raise

    async def stop(self) -> None:
        self._stopping = True
        self.hook_socket_server.pause_ingress()
        try:
            await self._janitor_task.stop()
            await self._external_binding_cleanup_task.stop()
            await self.hook_socket_server.stop()
            self.jsonl_file_watcher.stop()
            await self.external_reply_delivery_pump.stop_all()
            await self.session_supervisor.stop_all()
            await self._stop_background_tasks()
            await self.external_session_input_service.shutdown()
            self.external_binding_store.flush()
        finally:
            await self.bot.session.close()
            self._started = False

    def wire(self) -> None:
        auth_middleware = AuthMiddleware(
            self.settings.allowed_user_id_set,
            allow_all_users=self.settings.allow_all_users,
        )
        rate_limit_middleware = RateLimitMiddleware(
            limit=self.settings.rate_limit_max_requests,
            window_sec=self.settings.rate_limit_window_sec,
            bucket_ttl_sec=self.settings.effective_rate_limit_bucket_ttl_sec,
            cleanup_interval_sec=self.settings.rate_limit_bucket_cleanup_interval_sec,
            cleanup_batch_size=self.settings.rate_limit_bucket_cleanup_batch_size,
        )
        self.dispatcher.message.middleware(auth_middleware)
        self.dispatcher.callback_query.middleware(auth_middleware)
        self.dispatcher.message.middleware(rate_limit_middleware)
        self.dispatcher.callback_query.middleware(rate_limit_middleware)

        router = create_router(
            settings=self.settings,
            task_service=self.task_service,
            session_service=self.session_service,
            registry_service=self.session_registry,
            file_receiver=self.file_receiver,
            upload_queue=self.upload_queue,
            upload_background_tasks=self._upload_background_tasks,
            upload_processing_locks=self._upload_processing_locks,
            stream_background_tasks=self._stream_background_tasks,
            result_exporter=self.result_exporter,
            diff_generator=self.diff_generator,
            status_display=self.status_display,
            external_discovery=self.external_discovery,
            external_binder=self.external_binder,
            structured_session_store=self.structured_session_store,
            hook_socket_server=self.hook_socket_server,
            unbound_permission_handler=self.unbound_permission_handler,
            external_uq_state=self.external_uq_state,
            permission_gateway=self.permission_gateway,
            session_scanner=SessionScanner(),
            claude_paths=self.claude_paths,
            liveness_enabled=self.settings.external_binding_pid_liveness_enabled,
            external_binding_reaper=self.external_binding_reaper,
            title_resolver=lambda sid, cwd: self.claude_jsonl_parser.extract_session_title(session_id=sid, cwd=cwd),
            dead_unbound_cleanup=self._cleanup_dead_unbound_external_session,
            admin_password_service=self.admin_password_service,
            external_session_input_service=self.external_session_input_service,
            user_question_callback_registry=self.user_question_callback_registry,
        )
        self.dispatcher.include_router(router)
