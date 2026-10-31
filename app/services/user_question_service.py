from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.adapters.claude.hook_socket_server import HookSocketServer
from app.domain.protocols import (
    AdapterCapabilities,
    ClaudeTerminalRuntimeProtocol,
    ClaudeUserQuestionTransportProtocol,
    ExternalClaudeUserQuestionTransportProtocol,
)
from app.domain.session_models import PermissionDecisionPayload, SessionEvent, SessionEventType, SessionPhase, SessionState, ToolStatus
from app.domain.user_question_models import (
    ExternalQuestionActionResult,
    ExternalQuestionActionStatus,
    ExternalTerminalQuestionTarget,
    ExternalUserQuestionContext,
    UserQuestionPrompt,
    compose_user_question_answers,
    extract_user_question_prompts,
)
from app.infra.user_question_constants import (
    USER_QUESTION_TUI_FALLBACK_ERROR,
    USER_QUESTION_TUI_FALLBACK_ERROR_PREFIX,
)
from app.services.external_user_question_state import (
    ExternalQuestionPendingAmbiguous,
    ExternalQuestionPendingNone,
    ExternalQuestionPendingResolution,
    ExternalQuestionPendingUnique,
    ExternalUserQuestionState,
    PendingExternalUserQuestionSnapshot,
)
from app.services.session_service import SessionService
from app.services.session_store import SessionStore
from app.services.structured_session_resolver import StructuredSessionResolver
from app.services.user_question_callback_registry import (
    QuestionCallbackTokens,
    UserQuestionCallbackOrigin,
)

if TYPE_CHECKING:
    from app.services.user_question_callback_registry import UserQuestionCallbackRegistry


@dataclass
class _UserQuestionDraft:
    tool_use_id: str
    prompts: tuple[UserQuestionPrompt, ...]
    answers_by_index: dict[int, str] = field(default_factory=dict)
    selected_option_indexes_by_question: dict[int, set[int]] = field(default_factory=dict)
    use_text_transport: bool = False


class UserQuestionService:
    def __init__(
        self,
        *,
        session_service: SessionService,
        terminal_runtime: ClaudeTerminalRuntimeProtocol,
        user_question_transport: ClaudeUserQuestionTransportProtocol | None,
        structured_session_store: SessionStore | None,
        hook_socket_server: HookSocketServer | None,
        session_resolver: StructuredSessionResolver,
        capabilities_resolver: Callable[[str], AdapterCapabilities],
    ) -> None:
        self._session_service = session_service
        self._terminal_runtime = terminal_runtime
        self._user_question_transport = user_question_transport
        self._structured_session_store = structured_session_store
        self._hook_socket_server = hook_socket_server
        self._session_resolver = session_resolver
        self._capabilities_resolver = capabilities_resolver
        self._user_question_drafts: dict[int, _UserQuestionDraft] = {}
        self._completed_user_question_tool_use_ids_by_user: dict[int, set[str]] = {}
        self._user_question_locks: dict[int, asyncio.Lock] = {}
        self._external_uq_state: ExternalUserQuestionState | None = None
        self._external_question_transport: ExternalClaudeUserQuestionTransportProtocol | None = None
        self._callback_registry: UserQuestionCallbackRegistry | None = None

    def configure_external(
        self,
        *,
        external_uq_state: ExternalUserQuestionState | None,
        external_question_transport: ExternalClaudeUserQuestionTransportProtocol | None,
        callback_registry: UserQuestionCallbackRegistry | None,
    ) -> None:
        """Late-bind external terminal question state/transport/callback registry.

        Called once by the composition root after all external services are built,
        to avoid reordering the bootstrap dependency ring (design §6 / §10).
        """
        if self._external_uq_state is not None or self._external_question_transport is not None or self._callback_registry is not None:
            raise RuntimeError("external user-question context is already configured")
        self._external_uq_state = external_uq_state
        self._external_question_transport = external_question_transport
        self._callback_registry = callback_registry

    def clear_user(self, user_id: int) -> None:
        self._user_question_drafts.pop(user_id, None)
        self._completed_user_question_tool_use_ids_by_user.pop(user_id, None)
        self._user_question_locks.pop(user_id, None)

    def _get_user_question_lock(self, *, user_id: int) -> asyncio.Lock:
        lock = self._user_question_locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._user_question_locks[user_id] = lock
        return lock

    async def get_structured_user_question_cursor(self, user_id: int) -> str | None:
        if self._structured_session_store is None:
            return None
        state = await self._session_resolver.get_structured_session(user_id, log_missing=False)
        if state is not None:
            cursor = self._structured_session_store.get_structured_user_question_cursor(state.session_id)
            if cursor is not None:
                return cursor
        draft = self._user_question_drafts.get(user_id)
        if draft is not None:
            targeted_state = self._structured_session_store.find_by_active_user_question_tool_use_id(draft.tool_use_id)
            if targeted_state is not None and await self._session_resolver.is_state_owned_by_user(state=targeted_state, user_id=user_id):
                return self._structured_session_store.get_structured_user_question_cursor(targeted_state.session_id)
        return None

    async def acknowledge_structured_user_question(self, user_id: int, *, question_key: str | None = None) -> None:
        if self._structured_session_store is None or question_key is None:
            return
        state = self._structured_session_store.find_by_active_user_question_key(question_key)
        if state is not None and not await self._session_resolver.is_state_owned_by_user(state=state, user_id=user_id):
            state = None
        if state is None:
            draft = self._user_question_drafts.get(user_id)
            if draft is not None and draft.tool_use_id:
                state = self._structured_session_store.find_by_active_user_question_tool_use_id(draft.tool_use_id)
                if state is not None and not await self._session_resolver.is_state_owned_by_user(state=state, user_id=user_id):
                    state = None
        if state is None:
            state = await self._session_resolver.get_structured_session(user_id, log_missing=False)
        if state is None or not await self._session_resolver.is_state_owned_by_user(state=state, user_id=user_id):
            return
        self._structured_session_store.mark_structured_user_question_emitted(state.session_id, question_key=question_key)

    async def get_pending_user_questions(self, user_id: int) -> tuple[UserQuestionPrompt, ...]:
        _, prompts = await self._resolve_active_user_question_context(user_id=user_id)
        self._sync_user_question_draft(user_id, prompts)
        return prompts

    async def register_question_callback_tokens(
        self,
        *,
        user_id: int,
        prompt: UserQuestionPrompt,
    ) -> QuestionCallbackTokens:
        """Register opaque callback tokens for one question prompt.

        Dispatches by active context: an active external terminal pending question
        yields ``EXTERNAL_TERMINAL`` tokens; otherwise the managed structured session
        state yields ``MANAGED`` tokens. Returns a non-tokenised bundle (legacy inline
        callback_data) when no callback registry or session id is available.
        """
        registry = self._callback_registry
        if registry is None or not prompt.options:
            return QuestionCallbackTokens()
        external_state = self._external_uq_state
        if external_state is not None:
            pending = external_state.get_active(prompt.tool_use_id)
            if pending is not None and pending.user_id == user_id and isinstance(pending.target, ExternalTerminalQuestionTarget):
                return await registry.register_question_tokens(
                    owner_user_id=user_id,
                    session_id=pending.session_id,
                    tool_use_id=prompt.tool_use_id,
                    question_index=prompt.question_index,
                    option_count=len(prompt.options),
                    multi_select=prompt.multi_select,
                    origin=UserQuestionCallbackOrigin.EXTERNAL_TERMINAL,
                )
        session_id = await self._resolve_question_session_id(user_id=user_id, tool_use_id=prompt.tool_use_id)
        if session_id is None:
            return QuestionCallbackTokens()
        return await registry.register_question_tokens(
            owner_user_id=user_id,
            session_id=session_id,
            tool_use_id=prompt.tool_use_id,
            question_index=prompt.question_index,
            option_count=len(prompt.options),
            multi_select=prompt.multi_select,
            origin=UserQuestionCallbackOrigin.MANAGED,
        )

    async def _resolve_question_session_id(self, *, user_id: int, tool_use_id: str) -> str | None:
        if self._structured_session_store is None or not tool_use_id:
            return None
        state = self._structured_session_store.find_by_active_user_question_tool_use_id(tool_use_id)
        if state is None or not await self._session_resolver.is_state_owned_by_user(state=state, user_id=user_id):
            state = await self._session_resolver.get_structured_session(user_id, log_missing=False)
        if state is None or not await self._session_resolver.is_state_owned_by_user(state=state, user_id=user_id):
            return None
        return state.session_id

    async def answer_pending_user_question_option(
        self,
        *,
        user_id: int,
        tool_use_id: str,
        question_index: int,
        option_index: int,
    ) -> tuple[bool, str, UserQuestionPrompt | None]:
        async with self._get_user_question_lock(user_id=user_id):
            state, prompts = await self._resolve_active_user_question_context(
                user_id=user_id,
                expected_tool_use_id=tool_use_id,
            )
            if not prompts:
                external_prompts = self._external_pending_prompts(tool_use_id=tool_use_id, user_id=user_id)
                if external_prompts is None:
                    return False, "当前没有待处理的选择题", None
                prompts = external_prompts

            draft, prompt = self._locate_user_question_prompt(
                user_id=user_id,
                prompts=prompts,
                tool_use_id=tool_use_id,
                question_index=question_index,
            )
            if prompt is None:
                return False, "这个选择按钮已经过期，请等待最新的问题", None
            if option_index < 0 or option_index >= len(prompt.options):
                return False, "无效的选项", None
            if prompt.multi_select:
                return False, "这个问题需要先勾选，再点击提交选择", None

            answer = prompt.options[option_index].label
            if self._external_pending_active(tool_use_id=tool_use_id, user_id=user_id):
                return await self._submit_external_user_question_answer(
                    user_id=user_id,
                    draft=draft,
                    prompt=prompt,
                    answer=answer,
                    action="select",
                    option_index=option_index,
                )
            return await self._submit_user_question_answer(
                user_id=user_id,
                draft=draft,
                prompt=prompt,
                answer=answer,
                state=state,
                selected_option_index=option_index,
            )

    async def toggle_pending_user_question_multi_select_option(
        self,
        *,
        user_id: int,
        tool_use_id: str,
        question_index: int,
        option_index: int,
    ) -> tuple[bool, str, UserQuestionPrompt | None, frozenset[int] | None]:
        async with self._get_user_question_lock(user_id=user_id):
            state, prompts = await self._resolve_active_user_question_context(
                user_id=user_id,
                expected_tool_use_id=tool_use_id,
            )
            if not prompts:
                external_prompts = self._external_pending_prompts(tool_use_id=tool_use_id, user_id=user_id)
                if external_prompts is None:
                    return False, "当前没有待处理的选择题", None, None
                prompts = external_prompts

            draft, prompt = self._locate_user_question_prompt(
                user_id=user_id,
                prompts=prompts,
                tool_use_id=tool_use_id,
                question_index=question_index,
            )
            if prompt is None:
                return False, "这个选择按钮已经过期，请等待最新的问题", None, None
            if not prompt.multi_select:
                return False, "这个问题不是多选题", None, None
            if option_index < 0 or option_index >= len(prompt.options):
                return False, "无效的选项", None, None

            if self._external_pending_active(tool_use_id=tool_use_id, user_id=user_id):
                toggled, text = await self._apply_external_user_question_toggle(
                    prompt=prompt,
                    option_index=option_index,
                )
                if not toggled:
                    # fail-closed on external terminal: no managed-terminal fallback
                    return False, text, None, None

                selected_option_indexes = draft.selected_option_indexes_by_question.setdefault(question_index, set())
                if option_index in selected_option_indexes:
                    selected_option_indexes.remove(option_index)
                    message = f"已取消: {prompt.options[option_index].label}"
                else:
                    selected_option_indexes.add(option_index)
                    message = f"已选择: {prompt.options[option_index].label}"
                if not selected_option_indexes:
                    draft.selected_option_indexes_by_question.pop(question_index, None)
                return True, message, prompt, frozenset(sorted(selected_option_indexes))

            if not draft.use_text_transport:
                toggled, text = await self._toggle_user_question_option_in_terminal(
                    user_id=user_id,
                    state=state,
                    prompt=prompt,
                    option_index=option_index,
                )
                if not toggled:
                    if self._is_user_question_text_transport_fallback_error(text):
                        draft.use_text_transport = True
                    else:
                        return False, text, None, None

            selected_option_indexes = draft.selected_option_indexes_by_question.setdefault(question_index, set())
            if option_index in selected_option_indexes:
                selected_option_indexes.remove(option_index)
                message = f"已取消: {prompt.options[option_index].label}"
            else:
                selected_option_indexes.add(option_index)
                message = f"已选择: {prompt.options[option_index].label}"
            if not selected_option_indexes:
                draft.selected_option_indexes_by_question.pop(question_index, None)
            return True, message, prompt, frozenset(sorted(selected_option_indexes))

    async def submit_pending_user_question_multi_select(
        self,
        *,
        user_id: int,
        tool_use_id: str,
        question_index: int,
    ) -> tuple[bool, str, UserQuestionPrompt | None]:
        async with self._get_user_question_lock(user_id=user_id):
            state, prompts = await self._resolve_active_user_question_context(
                user_id=user_id,
                expected_tool_use_id=tool_use_id,
            )
            if not prompts:
                external_prompts = self._external_pending_prompts(tool_use_id=tool_use_id, user_id=user_id)
                if external_prompts is None:
                    return False, "当前没有待处理的选择题", None
                prompts = external_prompts

            draft, prompt = self._locate_user_question_prompt(
                user_id=user_id,
                prompts=prompts,
                tool_use_id=tool_use_id,
                question_index=question_index,
            )
            if prompt is None:
                return False, "这个选择按钮已经过期，请等待最新的问题", None
            if not prompt.multi_select:
                return False, "这个问题不是多选题", None

            selected_option_indexes = draft.selected_option_indexes_by_question.get(question_index, set())
            if not selected_option_indexes:
                return False, "请至少勾选一项再提交", None

            answer = "、".join(prompt.options[index].label for index in range(len(prompt.options)) if index in selected_option_indexes)
            remaining = self._remaining_user_question_prompts(draft=draft, prompt=prompt)
            if self._external_pending_active(tool_use_id=tool_use_id, user_id=user_id):
                # Move Claude TUI past current question, without dispersing selected indexes.
                if not remaining:
                    return await self._submit_external_user_question_answer(
                        user_id=user_id,
                        draft=draft,
                        prompt=prompt,
                        answer=answer,
                        action="advance_multi",
                    )
                advanced, text = await self._apply_external_user_question_advance(
                    prompt=prompt,
                    final_question=False,
                )
                if not advanced:
                    return False, text, None
                draft.answers_by_index[prompt.question_index] = answer
                draft.selected_option_indexes_by_question.pop(prompt.question_index, None)
                next_prompt = remaining[0]
                await self._mark_user_question_prompt_emitted(user_id=user_id, state=state, prompt=next_prompt)
                return True, f"已记录选择: {answer}", next_prompt
            if not draft.use_text_transport:
                advanced, text = await self._advance_user_question_after_multi_select(
                    user_id=user_id,
                    state=state,
                    final_question=not remaining,
                )
                if not advanced:
                    if self._is_user_question_text_transport_fallback_error(text):
                        draft.use_text_transport = True
                    else:
                        return False, text, None
            return await self._submit_user_question_answer(
                user_id=user_id,
                draft=draft,
                prompt=prompt,
                answer=answer,
                state=state,
                answer_applied_in_terminal=not draft.use_text_transport,
            )

    async def answer_pending_user_question_text(
        self,
        *,
        user_id: int,
        text: str,
    ) -> tuple[bool, str, UserQuestionPrompt | None]:
        answer = text.strip()
        if not answer:
            return False, "回复内容不能为空", None

        async with self._get_user_question_lock(user_id=user_id):
            # Resolve the managed structured-session pending question FIRST. When
            # the user has an active managed AskUserQuestion, a typed answer must
            # answer that managed question — never a simultaneously-live external
            # terminal question. Previously the external probe ran first and, when
            # exactly one external terminal question existed for the same user, it
            # hijacked the answer to the external terminal, leaving the managed
            # session's permission request permanently unanswered (Claude blocked).
            state, prompts = await self._resolve_active_user_question_context(user_id=user_id)
            if not prompts:
                # External terminal free-text routing (design §5): eligible only when
                # there is exactly one active external terminal question AND no
                # managed pending question. Ambiguous → reject and consume the
                # message; never fall back to managed. ``answer`` is stripped only
                # for the emptiness guard — the raw text is forwarded to the terminal.
                external_target = self._resolve_external_free_text_target(user_id=user_id)
                if isinstance(external_target, ExternalQuestionPendingAmbiguous):
                    return False, "存在多个待处理的外部问题，请用按钮精确回答", None
                if isinstance(external_target, ExternalQuestionPendingUnique):
                    pending = external_target.pending
                    draft = self._ensure_user_question_draft(
                        user_id=user_id,
                        prompts=pending.prompts,
                    )
                    prompt = next(
                        (item for item in pending.prompts if item.question_index not in draft.answers_by_index),
                        None,
                    )
                    if prompt is None:
                        return False, "当前没有待处理的选择题", None
                    return await self._submit_external_user_question_answer(
                        user_id=user_id,
                        draft=draft,
                        prompt=prompt,
                        answer=answer,
                        action="answer_text",
                        text=text,
                    )
                return False, "当前没有待处理的选择题", None

            draft = self._ensure_user_question_draft(user_id=user_id, prompts=prompts)
            prompt = next((item for item in prompts if item.question_index not in draft.answers_by_index), None)
            if prompt is None:
                return False, "当前没有待处理的选择题", None

            return await self._submit_user_question_answer(
                user_id=user_id,
                draft=draft,
                prompt=prompt,
                answer=answer,
                state=state,
            )

    def _extract_active_user_questions(self, state: SessionState | None) -> tuple[UserQuestionPrompt, ...]:
        if state is None:
            return ()

        pending = getattr(state, "pending_permission", None)
        if pending is not None:
            pending_prompts = extract_user_question_prompts(
                tool_use_id=pending.tool_use_id,
                tool_name=pending.tool_name,
                tool_input=pending.tool_input,
            )
            if pending_prompts:
                return pending_prompts

        waiting_prompts = self._extract_latest_user_question_prompts_from_tools(
            state,
            allowed_statuses={ToolStatus.WAITING_FOR_APPROVAL},
        )
        if waiting_prompts:
            return waiting_prompts

        if state.phase in {SessionPhase.PROCESSING, SessionPhase.COMPACTING}:
            return self._extract_latest_user_question_prompts_from_tools(
                state,
                allowed_statuses={ToolStatus.RUNNING},
            )
        return ()

    def _extract_latest_user_question_prompts_from_tools(
        self,
        state: SessionState,
        *,
        allowed_statuses: set[ToolStatus],
    ) -> tuple[UserQuestionPrompt, ...]:
        latest_prompts: tuple[UserQuestionPrompt, ...] = ()
        latest_started_at = None
        for tool in state.tool_calls.values():
            if tool.status not in allowed_statuses:
                continue
            prompts = extract_user_question_prompts(
                tool_use_id=tool.tool_use_id,
                tool_name=tool.name,
                tool_input=tool.input,
            )
            if not prompts:
                continue
            if latest_started_at is None or tool.started_at >= latest_started_at:
                latest_started_at = tool.started_at
                latest_prompts = prompts
        return latest_prompts

    def _extract_user_question_prompts_for_tool_use_id(
        self,
        state: SessionState | None,
        *,
        tool_use_id: str,
    ) -> tuple[UserQuestionPrompt, ...]:
        if state is None or not tool_use_id:
            return ()

        pending = getattr(state, "pending_permission", None)
        if pending is not None and pending.tool_use_id == tool_use_id:
            pending_prompts = extract_user_question_prompts(
                tool_use_id=pending.tool_use_id,
                tool_name=pending.tool_name,
                tool_input=pending.tool_input,
            )
            if pending_prompts:
                return pending_prompts

        tool = state.tool_calls.get(tool_use_id)
        if tool is None or tool.status not in {ToolStatus.RUNNING, ToolStatus.WAITING_FOR_APPROVAL}:
            return ()
        return extract_user_question_prompts(
            tool_use_id=tool.tool_use_id,
            tool_name=tool.name,
            tool_input=tool.input,
        )

    def _sync_user_question_draft(self, user_id: int, prompts: tuple[UserQuestionPrompt, ...]) -> None:
        draft = self._user_question_drafts.get(user_id)
        if not prompts:
            self._user_question_drafts.pop(user_id, None)
            return
        if draft is None:
            return
        if draft.tool_use_id != prompts[0].tool_use_id:
            self._user_question_drafts.pop(user_id, None)
            return
        draft.prompts = prompts
        prompt_by_index = {prompt.question_index: prompt for prompt in prompts}
        draft.answers_by_index = {
            question_index: answer for question_index, answer in draft.answers_by_index.items() if question_index in prompt_by_index
        }
        filtered_selected_option_indexes_by_question: dict[int, set[int]] = {}
        for question_index, selected_option_indexes in draft.selected_option_indexes_by_question.items():
            prompt = prompt_by_index.get(question_index)
            if prompt is None or not prompt.multi_select:
                continue
            valid_selected_option_indexes = {
                option_index for option_index in selected_option_indexes if 0 <= option_index < len(prompt.options)
            }
            if valid_selected_option_indexes:
                filtered_selected_option_indexes_by_question[question_index] = valid_selected_option_indexes
        draft.selected_option_indexes_by_question = filtered_selected_option_indexes_by_question

    def _ensure_user_question_draft(self, *, user_id: int, prompts: tuple[UserQuestionPrompt, ...]) -> _UserQuestionDraft:
        draft = self._user_question_drafts.get(user_id)
        tool_use_id = prompts[0].tool_use_id
        if draft is None or draft.tool_use_id != tool_use_id:
            draft = _UserQuestionDraft(tool_use_id=tool_use_id, prompts=prompts)
            self._user_question_drafts[user_id] = draft
            return draft
        draft.prompts = prompts
        return draft

    def _locate_user_question_prompt(
        self,
        *,
        user_id: int,
        prompts: tuple[UserQuestionPrompt, ...],
        tool_use_id: str,
        question_index: int,
    ) -> tuple[_UserQuestionDraft, UserQuestionPrompt | None]:
        draft = self._ensure_user_question_draft(user_id=user_id, prompts=prompts)
        if not prompts or prompts[0].tool_use_id != tool_use_id:
            return draft, None
        current_prompt = next((item for item in prompts if item.question_index not in draft.answers_by_index), None)
        if current_prompt is None or current_prompt.question_index != question_index:
            return draft, None
        return draft, current_prompt

    async def _resolve_active_user_question_context(
        self,
        *,
        user_id: int,
        expected_tool_use_id: str | None = None,
    ) -> tuple[SessionState | None, tuple[UserQuestionPrompt, ...]]:
        state = None
        if expected_tool_use_id and self._structured_session_store is not None:
            candidate = self._structured_session_store.find_by_active_user_question_tool_use_id(expected_tool_use_id)
            if candidate is not None and await self._session_resolver.is_state_owned_by_user(state=candidate, user_id=user_id):
                state = candidate
        if state is None:
            state = await self._session_resolver.get_structured_session(user_id, log_missing=False)
            if state is not None and not await self._session_resolver.is_state_owned_by_user(state=state, user_id=user_id):
                state = None
        current_state = state
        prompts = (
            self._extract_user_question_prompts_for_tool_use_id(state, tool_use_id=expected_tool_use_id)
            if expected_tool_use_id
            else self._extract_active_user_questions(state)
        )
        if not prompts and expected_tool_use_id is not None and current_state is not None:
            prompts = self._extract_active_user_questions(current_state)
        if prompts and expected_tool_use_id is not None and state is not None:
            active_prompts = self._extract_active_user_questions(state)
            if active_prompts and active_prompts[0].tool_use_id != expected_tool_use_id:
                return state, active_prompts
            if not active_prompts:
                return state, ()
        completed_tool_use_ids = self._completed_user_question_tool_use_ids_by_user.get(user_id, set())
        if prompts and prompts[0].tool_use_id in completed_tool_use_ids:
            return state, ()
        if not prompts and expected_tool_use_id is None and self._structured_session_store is not None:
            draft = self._user_question_drafts.get(user_id)
            if draft is not None:
                draft_state = self._structured_session_store.find_by_active_user_question_tool_use_id(draft.tool_use_id)
                if draft_state is not None and await self._session_resolver.is_state_owned_by_user(state=draft_state, user_id=user_id):
                    state = draft_state
                    prompts = self._extract_user_question_prompts_for_tool_use_id(state, tool_use_id=draft.tool_use_id)
        return state, prompts

    # ─── external terminal dispatch (design §6 / §9 / §10) ────────────

    def _external_pending_active(self, *, tool_use_id: str, user_id: int) -> bool:
        state = self._external_uq_state
        if state is None or not tool_use_id:
            return False
        snapshot = state.get_active(tool_use_id)
        return snapshot is not None and snapshot.user_id == user_id and isinstance(snapshot.target, ExternalTerminalQuestionTarget)

    def _external_pending_prompts(self, *, tool_use_id: str, user_id: int) -> tuple[UserQuestionPrompt, ...] | None:
        """Return the prompts of an active external terminal question (owner-checked).

        The option/toggle/submit button paths first try the managed structured
        session for prompts. For an external terminal session there is no managed
        pending — ``_resolve_active_user_question_context`` returns empty prompts,
        so we fall back to the external pending stored at push time. Returns None
        when no active external terminal question matches.
        """
        state = self._external_uq_state
        if state is None or not tool_use_id:
            return None
        snapshot = state.get_active(tool_use_id)
        if snapshot is None or snapshot.user_id != user_id or not isinstance(snapshot.target, ExternalTerminalQuestionTarget):
            return None
        return snapshot.prompts

    def _external_context(
        self,
        *,
        tool_use_id: str,
        snapshot_pending: PendingExternalUserQuestionSnapshot | None,
    ) -> ExternalUserQuestionContext | None:
        if snapshot_pending is None:
            return None
        target = snapshot_pending.target
        if not isinstance(target, ExternalTerminalQuestionTarget):
            return None
        return ExternalUserQuestionContext(
            tool_use_id=tool_use_id,
            session_id=snapshot_pending.session_id,
            user_id=snapshot_pending.user_id,
            target=target,
        )

    def _resolve_external_free_text_target(
        self,
        *,
        user_id: int,
    ) -> ExternalQuestionPendingResolution:
        state = self._external_uq_state
        if state is None:
            return ExternalQuestionPendingNone()
        return state.resolve_unique_active_for_user(user_id, group="external_terminal")

    async def _apply_external_user_question_toggle(
        self,
        *,
        prompt: UserQuestionPrompt,
        option_index: int,
    ) -> tuple[bool, str]:
        transport = self._external_question_transport
        state = self._external_uq_state
        if transport is None or state is None:
            return False, "外部终端问答功能未启用"
        context = self._external_context(
            tool_use_id=prompt.tool_use_id,
            snapshot_pending=state.get_active(prompt.tool_use_id),
        )
        if context is None:
            return False, "问题已过期或目标已变化"
        result = await transport.select_option(
            context=context,
            question_index=prompt.question_index,
            option_count=len(prompt.options),
            option_index=option_index,
            submit_after=False,
        )
        return self._external_action_to_bool(result)

    async def _apply_external_user_question_advance(
        self,
        *,
        prompt: UserQuestionPrompt,
        final_question: bool,
    ) -> tuple[bool, str]:
        transport = self._external_question_transport
        state = self._external_uq_state
        if transport is None or state is None:
            return False, "外部终端问答功能未启用"
        context = self._external_context(
            tool_use_id=prompt.tool_use_id,
            snapshot_pending=state.get_active(prompt.tool_use_id),
        )
        if context is None:
            return False, "问题已过期或目标已变化"
        result = await transport.advance_after_multi_select(
            context=context,
            question_index=prompt.question_index,
            option_count=len(prompt.options),
            final_question=final_question,
        )
        return self._external_action_to_bool(result)

    @staticmethod
    def _external_action_to_bool(result: ExternalQuestionActionResult) -> tuple[bool, str]:
        if result.status is ExternalQuestionActionStatus.APPLIED:
            return True, ""
        return False, result.message

    async def _submit_external_user_question_answer(
        self,
        *,
        user_id: int,
        draft: _UserQuestionDraft,
        prompt: UserQuestionPrompt,
        answer: str,
        action: str,
        option_index: int | None = None,
        text: str | None = None,
    ) -> tuple[bool, str, UserQuestionPrompt | None]:
        """Exactly-once external terminal question submit (design §9 / §10).

        Per-user lock already held by the caller; the transport acquires the
        per-session input lock internally. Final action sequence:
          transport.action (input lock: TUI + ACTIVE→TERMINAL_ACTION_APPLIED → release)
          → Hook allow (outside input lock)
          → question_completed (input lock: TERMINAL→COMPLETED + drain wake)
          → invalidate tokens / draft / completed set
        Fail-closed on REJECTED/INDETERMINATE: no managed-terminal fallback.
        """
        transport = self._external_question_transport
        state = self._external_uq_state
        if transport is None or state is None:
            return False, "外部终端问答功能未启用", None
        context = self._external_context(
            tool_use_id=prompt.tool_use_id,
            snapshot_pending=state.get_active(prompt.tool_use_id),
        )
        if context is None:
            return False, "问题已过期或目标已变化", None

        final = not self._remaining_user_question_prompts(draft=draft, prompt=prompt)

        if action == "select":
            result = await transport.select_option(
                context=context,
                question_index=prompt.question_index,
                option_count=len(prompt.options),
                option_index=option_index if option_index is not None else -1,
                submit_after=final,
            )
        elif action == "answer_text":
            result = await transport.answer_with_text(
                context=context,
                question_index=prompt.question_index,
                option_count=len(prompt.options),
                text=text if text is not None else answer,
                submit_after=final,
            )
        elif action == "advance_multi":
            result = await transport.advance_after_multi_select(
                context=context,
                question_index=prompt.question_index,
                option_count=len(prompt.options),
                final_question=final,
            )
        else:  # pragma: no cover - guarded by callers
            return False, "不支持的问答操作", None

        if result.status is ExternalQuestionActionStatus.REJECTED:
            return False, result.message, None
        if result.status is ExternalQuestionActionStatus.INDETERMINATE:
            await self._invalidate_external_question_tokens(session_id=context.session_id, tool_use_id=prompt.tool_use_id)
            return False, result.message or "终端操作结果不确定，请检查终端", None

        if not final:
            # Intermediate question: do not respond to the Hook permission.
            draft.answers_by_index[prompt.question_index] = answer
            draft.selected_option_indexes_by_question.pop(prompt.question_index, None)
            next_prompt = self._remaining_user_question_prompts(draft=draft, prompt=prompt)[0]
            await self._mark_user_question_prompt_emitted(user_id=user_id, state=None, prompt=next_prompt)
            return True, f"已记录选择: {answer}", next_prompt

        # Final question: input lock is already released by the transport. Call Hook
        # allow outside the external input lock (design §10). External sessions have
        # no managed pending_permission; send the allow straight to the Hook socket
        # so Claude can continue (the managed approve helper would skip it silently).
        approved, approve_text = await self._approve_external_pending_permission(
            tool_use_id=prompt.tool_use_id,
        )
        if not approved:
            await transport.question_indeterminate(
                context=context,
                reason=approve_text or "Hook allow failed",
            )
            await self._invalidate_external_question_tokens(session_id=context.session_id, tool_use_id=prompt.tool_use_id)
            return False, approve_text or "待处理权限请求已失效，请等待 Claude 重新发起", None

        await transport.question_completed(context=context)
        await self._invalidate_external_question_tokens(session_id=context.session_id, tool_use_id=prompt.tool_use_id)
        self._completed_user_question_tool_use_ids_by_user.setdefault(user_id, set()).add(prompt.tool_use_id)
        self._user_question_drafts.pop(user_id, None)
        return True, "已提交你的选择，Claude 继续执行中", None

    async def _invalidate_external_question_tokens(self, *, session_id: str, tool_use_id: str) -> None:
        registry = self._callback_registry
        if registry is None:
            return
        await registry.invalidate_tool(session_id=session_id, tool_use_id=tool_use_id)

    async def _submit_user_question_answer(
        self,
        *,
        user_id: int,
        draft: _UserQuestionDraft,
        prompt: UserQuestionPrompt,
        answer: str,
        state: SessionState | None = None,
        selected_option_index: int | None = None,
        answer_applied_in_terminal: bool = False,
    ) -> tuple[bool, str, UserQuestionPrompt | None]:
        normalized_answer = answer.strip()
        if not normalized_answer:
            return False, "回复内容不能为空", None

        remaining = self._remaining_user_question_prompts(draft=draft, prompt=prompt)
        if prompt.options and not answer_applied_in_terminal and not draft.use_text_transport:
            applied, text = await self._apply_user_question_answer_to_terminal(
                user_id=user_id,
                state=state,
                prompt=prompt,
                answer=normalized_answer,
                selected_option_index=selected_option_index,
                submit_after=not remaining,
            )
            if not applied:
                if self._is_user_question_text_transport_fallback_error(text):
                    draft.use_text_transport = True
                else:
                    return False, text, None

        if remaining:
            draft.answers_by_index[prompt.question_index] = normalized_answer
            draft.selected_option_indexes_by_question.pop(prompt.question_index, None)
            next_prompt = remaining[0]
            await self._mark_user_question_prompt_emitted(user_id=user_id, state=state, prompt=next_prompt)
            return True, f"已记录选择: {normalized_answer}", next_prompt

        if prompt.options and not draft.use_text_transport:
            approved, text = await self._approve_pending_user_question_permission(
                user_id=user_id,
                state=state,
                tool_use_id=prompt.tool_use_id,
            )
            if not approved:
                return False, text, None
            draft.answers_by_index[prompt.question_index] = normalized_answer
            draft.selected_option_indexes_by_question.pop(prompt.question_index, None)
            await self._mark_user_question_completed(user_id=user_id, state=state, tool_use_id=prompt.tool_use_id)
            self._completed_user_question_tool_use_ids_by_user.setdefault(user_id, set()).add(prompt.tool_use_id)
            self._user_question_drafts.pop(user_id, None)
            return True, "已提交你的选择，Claude 继续执行中", None

        draft.answers_by_index[prompt.question_index] = normalized_answer
        answer_text = compose_user_question_answers(draft.prompts, draft.answers_by_index)
        if not answer_text:
            return False, "答案内容为空，无法提交", None

        sent, text = await self._send_interactive_text(
            user_id=user_id,
            text=answer_text,
            terminal_id=state.terminal_id if state is not None else None,
            workdir=state.workdir if state is not None else None,
        )
        if sent:
            approved, approval_text = await self._approve_pending_user_question_permission(
                user_id=user_id,
                state=state,
                tool_use_id=prompt.tool_use_id,
            )
            if not approved:
                draft.answers_by_index.pop(prompt.question_index, None)
                return False, approval_text, None
            draft.selected_option_indexes_by_question.pop(prompt.question_index, None)
            await self._mark_user_question_completed(user_id=user_id, state=state, tool_use_id=prompt.tool_use_id)
            self._completed_user_question_tool_use_ids_by_user.setdefault(user_id, set()).add(prompt.tool_use_id)
            self._user_question_drafts.pop(user_id, None)
            return True, "已提交你的选择，Claude 继续执行中", None
        return False, text, None

    async def _approve_pending_user_question_permission(
        self,
        *,
        user_id: int,
        state: SessionState | None,
        tool_use_id: str,
    ) -> tuple[bool, str]:
        if self._hook_socket_server is None or not tool_use_id:
            return True, ""

        target_state = await self._resolve_user_question_target_state(user_id=user_id, state=state, tool_use_id=tool_use_id)
        pending = target_state.pending_permission if target_state is not None else None
        if pending is None or pending.tool_use_id != tool_use_id:
            return True, ""

        sent = await self._hook_socket_server.respond_to_permission(tool_use_id=tool_use_id, decision="allow")
        if not sent:
            return False, "待处理权限请求已失效，请等待 Claude 重新发起"
        return True, ""

    async def _approve_external_pending_permission(self, *, tool_use_id: str) -> tuple[bool, str]:
        """Allow exactly once for an external terminal session's final question.

        External sessions carry no managed ``pending_permission`` entry, so the
        managed approve helper would silently skip the Hook call (it treats a
        missing pending as “nothing to allow”). Claude is however blocked on
        this ``tool_use_id`` awaiting the allow decision; we therefore send it
        directly to the Hook socket. A hookless wiring (``_hook_socket_server``
        is None, e.g. tests / hookless configs) is treated as approved so the
        completion path doesn't hang.

        Race tolerance: ``respond_to_permission_outcome`` distinguishes the two
        False cases the boolean could not. ``"not_pending"`` means no pending
        connection exists for ``tool_use_id`` — PostToolUse already closed it
        after the transport delivered the TUI keystrokes and Claude read the
        answer, so the approve outcome is achieved in fact and we keep returning
        success. ``"write_failed"`` means the connection existed but the ``allow``
        write failed: Claude is still blocked on this ``allow``, so we must NOT
        claim success — convergence to ``INDETERMINATE`` happens in
        ``_submit_external_user_question_answer``.

        Call-once invariant: ``respond_to_permission_outcome`` POPS the pending
        connection under its lock before returning, so each ``tool_use_id`` maps to
        at most one meaningful call. The current final path calls this exactly once
        (``_submit_external_user_question_answer`` → allow → ``question_completed``),
        so the pop-and-forget semantics are safe. Do NOT add a second ``allow`` call
        on the same ``tool_use_id`` within one final-question submit: the first call
        would have already popped the connection, so the second would get
        ``"not_pending"`` — a status we deliberately treat as success here — and the
        allow would be falsely reported as delivered even though nothing was written.
        """
        if self._hook_socket_server is None or not tool_use_id:
            return True, ""
        outcome = await self._hook_socket_server.respond_to_permission_outcome(tool_use_id=tool_use_id, decision="allow")
        if outcome in ("wrote", "not_pending"):
            return True, ""
        return False, "待处理权限请求已失效，请等待 Claude 重新发起"

    async def _resolve_user_question_target_state(
        self,
        *,
        user_id: int,
        state: SessionState | None,
        tool_use_id: str,
    ) -> SessionState | None:
        if self._structured_session_store is None or not tool_use_id:
            return None

        target_state = self._structured_session_store.find_by_active_user_question_tool_use_id(tool_use_id)
        if target_state is None and state is not None:
            target_state = self._structured_session_store.get(state.session_id)
        if target_state is None or not await self._session_resolver.is_state_owned_by_user(state=target_state, user_id=user_id):
            return None
        return target_state

    async def _mark_user_question_completed(self, *, user_id: int, state: SessionState | None, tool_use_id: str) -> None:
        target_state = await self._resolve_user_question_target_state(user_id=user_id, state=state, tool_use_id=tool_use_id)
        if target_state is None or self._structured_session_store is None:
            return

        self._structured_session_store.process(
            SessionEvent(
                session_id=target_state.session_id,
                type=SessionEventType.PERMISSION_APPROVED,
                payload=PermissionDecisionPayload(tool_use_id=tool_use_id),
            )
        )

    async def _mark_user_question_prompt_emitted(self, *, user_id: int, state: SessionState | None, prompt: UserQuestionPrompt) -> None:
        if self._structured_session_store is None:
            return

        target_state = self._structured_session_store.find_by_active_user_question_key(prompt.key)
        if target_state is not None and not await self._session_resolver.is_state_owned_by_user(state=target_state, user_id=user_id):
            target_state = None
        if target_state is None and prompt.tool_use_id:
            target_state = self._structured_session_store.find_by_active_user_question_tool_use_id(prompt.tool_use_id)
            if target_state is not None and not await self._session_resolver.is_state_owned_by_user(state=target_state, user_id=user_id):
                target_state = None
        if target_state is None and state is not None:
            target_state = self._structured_session_store.get(state.session_id)
        if target_state is None or not await self._session_resolver.is_state_owned_by_user(state=target_state, user_id=user_id):
            return

        self._structured_session_store.mark_structured_user_question_emitted(
            target_state.session_id,
            question_key=prompt.key,
        )

    def _is_user_question_text_transport_fallback_error(self, text: str) -> bool:
        normalized = text.strip()
        return normalized.startswith(USER_QUESTION_TUI_FALLBACK_ERROR_PREFIX)

    def _remaining_user_question_prompts(
        self,
        *,
        draft: _UserQuestionDraft,
        prompt: UserQuestionPrompt,
    ) -> list[UserQuestionPrompt]:
        return [
            item
            for item in draft.prompts
            if item.question_index != prompt.question_index and item.question_index not in draft.answers_by_index
        ]

    async def _apply_user_question_answer_to_terminal(
        self,
        *,
        user_id: int,
        state: SessionState | None,
        prompt: UserQuestionPrompt,
        answer: str,
        selected_option_index: int | None,
        submit_after: bool,
    ) -> tuple[bool, str]:
        resolved_option_index = selected_option_index
        if resolved_option_index is None:
            resolved_option_index = next(
                (index for index, option in enumerate(prompt.options) if option.label == answer),
                None,
            )

        if resolved_option_index is not None:
            if prompt.multi_select:
                toggled, text = await self._toggle_user_question_option_in_terminal(
                    user_id=user_id,
                    state=state,
                    prompt=prompt,
                    option_index=resolved_option_index,
                )
                if not toggled:
                    return False, text
                return await self._advance_user_question_after_multi_select(
                    user_id=user_id,
                    state=state,
                    final_question=submit_after,
                )
            return await self._select_user_question_option_in_terminal(
                user_id=user_id,
                state=state,
                option_index=resolved_option_index,
                submit_after=submit_after,
            )

        return await self._answer_user_question_text_in_terminal(
            user_id=user_id,
            state=state,
            option_count=len(prompt.options),
            text=answer,
            submit_after=submit_after,
        )

    async def _resolve_user_question_terminal(
        self,
        *,
        user_id: int,
        state: SessionState | None,
    ) -> tuple[str | None, str | None, str | None]:
        if state is not None and state.terminal_id and state.workdir:
            if not await self._session_resolver.is_state_owned_by_user(state=state, user_id=user_id):
                return None, None, "当前没有可用的 Claude 持久终端"
            return state.terminal_id, state.workdir, None

        session = await self._session_service.get(user_id)
        if session is None or not self._capabilities_resolver(session.provider).user_question_tui:
            return None, None, "当前没有 Claude 会话"
        if not session.terminal_mode or not session.terminal_id:
            return None, None, "当前没有可用的 Claude 持久终端"
        return session.terminal_id, session.workdir, None

    async def _resolve_user_question_transport_context(
        self,
        *,
        user_id: int,
        state: SessionState | None,
    ) -> tuple[ClaudeUserQuestionTransportProtocol | None, str | None, str | None, str | None]:
        terminal_id, workdir, err = await self._resolve_user_question_terminal(user_id=user_id, state=state)
        if err is not None or terminal_id is None or workdir is None:
            return None, None, None, err or "当前没有可用的 Claude 持久终端"
        if self._user_question_transport is None:
            return None, None, None, USER_QUESTION_TUI_FALLBACK_ERROR
        return self._user_question_transport, terminal_id, workdir, None

    async def _select_user_question_option_in_terminal(
        self,
        *,
        user_id: int,
        state: SessionState | None,
        option_index: int,
        submit_after: bool,
    ) -> tuple[bool, str]:
        transport, terminal_id, workdir, err = await self._resolve_user_question_transport_context(user_id=user_id, state=state)
        if err is not None or transport is None or terminal_id is None or workdir is None:
            return False, err or USER_QUESTION_TUI_FALLBACK_ERROR
        return await transport.select_option(
            terminal_key=terminal_id,
            workdir=workdir,
            option_index=option_index,
            submit_after=submit_after,
        )

    async def _toggle_user_question_option_in_terminal(
        self,
        *,
        user_id: int,
        state: SessionState | None,
        prompt: UserQuestionPrompt,
        option_index: int,
    ) -> tuple[bool, str]:
        if not prompt.multi_select:
            return False, "这个问题不是多选题"
        return await self._select_user_question_option_in_terminal(
            user_id=user_id,
            state=state,
            option_index=option_index,
            submit_after=False,
        )

    async def _answer_user_question_text_in_terminal(
        self,
        *,
        user_id: int,
        state: SessionState | None,
        option_count: int,
        text: str,
        submit_after: bool,
    ) -> tuple[bool, str]:
        transport, terminal_id, workdir, err = await self._resolve_user_question_transport_context(user_id=user_id, state=state)
        if err is not None or transport is None or terminal_id is None or workdir is None:
            return False, err or USER_QUESTION_TUI_FALLBACK_ERROR
        return await transport.answer_with_text(
            terminal_key=terminal_id,
            workdir=workdir,
            option_count=option_count,
            text=text,
            submit_after=submit_after,
        )

    async def _advance_user_question_after_multi_select(
        self,
        *,
        user_id: int,
        state: SessionState | None,
        final_question: bool,
    ) -> tuple[bool, str]:
        transport, terminal_id, workdir, err = await self._resolve_user_question_transport_context(user_id=user_id, state=state)
        if err is not None or transport is None or terminal_id is None or workdir is None:
            return False, err or USER_QUESTION_TUI_FALLBACK_ERROR
        return await transport.advance_after_multi_select(
            terminal_key=terminal_id,
            workdir=workdir,
            final_question=final_question,
        )

    async def _send_interactive_text(
        self,
        *,
        user_id: int,
        text: str,
        terminal_id: str | None = None,
        workdir: str | None = None,
    ) -> tuple[bool, str]:
        session = await self._session_service.get(user_id)
        resolved_terminal_id = terminal_id
        resolved_workdir = workdir
        if resolved_terminal_id is None or resolved_workdir is None:
            if session is None or not self._capabilities_resolver(session.provider).interactive_input:
                return False, "当前没有 Claude 会话"
            if not session.terminal_mode or not session.terminal_id:
                return False, "当前没有可用的 Claude 持久终端"
            resolved_terminal_id = session.terminal_id
            resolved_workdir = session.workdir
        return await self._terminal_runtime.send_interactive_input(
            terminal_key=resolved_terminal_id,
            workdir=resolved_workdir,
            text=text,
        )
