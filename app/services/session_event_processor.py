"""Session event processor: event → state transitions."""

from __future__ import annotations

from app.domain.hook_models import HookEvent, validate_session_id
from app.domain.session_models import (
    ConversationTurn,
    PendingPermission,
    SessionEvent,
    SessionEventType,
    SessionPhase,
    SessionState,
    SubagentState,
    SubagentToolCall,
    ToolCallRecord,
    ToolStatus,
)
from app.services.session_notifier import SessionNotifier
from app.services.session_state_cache import SessionStateCache
from app.services.session_state_repository import SessionStateRepository


class SessionEventProcessor:
    """Extracted from SessionEventProcessingMixin.

    Takes an event, resolves the state from cache, applies transitions,
    persists via repository, and publishes via notifier.
    """

    def __init__(
        self,
        cache: SessionStateCache,
        repository: SessionStateRepository,
        notifier: SessionNotifier,
    ) -> None:
        self._cache = cache
        self._repository = repository
        self._notifier = notifier

    def _persist(self, state: SessionState, *, publish: bool = True) -> None:
        from app.services.session_store import persist_session_state

        persist_session_state(state, self._cache, self._repository, self._notifier, publish=publish)

    def process(self, event: SessionEvent) -> SessionState:
        event.session_id = validate_session_id(event.session_id)
        state = self._cache.get(event.session_id)
        if state is None:
            state = self._cache.get_or_create(session_id=event.session_id)
        was_ended = state.phase == SessionPhase.ENDED
        state.last_activity = event.at
        payload = event.payload_dict()
        if was_ended and event.type in {
            SessionEventType.SESSION_STARTED,
            SessionEventType.TURN_STARTED,
            SessionEventType.PARSER_UPDATED,
            SessionEventType.TURN_COMPLETED,
            SessionEventType.CLEAR_DETECTED,
        }:
            self._keep_ended_absorbed(state, event)
            self._persist(state)
            return state

        if event.type == SessionEventType.SESSION_STARTED:
            state.phase = SessionPhase.PROCESSING
            state.interrupted = False
        elif event.type == SessionEventType.TURN_STARTED:
            raw_turn_id = payload.get("turn_id")
            if raw_turn_id is None:
                if was_ended:
                    self._keep_ended_absorbed(state, event)
                self._persist(state)
                return state
            turn = ConversationTurn(turn_id=str(raw_turn_id), role=str(payload.get("role", "assistant")))
            state.turns.append(turn)
            state.current_turn_id = turn.turn_id
            state.phase = SessionPhase.PROCESSING
            state.interrupted = False
        elif event.type == SessionEventType.PARSER_UPDATED:
            parser_turn = state.current_turn()
            if parser_turn is not None:
                parser_turn.text = str(payload.get("text", parser_turn.text))
                parser_turn.is_complete = bool(payload.get("is_complete", parser_turn.is_complete))
                state.last_reply = parser_turn.text.strip() or None
                state.last_reply_role = parser_turn.role
                state.last_tool_name = None
        elif event.type == SessionEventType.TURN_COMPLETED:
            completed_turn = state.current_turn()
            if completed_turn is not None:
                completed_turn.is_complete = True
                completed_turn.ended_at = event.at
                state.last_reply = completed_turn.text.strip() or None
                state.last_reply_role = completed_turn.role
                state.last_tool_name = None
            else:
                pass
            state.phase = SessionPhase.WAITING_FOR_INPUT
            state.current_turn_id = None
        elif event.type == SessionEventType.SESSION_ENDED:
            state.phase = SessionPhase.ENDED
            state.interrupted = self._interrupt_session_tools(state, event.at) or state.interrupted
        elif event.type == SessionEventType.HOOK_RECEIVED:
            self._process_hook_event(state, event)
        elif event.type in {SessionEventType.FILE_SYNCED, SessionEventType.HISTORY_LOADED}:
            self._process_file_synced(state, event)
        elif event.type == SessionEventType.CLEAR_DETECTED:
            self._clear_state(state)
        elif event.type == SessionEventType.INTERRUPT_DETECTED:
            self._process_interrupt_detected(state, event)
        elif event.type == SessionEventType.PERMISSION_APPROVED:
            self._process_permission_decision(state, event, approved=True)
        elif event.type == SessionEventType.PERMISSION_DENIED:
            self._process_permission_decision(state, event, approved=False)
        elif event.type == SessionEventType.PERMISSION_RESPONSE_FAILED:
            self._process_permission_response_failed(state, event)

        if state.phase == SessionPhase.ENDED:
            self._keep_ended_absorbed(state, event)
        self._persist(state)
        return state

    def _process_hook_event(self, state: SessionState, event: SessionEvent) -> None:
        payload = event.payload_dict()
        raw = payload.get("hook") if isinstance(payload.get("hook"), dict) else payload
        if not isinstance(raw, dict):
            return
        hook = HookEvent.from_dict(raw)
        state.workdir = hook.cwd or state.workdir
        state.claude_session_id = hook.session_id or state.claude_session_id or state.session_id
        is_session_end = hook.event == "SessionEnd" or hook.status == "ended"
        if state.phase == SessionPhase.ENDED:
            if is_session_end:
                state.interrupted = self._interrupt_session_tools(state, event.at) or state.interrupted
            return
        previous_interrupted = state.interrupted
        state.interrupted = False

        if hook.event == "PreToolUse" and hook.tool_use_id:
            existing = state.tool_calls.get(hook.tool_use_id)
            state.tool_calls[hook.tool_use_id] = ToolCallRecord(
                tool_use_id=hook.tool_use_id,
                name=hook.tool or (existing.name if existing else "Tool"),
                input=hook.tool_input or (existing.input if existing else {}),
                status=existing.status if existing is not None else ToolStatus.RUNNING,
                result=existing.result if existing is not None else None,
                structured_result=existing.structured_result if existing is not None else None,
                subagent_tools=existing.subagent_tools if existing is not None else [],
                started_at=existing.started_at if existing is not None else event.at,
                completed_at=existing.completed_at if existing is not None else None,
            )
            if state.subagent_state.has_active_subagent and not state.tool_calls[hook.tool_use_id].is_subagent_container:
                current_task = state.subagent_state.current_task()
                if current_task is not None:
                    tool = SubagentToolCall(
                        tool_use_id=hook.tool_use_id,
                        name=hook.tool or "Tool",
                        input=hook.tool_input or {},
                        status=ToolStatus.RUNNING,
                        started_at=event.at,
                    )
                    state.subagent_state.add_subagent_tool(current_task.task_tool_id, tool)
                    state.tool_calls[current_task.task_tool_id].subagent_tools = current_task.subagent_tools
            elif state.tool_calls[hook.tool_use_id].is_subagent_container:
                description = hook.tool_input.get("description") if isinstance(hook.tool_input, dict) else None
                state.subagent_state.start_task(
                    task_tool_id=hook.tool_use_id, description=str(description) if description is not None else None
                )
            state.phase = SessionPhase.PROCESSING
        elif hook.event == "PermissionRequest" and hook.tool_use_id:
            existing = state.tool_calls.get(hook.tool_use_id)
            state.tool_calls[hook.tool_use_id] = ToolCallRecord(
                tool_use_id=hook.tool_use_id,
                name=hook.tool or (existing.name if existing else "Tool"),
                input=hook.tool_input or (existing.input if existing else {}),
                status=ToolStatus.WAITING_FOR_APPROVAL,
                result=existing.result if existing is not None else None,
                structured_result=existing.structured_result if existing is not None else None,
                started_at=existing.started_at if existing is not None else event.at,
                completed_at=existing.completed_at if existing is not None else None,
            )
            state.pending_permission = PendingPermission(
                tool_use_id=hook.tool_use_id,
                tool_name=hook.tool or "Tool",
                tool_input=hook.tool_input,
                received_at=event.at,
            )
            state.phase = SessionPhase.WAITING_FOR_APPROVAL
        elif hook.event in {"PostToolUse", "PostToolUseFailure"} and hook.tool_use_id:
            existing = state.tool_calls.get(hook.tool_use_id)
            if existing is not None:
                existing.status = ToolStatus.SUCCESS if hook.event == "PostToolUse" else ToolStatus.ERROR
                existing.completed_at = event.at
            if state.pending_permission and state.pending_permission.tool_use_id == hook.tool_use_id:
                state.pending_permission = None
            if existing is not None and existing.is_subagent_container:
                state.subagent_state.stop_task(task_tool_id=hook.tool_use_id)
            elif state.subagent_state.has_active_subagent:
                current_task = state.subagent_state.current_task()
                if current_task is not None:
                    state.subagent_state.update_subagent_tool_status(
                        current_task.task_tool_id, hook.tool_use_id, ToolStatus.SUCCESS if hook.event == "PostToolUse" else ToolStatus.ERROR
                    )
                    container = state.tool_calls.get(current_task.task_tool_id)
                    if container is not None:
                        container.subagent_tools = current_task.subagent_tools
            self._move_to_next_phase(state, default=SessionPhase.PROCESSING)
        elif hook.event == "PreCompact":
            state.phase = SessionPhase.COMPACTING
        elif hook.event == "StopFailure":
            self._interrupt_session_tools(state, event.at)
            state.interrupted = True
            state.phase = SessionPhase.WAITING_FOR_INPUT
        elif is_session_end:
            state.interrupted = self._interrupt_session_tools(state, event.at) or previous_interrupted
            state.phase = SessionPhase.ENDED
        elif hook.event == "SubagentStop":
            if state.subagent_state.has_active_subagent:
                current_task = state.subagent_state.current_task()
                if current_task is not None:
                    state.subagent_state.stop_task(task_tool_id=current_task.task_tool_id)
            self._move_to_next_phase(state, default=SessionPhase.PROCESSING)
        elif hook.event == "Stop" or hook.status == "waiting_for_input":
            state.interrupted = self._interrupt_session_tools(state, event.at)
            state.phase = SessionPhase.WAITING_FOR_INPUT
        elif hook.status in {"running_tool", "processing", "starting"}:
            state.phase = SessionPhase.PROCESSING

    def _process_file_synced(self, state: SessionState, event: SessionEvent) -> None:
        payload = event.payload_dict()
        state.workdir = str(payload.get("cwd", state.workdir))
        state.claude_session_id = str(payload.get("claude_session_id") or state.claude_session_id or state.session_id)
        try:
            last_offset = int(payload["last_offset"]) if payload.get("last_offset") is not None else None
        except (ValueError, TypeError):
            last_offset = None
        reset_detected = bool(payload.get("reset_detected", False))
        turns_payload = payload.get("turns", [])
        parsed_turns = [item if isinstance(item, ConversationTurn) else ConversationTurn.from_dict(item) for item in turns_payload]
        tool_calls_payload = payload.get("tool_calls", {})
        parsed_tool_calls: dict[str, ToolCallRecord] = {}
        if isinstance(tool_calls_payload, dict):
            for key, value in tool_calls_payload.items():
                parsed_tool_calls[str(key)] = value if isinstance(value, ToolCallRecord) else ToolCallRecord.from_dict(value)
        self._preserve_hook_only_runtime_state(state, parsed_tool_calls)

        if last_offset is not None and last_offset < state.checkpoint.last_offset and not reset_detected:
            has_newer_turns = len(parsed_turns) > len(state.turns)
            has_more_tool_calls = len(parsed_tool_calls) > len(state.tool_calls)
            if not has_newer_turns and not has_more_tool_calls:
                return

        if payload.get("clear_detected"):
            state.turns = parsed_turns
            state.tool_calls = parsed_tool_calls
            state.pending_permission = None
        elif payload.get("turns") is not None:
            state.turns = parsed_turns
            state.tool_calls = parsed_tool_calls

        for task_tool_id, container in state.tool_calls.items():
            if container.subagent_tools:
                state.subagent_state.populate_from_container(task_tool_id, container.subagent_tools)

        if state.subagent_state.has_active_subagent:
            for task_tool_id, task in state.subagent_state.active_tasks.items():
                task_container = state.tool_calls.get(task_tool_id)
                if task_container is not None and not task_container.subagent_tools:
                    task_container.subagent_tools = task.subagent_tools

        state.current_turn_id = None
        state.summary = str(payload["summary"]) if payload.get("summary") is not None else state.summary
        state.last_reply = str(payload["last_reply"]) if payload.get("last_reply") is not None else state.last_reply
        state.last_reply_role = str(payload["last_reply_role"]) if payload.get("last_reply_role") is not None else state.last_reply_role
        state.last_tool_name = str(payload["last_tool_name"]) if payload.get("last_tool_name") is not None else state.last_tool_name
        state.history_loaded = True
        state.clear_detected = bool(payload.get("clear_detected", False))
        if last_offset is not None:
            state.checkpoint.last_offset = last_offset
        state.checkpoint.clear_pending = state.clear_detected
        state.checkpoint.last_summary = state.summary or ""
        state.checkpoint.seen_tool_ids = list(state.tool_calls.keys())
        state.checkpoint.completed_tool_ids = [
            tool_id
            for tool_id, tool in state.tool_calls.items()
            if tool.status in {ToolStatus.SUCCESS, ToolStatus.ERROR, ToolStatus.INTERRUPTED}
        ]
        state.checkpoint.tool_id_to_name = {tool_id: tool.name for tool_id, tool in state.tool_calls.items()}

        self._move_to_next_phase(state, default=SessionPhase.IDLE)

    def _process_interrupt_detected(self, state: SessionState, event: SessionEvent) -> None:
        self._interrupt_session_tools(state, event.at)
        state.interrupted = True
        self._move_to_next_phase(state, default=SessionPhase.WAITING_FOR_INPUT)

    def _preserve_hook_only_runtime_state(self, state: SessionState, parsed_tool_calls: dict[str, ToolCallRecord]) -> None:
        for tool_id, existing in state.tool_calls.items():
            if existing.status != ToolStatus.WAITING_FOR_APPROVAL:
                continue
            candidate = parsed_tool_calls.get(tool_id)
            if candidate is None:
                parsed_tool_calls[tool_id] = ToolCallRecord.from_dict(existing.to_dict())
                continue
            if candidate.status == ToolStatus.RUNNING:
                candidate.status = ToolStatus.WAITING_FOR_APPROVAL
                if not candidate.input and existing.input:
                    candidate.input = dict(existing.input)

    def _process_permission_decision(self, state: SessionState, event: SessionEvent, *, approved: bool) -> None:
        if state.phase == SessionPhase.ENDED:
            return
        payload = event.payload_dict()
        tool_use_id = str(payload.get("tool_use_id", ""))
        if not tool_use_id:
            return
        tool = state.tool_calls.get(tool_use_id)
        if tool is not None:
            tool.status = ToolStatus.RUNNING if approved else ToolStatus.ERROR
            if not approved:
                tool.completed_at = event.at
        if state.pending_permission and state.pending_permission.tool_use_id == tool_use_id:
            state.pending_permission = None
        state.interrupted = not approved
        self._move_to_next_phase(state, default=SessionPhase.PROCESSING if approved else SessionPhase.IDLE)

    def _process_permission_response_failed(self, state: SessionState, event: SessionEvent) -> None:
        payload = event.payload_dict()
        tool_use_id = str(payload.get("tool_use_id", ""))
        if not tool_use_id:
            state.interrupted = self._interrupt_session_tools(state, event.at)
            self._move_to_next_phase(state, default=SessionPhase.WAITING_FOR_INPUT)
            return
        tool = state.tool_calls.get(tool_use_id)
        if tool is not None and tool.status in {ToolStatus.RUNNING, ToolStatus.WAITING_FOR_APPROVAL}:
            tool.status = ToolStatus.INTERRUPTED
            tool.completed_at = tool.completed_at or event.at
            state.interrupted = True
        if state.pending_permission and state.pending_permission.tool_use_id == tool_use_id:
            state.pending_permission = None
        self._move_to_next_phase(state, default=SessionPhase.WAITING_FOR_INPUT)

    def _keep_ended_absorbed(self, state: SessionState, event: SessionEvent) -> None:
        state.phase = SessionPhase.ENDED
        if self._interrupt_session_tools(state, event.at):
            state.interrupted = True

    def _move_to_next_phase(self, state: SessionState, *, default: SessionPhase) -> None:
        if state.phase == SessionPhase.ENDED:
            state.pending_permission = None
            return
        next_pending = None
        has_running_tool = False
        for tool in state.tool_calls.values():
            if tool.status == ToolStatus.WAITING_FOR_APPROVAL:
                next_pending = tool
                break
            if tool.status == ToolStatus.RUNNING:
                has_running_tool = True
        if next_pending is not None:
            state.pending_permission = PendingPermission(
                tool_use_id=next_pending.tool_use_id,
                tool_name=next_pending.name,
                tool_input=next_pending.input,
            )
            state.phase = SessionPhase.WAITING_FOR_APPROVAL
            return
        state.pending_permission = None
        if has_running_tool:
            state.phase = SessionPhase.PROCESSING
            return
        if default == SessionPhase.ENDED:
            state.phase = SessionPhase.ENDED
            return
        if state.turns and state.turns[-1].role == "assistant" and state.turns[-1].is_complete:
            state.phase = SessionPhase.WAITING_FOR_INPUT
            return
        state.phase = default

    def _clear_state(self, state: SessionState) -> None:
        state.turns = []
        state.tool_calls = {}
        state.pending_permission = None
        state.current_turn_id = None
        state.summary = None
        state.last_reply = None
        state.last_reply_role = None
        state.last_tool_name = None
        state.subagent_state = SubagentState()
        state.history_loaded = True
        state.clear_detected = True
        state.interrupted = False
        state.checkpoint.clear_pending = True
        state.checkpoint.seen_tool_ids = []
        state.checkpoint.completed_tool_ids = []
        state.checkpoint.tool_id_to_name = {}
        state.phase = SessionPhase.WAITING_FOR_INPUT

    def _interrupt_session_tools(self, state: SessionState, at) -> bool:
        interrupted = False
        for tool in state.tool_calls.values():
            if tool.status in {ToolStatus.RUNNING, ToolStatus.WAITING_FOR_APPROVAL}:
                tool.status = ToolStatus.INTERRUPTED
                tool.completed_at = tool.completed_at or at
                interrupted = True
            for subagent_tool in tool.subagent_tools:
                if subagent_tool.status in {ToolStatus.RUNNING, ToolStatus.WAITING_FOR_APPROVAL}:
                    subagent_tool.status = ToolStatus.INTERRUPTED
                    subagent_tool.completed_at = subagent_tool.completed_at or at
                    interrupted = True
        for task in state.subagent_state.active_tasks.values():
            for subagent_tool in task.subagent_tools:
                if subagent_tool.status in {ToolStatus.RUNNING, ToolStatus.WAITING_FOR_APPROVAL}:
                    subagent_tool.status = ToolStatus.INTERRUPTED
                    subagent_tool.completed_at = subagent_tool.completed_at or at
                    interrupted = True
        state.pending_permission = None
        return interrupted
