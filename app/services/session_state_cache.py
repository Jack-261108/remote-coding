from __future__ import annotations

import logging
from collections import OrderedDict

from app.domain.hook_models import validate_session_id
from app.domain.session_models import (
    SessionPhase,
    SessionState,
    is_claude_session_id,
)
from app.services.session_state_repository import SessionStateRepository

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SIZE = 512


class SessionStateCache:
    """In-memory LRU cache for SessionState objects with load-on-miss from repository.

    Single responsibility: store and retrieve SessionState objects in memory,
    delegating to SessionStateRepository for persistence on miss.
    Entries are evicted in least-recently-used order when the cache exceeds maxsize.
    """

    def __init__(self, repository: SessionStateRepository, maxsize: int = _DEFAULT_MAX_SIZE) -> None:
        self._repository = repository
        self._maxsize = maxsize
        self._states: OrderedDict[str, SessionState] = OrderedDict()

    def _touch(self, session_id: str) -> None:
        """Move session_id to the end (most recently used)."""
        self._states.move_to_end(session_id)

    def _evict_if_needed(self) -> None:
        """Evict least recently used entries until the cache is within maxsize."""
        while len(self._states) > self._maxsize:
            evicted_id, _ = self._states.popitem(last=False)
            logger.debug("Evicted session %s from cache (LRU)", evicted_id)

    def get(self, session_id: str) -> SessionState | None:
        """Retrieve a cached SessionState by session_id.

        Checks the in-memory cache first, then falls back to the repository.
        Returns None if not found in either location.
        """
        session_id = validate_session_id(session_id)
        state = self._states.get(session_id)
        if state is not None:
            self._touch(session_id)
            return state
        loaded = self._repository.load(session_id)
        if loaded is None:
            return None
        self._states[session_id] = loaded
        self._evict_if_needed()
        return loaded

    def get_or_create(
        self,
        *,
        session_id: str,
        user_id: int | None = None,
        provider: str = "claude_code",
        workdir: str = ".",
        terminal_id: str | None = None,
        claude_session_id: str | None = None,
    ) -> SessionState:
        """Get a cached state, load from repository, or create a new one.

        If the state is already cached, updates its mutable fields and returns it.
        If found in the repository, hydrates, caches, and returns it.
        Otherwise, creates a fresh SessionState with the provided parameters.
        """
        session_id = validate_session_id(session_id)
        resolved_claude_session_id = claude_session_id or (session_id if is_claude_session_id(session_id) else None)

        state = self._states.get(session_id)
        if state is not None:
            self._touch(session_id)
            if user_id is not None:
                state.user_id = user_id
            state.provider = provider
            if workdir and workdir != ".":
                state.workdir = workdir
            if terminal_id is not None:
                state.terminal_id = terminal_id
            state.claude_session_id = resolved_claude_session_id or state.claude_session_id or state.session_id
            return state

        loaded = self._repository.load(session_id)
        if loaded is not None:
            state = self._hydrate_and_merge(loaded)
            if user_id is not None:
                state.user_id = user_id
            state.provider = provider
            if workdir and workdir != ".":
                state.workdir = workdir
            if terminal_id is not None:
                state.terminal_id = terminal_id
            state.claude_session_id = resolved_claude_session_id or state.claude_session_id or state.session_id
        else:
            state = SessionState(
                session_id=session_id,
                user_id=user_id,
                provider=provider,
                workdir=workdir,
                terminal_id=terminal_id,
                claude_session_id=resolved_claude_session_id or session_id,
            )
            checkpoint = self._repository.load_checkpoint(session_id)
            state.checkpoint = checkpoint
            state.turns = self._repository.load_conversation(session_id)
            if state.turns:
                current = state.turns[-1]
                if not current.is_complete:
                    state.current_turn_id = current.turn_id
                    state.phase = SessionPhase.PROCESSING
                else:
                    state.last_reply = current.text.strip() or None
                    state.last_reply_role = current.role

        state.history_loaded = _has_loaded_history(state)
        self._states[session_id] = state
        self._evict_if_needed()
        return state

    def put(self, state: SessionState) -> None:
        """Store a SessionState in the cache, overwriting any existing entry."""
        self._states[state.session_id] = state
        self._touch(state.session_id)
        self._evict_if_needed()

    def remove(self, session_id: str) -> bool:
        """Remove a SessionState from the cache.

        Returns:
            True if the state was removed, False if it wasn't in the cache.
        """
        session_id = validate_session_id(session_id)
        if session_id in self._states:
            del self._states[session_id]
            return True
        return False

    def values(self) -> list[SessionState]:
        """Return all currently cached session states (snapshot)."""
        return list(self._states.values())

    def hydrate_and_cache(self, state: SessionState) -> SessionState:
        """Hydrate a SessionState from persistence and cache it.

        If the state is already cached, merges new fields from the persisted state
        into the cached version. Otherwise, loads checkpoint and conversation data
        from the repository and caches the result.
        """
        return self._hydrate_and_merge(state)

    def _hydrate_and_merge(self, state: SessionState) -> SessionState:
        """Internal hydration logic: merge persisted data into cache.

        Handles the case where a state is already cached (merges fields from the
        persisted version) and the case where it's new (loads checkpoint + turns).
        """
        loaded_checkpoint = self._repository.load_checkpoint(state.session_id)
        loaded_turns = state.turns or self._repository.load_conversation(state.session_id)

        cached = self._states.get(state.session_id)
        if cached is not None:
            self._touch(state.session_id)
            if not cached.turns and loaded_turns:
                cached.turns = loaded_turns
            if not cached.tool_calls and state.tool_calls:
                cached.tool_calls = state.tool_calls
            if cached.pending_permission is None and state.pending_permission is not None:
                cached.pending_permission = state.pending_permission
            if cached.user_id is None and state.user_id is not None:
                cached.user_id = state.user_id
            if cached.terminal_id is None and state.terminal_id is not None:
                cached.terminal_id = state.terminal_id
            if cached.claude_session_id is None and state.claude_session_id is not None:
                cached.claude_session_id = state.claude_session_id
            if cached.checkpoint.last_offset == 0 and loaded_checkpoint.last_offset != 0:
                cached.checkpoint = loaded_checkpoint
            cached.history_loaded = _has_loaded_history(cached)
            return cached

        state.checkpoint = loaded_checkpoint
        if not state.turns:
            state.turns = loaded_turns
        state.history_loaded = _has_loaded_history(state)
        self._states[state.session_id] = state
        self._evict_if_needed()
        return state


def _has_loaded_history(state: SessionState) -> bool:
    """Check if a state has any loaded history content."""
    return bool(state.turns or state.tool_calls or state.pending_permission is not None)
