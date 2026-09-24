from __future__ import annotations

import asyncio
import copy

from app.adapters.storage.file_session_store import FileSessionStore
from app.domain.models import SessionContext


class FileSessionContextStore:
    def __init__(self, file_store: FileSessionStore) -> None:
        self._file_store = file_store
        self._lock = asyncio.Lock()
        self._list_cache: list[SessionContext] | None = None
        self._claude_session_index: dict[str, SessionContext] = {}
        self._index_loaded = False  # Track whether index has been loaded from disk

    async def _reload_cache_locked(self) -> list[SessionContext]:
        """(Re)load every context from disk off the event loop and refresh caches.

        Caller must hold ``self._lock``. Corrupt files are skipped by
        ``list_session_contexts`` and therefore absent from the cache, matching
        ``load_session_context``'s tolerant single-file read.
        """
        contexts = await asyncio.to_thread(self._file_store.list_session_contexts)
        self._list_cache = contexts
        self._claude_session_index = {ctx.claude_session_id: ctx for ctx in contexts if ctx.claude_session_id}
        self._index_loaded = True
        return contexts

    async def get(self, user_id: int) -> SessionContext | None:
        async with self._lock:
            # The cache, once loaded, mirrors disk (save updates it in place,
            # delete drops it), so the per-message router filter + session-guard
            # middleware double get() reads memory instead of blocking the
            # event loop on JSON file loads.
            cache = self._list_cache
            if cache is None:
                cache = await self._reload_cache_locked()
            for ctx in cache:
                if ctx.user_id == user_id:
                    # Callers mutate the returned context in place and save it
                    # back (SessionService.get_or_create), so hand out a copy —
                    # matching the fresh object the old per-call disk parse
                    # returned — to keep the cache snapshot intact.
                    return copy.deepcopy(ctx)
            return None

    async def list_all(self) -> list[SessionContext]:
        async with self._lock:
            cache = self._list_cache
            if cache is None:
                cache = await self._reload_cache_locked()
            return [copy.deepcopy(ctx) for ctx in cache]

    async def get_by_claude_session_id(self, claude_session_id: str) -> SessionContext | None:
        """O(1) index lookup by claude_session_id. Returns None on miss.

        On cold start (index never loaded), triggers a full reload to
        populate the index before looking up.
        """
        async with self._lock:
            if not self._index_loaded:
                # Cold start: load from disk to populate index
                await self._reload_cache_locked()
            ctx = self._claude_session_index.get(claude_session_id)
            return copy.deepcopy(ctx) if ctx is not None else None

    async def save(self, session: SessionContext) -> None:
        async with self._lock:
            # Remove stale index entry if claude_session_id changed or was cleared
            old = self._file_store.load_session_context(session.user_id)
            if old is not None and old.claude_session_id and old.claude_session_id != session.claude_session_id:
                self._claude_session_index.pop(old.claude_session_id, None)

            # Persist
            self._file_store.save_session_context(session)

            # Store an isolated copy in cache to protect against subsequent caller mutations
            cached = copy.deepcopy(session)

            # Update cache if it exists (avoid invalidating the entire cache)
            if self._list_cache is not None:
                # Find and replace the session in the cache
                for i, ctx in enumerate(self._list_cache):
                    if ctx.user_id == session.user_id:
                        self._list_cache[i] = cached
                        break
                else:
                    # Session not in cache, add it
                    self._list_cache.append(cached)

            self._index_loaded = True  # Index is being maintained

            # Update index (last-writer-wins when two contexts share the same id)
            if cached.claude_session_id:
                self._claude_session_index[cached.claude_session_id] = cached

    async def delete(self, user_id: int) -> bool:
        async with self._lock:
            # Load current context to clean up index
            old = self._file_store.load_session_context(user_id)
            if old is None:
                return False

            # Remove from index if it has a claude_session_id
            if old.claude_session_id:
                self._claude_session_index.pop(old.claude_session_id, None)

            # Delete from disk using public method
            context_path = self._file_store.session_context_path(user_id)
            if context_path.exists():
                context_path.unlink()

            # Invalidate cache
            self._list_cache = None
            return True
