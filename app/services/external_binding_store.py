from __future__ import annotations

import json
import logging
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.domain.external_session_models import ExternalBinding, ExternalTerminalTarget
from app.domain.external_terminal_models import TerminalKind

logger = logging.getLogger(__name__)


def _normalize_to_utc(value: datetime) -> datetime:
    """Return ``value`` as a timezone-aware UTC datetime.

    Naive datetimes are assumed to represent UTC and have ``timezone.utc``
    attached. Aware datetimes in non-UTC timezones are converted to UTC via
    ``astimezone``. This guarantees all loaded timestamps participate safely
    in idle-age arithmetic (`utc_now() - last_activity_at`) without raising
    naive/aware comparison errors.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _terminal_target_to_dict(target: ExternalTerminalTarget) -> dict[str, object]:
    data: dict[str, object] = {
        "kind": target.kind,
        "terminal_id": target.terminal_id,
        "paired_tty": target.paired_tty,
        "paired_at": target.paired_at.isoformat(),
        "binding_id": target.binding_id,
    }
    if target.name is not None:
        data["name"] = target.name
    if target.cwd is not None:
        data["cwd"] = target.cwd
    return data


def _terminal_target_from_dict(raw: object) -> ExternalTerminalTarget | None:
    """Reconstruct a terminal target from persisted JSON.

    Returns ``None`` for missing/malformed data so a corrupt target entry
    degrades gracefully to "not paired" rather than failing binding load. A
    missing ``kind`` means the entry predates kitty support and can only have
    been a Ghostty surface.
    """
    if not isinstance(raw, dict):
        return None
    terminal_id = raw.get("terminal_id")
    paired_tty = raw.get("paired_tty")
    paired_at_raw = raw.get("paired_at")
    binding_id = raw.get("binding_id")
    if not (
        isinstance(terminal_id, str) and isinstance(paired_tty, str) and isinstance(binding_id, str) and isinstance(paired_at_raw, str)
    ):
        return None
    raw_kind = raw.get("kind")
    kind: TerminalKind = "ghostty"
    if raw_kind is not None:
        if raw_kind not in ("ghostty", "kitty"):
            return None
        kind = raw_kind
    try:
        paired_at = _normalize_to_utc(datetime.fromisoformat(paired_at_raw))
    except ValueError:
        return None
    name = raw.get("name")
    cwd = raw.get("cwd")
    return ExternalTerminalTarget(
        kind=kind,
        terminal_id=terminal_id,
        paired_tty=paired_tty,
        paired_at=paired_at,
        binding_id=binding_id,
        name=name if isinstance(name, str) else None,
        cwd=cwd if isinstance(cwd, str) else None,
    )


class ExternalBindingStore:
    """Persists external session bindings as JSON for restart survival."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._file_path = data_dir / "external_bindings.json"
        self._needs_migration_persist = False
        self._bindings: dict[str, ExternalBinding] = self.load_all()
        if self._needs_migration_persist:
            self._persist()
        # Per-session monotonic timestamps of the last on-disk persist driven
        # by ``touch_activity``. Used to throttle writes to at most once per
        # ``persist_min_interval_sec`` per session. Missing entry means the
        # next ``touch_activity`` call SHALL persist immediately.
        self._last_persist_at: dict[str, float] = {}

    def save_binding(self, binding: ExternalBinding) -> None:
        self._bindings[binding.session_id] = binding
        # Drop any stale throttle entry so the next ``touch_activity`` for a
        # (re-)bound session persists immediately. We pop rather than set-to-now
        # to ensure the first touch after a re-bind always hits disk.
        self._last_persist_at.pop(binding.session_id, None)
        self._persist()

    def remove_binding(self, session_id: str) -> None:
        self._bindings.pop(session_id, None)
        self._last_persist_at.pop(session_id, None)
        self._persist()

    def get_binding(self, session_id: str) -> ExternalBinding | None:
        return self._bindings.get(session_id)

    def get_bindings_for_user(self, user_id: int) -> list[ExternalBinding]:
        return [b for b in self._bindings.values() if b.user_id == user_id and b.ended_at is None]

    def mark_ended(self, session_id: str, ended_at: datetime) -> bool:
        binding = self._bindings.get(session_id)
        if binding is None:
            return False
        if binding.ended_at is not None:
            return False
        binding.ended_at = ended_at
        self._persist()
        return True

    def set_reply_cursor(self, session_id: str, turn_id: str | None) -> bool:
        binding = self._bindings.get(session_id)
        if binding is None:
            return False
        if binding.reply_cursor_initialized and binding.last_pushed_reply_turn_id == turn_id:
            return False
        binding.last_pushed_reply_turn_id = turn_id
        binding.reply_cursor_initialized = True
        self._persist()
        return True

    def set_title_if_current(
        self,
        session_id: str,
        expected_binding_id: str,
        title: str,
    ) -> bool:
        binding = self._bindings.get(session_id)
        if binding is None or binding.binding_id != expected_binding_id:
            return False
        if binding.title == title:
            return True
        binding.title = title
        self._persist()
        return True

    def set_terminal_target(
        self,
        session_id: str,
        expected_binding_id: str,
        *,
        kind: TerminalKind,
        terminal_id: str,
        paired_tty: str,
        paired_at: datetime,
        name: str | None = None,
        cwd: str | None = None,
    ) -> bool:
        """Persist a terminal pairing (Ghostty surface or kitty window).

        Generation-safe: only applies when the binding still exists and its
        ``binding_id`` matches ``expected_binding_id`` (the binding generation
        observed by the caller). Returns ``True`` when the target was set,
        ``False`` when the binding vanished or was re-bound under a new
        generation — in which case the caller must drop the pairing attempt
        (ABA barrier, same contract as :meth:`set_title_if_current`).

        When the binding's ``tty`` is still unset, it is back-filled from
        ``paired_tty`` so subsequent sends can re-derive the trust anchor even
        if the target is later cleared.
        """
        binding = self._bindings.get(session_id)
        if binding is None or binding.binding_id != expected_binding_id:
            return False
        binding.terminal_target = ExternalTerminalTarget(
            kind=kind,
            terminal_id=terminal_id,
            paired_tty=paired_tty,
            paired_at=paired_at,
            binding_id=expected_binding_id,
            name=name,
            cwd=cwd,
        )
        if not binding.tty and paired_tty:
            binding.tty = paired_tty
        self._persist()
        return True

    def clear_terminal_target(
        self,
        session_id: str,
        expected_binding_id: str,
    ) -> bool:
        """Remove a terminal pairing, generation-safe.

        Clears the target only when ``expected_binding_id`` still matches, so a
        stale failure path cannot wipe a target that belongs to a newer binding
        generation. ``tty`` and all other binding fields are left untouched.
        """
        binding = self._bindings.get(session_id)
        if binding is None or binding.binding_id != expected_binding_id:
            return False
        if binding.terminal_target is None:
            return False
        binding.terminal_target = None
        self._persist()
        return True

    def list_all(self) -> list[ExternalBinding]:
        """Return a snapshot list of all current bindings.

        The returned list is a fresh copy of ``self._bindings.values()`` so
        callers can iterate it safely while other code mutates the store
        (e.g. ``save_binding`` or ``remove_binding``).
        """
        return list(self._bindings.values())

    def touch_activity(
        self,
        session_id: str,
        last_activity_at: datetime,
        *,
        persist_min_interval_sec: int = 60,
        pid: int | None = None,
        tty: str | None = None,
    ) -> None:
        """Update the in-memory ``last_activity_at`` for ``session_id``.

        The in-memory update is always immediate. Persistence to disk is
        throttled per session: if no prior touch has persisted (or the binding
        was just (re-)saved), the call persists right away; otherwise it
        persists only when at least ``persist_min_interval_sec`` seconds have
        elapsed since the previous touch-driven persist for the same session.

        When ``pid`` is supplied and is a positive integer, the binding's stored
        ``pid`` is updated in memory immediately. A ``pid`` of ``None`` or any
        non-positive value leaves the existing stored ``pid`` unchanged (so an
        event that does not carry a usable pid never clobbers a previously
        recorded pid). Persistence of the ``pid`` change rides the same
        ``persist_min_interval_sec`` throttle as ``last_activity_at`` — there is
        no separate immediate-persist path for pid.

        ``tty`` follows the same non-clobbering rule as ``pid``: a non-empty
        value updates the stored ``tty`` in memory, while ``None`` or an empty
        string leaves any previously recorded ``tty`` intact. This lets a
        later hook that carries a TTY back-fill a binding that was bound before
        the TTY was available, without an event that omits the TTY wiping it.

        No-op if ``session_id`` is not present in the store.
        """
        binding = self._bindings.get(session_id)
        if binding is None:
            return

        # Always update in memory immediately so subsequent reads (e.g. the
        # cleanup service's re-read) observe the fresh activity timestamp.
        binding.last_activity_at = last_activity_at

        # Update the stored pid only when the caller supplied a usable value;
        # ``None`` or ``<= 0`` leaves any previously recorded pid intact.
        if pid is not None and pid > 0:
            binding.pid = pid

        # Same non-clobbering rule for tty.
        if tty:
            binding.tty = tty

        now = time.monotonic()
        last_persist = self._last_persist_at.get(session_id)
        if last_persist is None or (now - last_persist) >= persist_min_interval_sec:
            self._persist()
            self._last_persist_at[session_id] = now

    def load_all(self) -> dict[str, ExternalBinding]:
        if not self._file_path.exists():
            return {}
        try:
            data = json.loads(self._file_path.read_text(encoding="utf-8"))
            bindings: dict[str, ExternalBinding] = {}
            for session_id, entry in data.items():
                bound_at = _normalize_to_utc(datetime.fromisoformat(entry["bound_at"]))
                last_activity_raw = entry.get("last_activity_at")
                if last_activity_raw is None:
                    last_activity_at = bound_at
                else:
                    last_activity_at = _normalize_to_utc(datetime.fromisoformat(last_activity_raw))
                ended_at_raw = entry.get("ended_at")
                ended_at = _normalize_to_utc(datetime.fromisoformat(ended_at_raw)) if ended_at_raw is not None else None
                binding_id = entry.get("binding_id")
                if not binding_id:
                    binding_id = uuid4().hex
                    self._needs_migration_persist = True
                # Pre-kitty files stored the pairing under ``ghostty_target``.
                # Rewrite them into the new key/format on the next persist.
                if "terminal_target" not in entry and entry.get("ghostty_target") is not None:
                    self._needs_migration_persist = True
                bindings[session_id] = ExternalBinding(
                    session_id=session_id,
                    user_id=entry["user_id"],
                    cwd=entry["cwd"],
                    bound_at=bound_at,
                    jsonl_path=entry.get("jsonl_path"),
                    binding_id=binding_id,
                    pid=entry.get("pid"),
                    title=entry.get("title"),
                    ended_at=ended_at,
                    last_pushed_reply_turn_id=entry.get("last_pushed_reply_turn_id"),
                    reply_cursor_initialized=bool(entry.get("reply_cursor_initialized", False)),
                    last_activity_at_init=last_activity_at,
                    tty=entry.get("tty"),
                    terminal_target=_terminal_target_from_dict(
                        entry.get("terminal_target") if "terminal_target" in entry else entry.get("ghostty_target")
                    ),
                )
            return bindings
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
            logger.error("Failed to load external bindings from %s: %s", self._file_path, exc)
            return {}

    def flush(self) -> None:
        """Force persist all bindings to disk, bypassing throttle.

        This should be called during graceful shutdown to ensure no
        in-memory updates are lost.
        """
        self._persist()
        # Clear throttle state so next touch after restart persists immediately
        self._last_persist_at.clear()

    def _persist(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        data: dict[str, dict] = {}
        for session_id, binding in self._bindings.items():
            data[session_id] = {
                "binding_id": binding.binding_id,
                "user_id": binding.user_id,
                "cwd": binding.cwd,
                "bound_at": binding.bound_at.isoformat(),
                "last_activity_at": binding.last_activity_at.isoformat(),
                "jsonl_path": binding.jsonl_path,
                "pid": binding.pid,
                "title": binding.title,
                "ended_at": binding.ended_at.isoformat() if binding.ended_at is not None else None,
                "last_pushed_reply_turn_id": binding.last_pushed_reply_turn_id,
                "reply_cursor_initialized": binding.reply_cursor_initialized,
                "tty": binding.tty,
                "terminal_target": (_terminal_target_to_dict(binding.terminal_target) if binding.terminal_target is not None else None),
            }
        # Atomic write: write to temp file then rename to avoid corruption
        try:
            fd, tmp_path = tempfile.mkstemp(dir=str(self._data_dir), suffix=".tmp", prefix="external_bindings_")
            try:
                with open(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                Path(tmp_path).replace(self._file_path)
            except BaseException:
                Path(tmp_path).unlink(missing_ok=True)
                raise
        except OSError as exc:
            logger.error("Failed to persist external bindings: %s", exc)
