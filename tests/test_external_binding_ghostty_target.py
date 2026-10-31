"""Unit tests for ExternalBindingStore Ghostty input target & tty handling.

Covers the new ``tty`` field and the optional ``terminal_target`` nested object
(see docs/specs/2026-08-03-external-ghostty-input-design.md):

  * Persistence round-trip — a binding with tty + terminal_target survives
    ``_persist`` → ``load_all`` with all fields intact, incl. the nested
    ``binding_id`` ABA anchor and UTC-normalized ``paired_at``.
  * Backward compatibility — a pre-feature JSON lacking ``tty``/``terminal_target``
    loads every entry with tty ``None`` and target ``None``.
  * Generation-safe setters — ``set_terminal_target`` / ``clear_terminal_target``
    only apply when ``binding_id`` still matches (ABA barrier), mirroring
    ``set_title_if_current``'s contract.
  * tty back-fill — ``set_terminal_target`` back-fills binding.tty from
    ``paired_tty`` when unset, and ``touch_activity`` follows the same
    non-clobbering rule as ``pid``.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from app.domain.external_session_models import ExternalBinding, ExternalTerminalTarget
from app.domain.hook_models import HookEvent
from app.domain.models import utc_now
from app.services.external_binding_store import ExternalBindingStore
from tests.fakes.external_session import make_binding, make_hook_event

# --- Helpers ----------------------------------------------------------------


def _write_bindings_json(data_dir: Path, payload: dict[str, dict]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "external_bindings.json").write_text(json.dumps(payload), encoding="utf-8")


def _read_bindings_json(data_dir: Path) -> dict[str, dict]:
    return json.loads((data_dir / "external_bindings.json").read_text(encoding="utf-8"))


def _make_binding(
    *,
    session_id: str = "session-target",
    user_id: int = 42,
    binding_id: str | None = None,
    tty: str | None = None,
) -> ExternalBinding:
    return make_binding(
        session_id=session_id,
        user_id=user_id,
        binding_id=binding_id or "binding-gen-1",
        pid=4242,
        tty=tty,
        bound_at=utc_now() - timedelta(hours=1),
    )


def _make_hook(event: str, tty: str | None) -> HookEvent:
    """Build a minimal HookEvent for discovery/binder tests."""
    return make_hook_event(session_id="s1", event=event, status="running", cwd="/p", pid=100, tty=tty)


# --- Round-trip -------------------------------------------------------------


def test_terminal_target_persists_and_reloads_with_all_fields(tmp_path: Path) -> None:
    """tty + terminal_target round-trip through the JSON store unchanged."""
    binding = _make_binding(tty="/dev/ttys005")
    paired_at = utc_now() - timedelta(minutes=5)
    binding.terminal_target = ExternalTerminalTarget(
        kind="ghostty",
        terminal_id="AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE",
        paired_tty="/dev/ttys005",
        paired_at=paired_at,
        binding_id="binding-gen-1",
        name="claude — project",
        cwd="/home/user/project",
    )
    store = ExternalBindingStore(data_dir=tmp_path)
    store.save_binding(binding)

    on_disk = _read_bindings_json(tmp_path)["session-target"]
    assert on_disk["tty"] == "/dev/ttys005"
    target = on_disk["terminal_target"]
    assert target is not None
    assert target["terminal_id"] == "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"
    assert target["paired_tty"] == "/dev/ttys005"
    assert target["binding_id"] == "binding-gen-1"
    assert target["name"] == "claude — project"
    assert target["cwd"] == "/home/user/project"

    reloaded = ExternalBindingStore(data_dir=tmp_path).get_binding("session-target")
    assert reloaded is not None
    assert reloaded.tty == "/dev/ttys005"
    rt = reloaded.terminal_target
    assert rt is not None
    assert rt.terminal_id == "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"
    assert rt.paired_tty == "/dev/ttys005"
    assert rt.binding_id == "binding-gen-1"
    assert rt.paired_at == paired_at
    assert rt.name == "claude — project"
    assert rt.cwd == "/home/user/project"


# --- Backward compatibility -------------------------------------------------


def test_pre_feature_json_without_ghostty_fields_loads_as_none(tmp_path: Path) -> None:
    """A pre-feature file lacking ``tty`` and ``terminal_target`` loads cleanly
    with both fields None — no silent coercion, no raise."""
    bound_at = utc_now() - timedelta(hours=2)
    _write_bindings_json(
        tmp_path,
        {
            "session-a": {
                "user_id": 1,
                "cwd": "/home/user/a",
                "bound_at": bound_at.isoformat(),
                "jsonl_path": None,
            },
        },
    )
    store = ExternalBindingStore(data_dir=tmp_path)
    loaded = store.get_binding("session-a")
    assert loaded is not None
    assert loaded.tty is None
    assert loaded.terminal_target is None


def test_malformed_terminal_target_degrades_to_none(tmp_path: Path) -> None:
    """A corrupt ``terminal_target`` entry degrades to "not paired" rather than
    failing the whole load — graceful degradation."""
    bound_at = utc_now() - timedelta(hours=2)
    _write_bindings_json(
        tmp_path,
        {
            "session-bad": {
                "user_id": 1,
                "cwd": "/home/user/a",
                "bound_at": bound_at.isoformat(),
                "jsonl_path": None,
                "terminal_target": {"terminal_id": "only-this"},  # missing required fields
            },
        },
    )
    store = ExternalBindingStore(data_dir=tmp_path)
    loaded = store.get_binding("session-bad")
    assert loaded is not None
    assert loaded.terminal_target is None


def test_legacy_ghostty_target_key_migrates_on_next_persist(tmp_path: Path) -> None:
    """Pre-kitty files stored the pairing under ``ghostty_target`` with no
    ``kind``; loading must yield a ghostty target and rewrite the file."""
    bound_at = utc_now() - timedelta(hours=2)
    paired_at = bound_at + timedelta(minutes=5)
    _write_bindings_json(
        tmp_path,
        {
            "session-legacy": {
                "user_id": 1,
                "cwd": "/home/user/a",
                "bound_at": bound_at.isoformat(),
                "jsonl_path": None,
                "binding_id": "gen-1",
                "ghostty_target": {
                    "terminal_id": "LEGACY-SURFACE",
                    "paired_tty": "/dev/ttys007",
                    "paired_at": paired_at.isoformat(),
                    "binding_id": "gen-1",
                },
            },
        },
    )

    store = ExternalBindingStore(data_dir=tmp_path)
    loaded = store.get_binding("session-legacy")
    assert loaded is not None and loaded.terminal_target is not None
    assert loaded.terminal_target.kind == "ghostty"
    assert loaded.terminal_target.terminal_id == "LEGACY-SURFACE"

    on_disk = _read_bindings_json(tmp_path)["session-legacy"]
    assert "ghostty_target" not in on_disk, "legacy key must be rewritten on migration"
    assert on_disk["terminal_target"]["kind"] == "ghostty"


def test_unknown_terminal_kind_degrades_to_none(tmp_path: Path) -> None:
    """An unknown kind is not silently coerced to ghostty — fail closed."""
    bound_at = utc_now() - timedelta(hours=2)
    _write_bindings_json(
        tmp_path,
        {
            "session-weird": {
                "user_id": 1,
                "cwd": "/home/user/a",
                "bound_at": bound_at.isoformat(),
                "jsonl_path": None,
                "binding_id": "gen-1",
                "terminal_target": {
                    "kind": "iterm",
                    "terminal_id": "x",
                    "paired_tty": "/dev/ttys007",
                    "paired_at": bound_at.isoformat(),
                    "binding_id": "gen-1",
                },
            },
        },
    )
    store = ExternalBindingStore(data_dir=tmp_path)
    loaded = store.get_binding("session-weird")
    assert loaded is not None
    assert loaded.terminal_target is None


# --- Generation-safe setters -----------------------------------------------


def test_set_terminal_target_generation_safe(tmp_path: Path) -> None:
    """set_terminal_target only applies when binding_id still matches; a stale
    caller (post-unbind/rebind) is refused (ABA barrier)."""
    binding = _make_binding()
    store = ExternalBindingStore(data_dir=tmp_path)
    store.save_binding(binding)

    paired_at = utc_now()
    ok = store.set_terminal_target(
        "session-target",
        "binding-gen-1",
        kind="ghostty",
        terminal_id="uuid-1",
        paired_tty="/dev/ttys005",
        paired_at=paired_at,
    )
    assert ok is True
    got = store.get_binding("session-target")
    assert got is not None
    assert got.terminal_target is not None
    assert got.terminal_target.terminal_id == "uuid-1"

    # Stale generation (e.g. after unbind+rebind) must be refused.
    stale = store.set_terminal_target(
        "session-target",
        "binding-gen-other",
        kind="ghostty",
        terminal_id="uuid-2",
        paired_tty="/dev/ttys006",
        paired_at=paired_at,
    )
    assert stale is False
    got2 = store.get_binding("session-target")
    assert got2 is not None
    assert got2.terminal_target is not None
    assert got2.terminal_target.terminal_id == "uuid-1", "stale set MUST NOT overwrite"


def test_set_terminal_target_missing_binding_is_false(tmp_path: Path) -> None:
    store = ExternalBindingStore(data_dir=tmp_path)
    ok = store.set_terminal_target(
        "nope",
        "binding-gen-1",
        kind="ghostty",
        terminal_id="uuid-1",
        paired_tty="/dev/ttys005",
        paired_at=utc_now(),
    )
    assert ok is False


def test_set_terminal_target_backfills_tty_when_unset(tmp_path: Path) -> None:
    """Setting a target back-fills binding.tty from paired_tty when unset, so
    the trust anchor survives even if later the target is cleared."""
    binding = _make_binding(tty=None)
    store = ExternalBindingStore(data_dir=tmp_path)
    store.save_binding(binding)

    store.set_terminal_target(
        "session-target",
        "binding-gen-1",
        kind="ghostty",
        terminal_id="uuid-1",
        paired_tty="/dev/ttys009",
        paired_at=utc_now(),
    )
    got = store.get_binding("session-target")
    assert got is not None
    assert got.tty == "/dev/ttys009"


def test_set_terminal_target_does_not_overwrite_existing_tty(tmp_path: Path) -> None:
    """An existing tty must not be clobbered by paired_tty (non-clobbering)."""
    binding = _make_binding(tty="/dev/ttysA")
    store = ExternalBindingStore(data_dir=tmp_path)
    store.save_binding(binding)

    store.set_terminal_target(
        "session-target",
        "binding-gen-1",
        kind="ghostty",
        terminal_id="uuid-1",
        paired_tty="/dev/ttysB",
        paired_at=utc_now(),
    )
    got = store.get_binding("session-target")
    assert got is not None
    assert got.tty == "/dev/ttysA", "existing tty MUST NOT be overwritten"


def test_clear_terminal_target_generation_safe(tmp_path: Path) -> None:
    """clear_terminal_target only clears when binding_id matches, and is a no-op
    when no target is set."""
    binding = _make_binding()
    store = ExternalBindingStore(data_dir=tmp_path)
    store.save_binding(binding)

    # No target set yet -> clear is a no-op (returns False).
    assert store.clear_terminal_target("session-target", "binding-gen-1") is False

    store.set_terminal_target(
        "session-target",
        "binding-gen-1",
        kind="ghostty",
        terminal_id="uuid-1",
        paired_tty="/dev/ttys005",
        paired_at=utc_now(),
    )

    assert store.clear_terminal_target("session-target", "binding-gen-other") is False, "stale clear MUST NOT wipe newer target"
    got = store.get_binding("session-target")
    assert got is not None
    assert got.terminal_target is not None

    assert store.clear_terminal_target("session-target", "binding-gen-1") is True
    got2 = store.get_binding("session-target")
    assert got2 is not None
    assert got2.terminal_target is None
    # tty survives a target clear (separate field).
    assert got2.tty == "/dev/ttys005"


# --- touch_activity tty -----------------------------------------------------


def test_touch_activity_tty_non_clobbering(tmp_path: Path) -> None:
    """touch_activity(tty=...) follows the same non-clobbering rule as pid: a
    non-empty value updates; None/empty leaves the existing tty intact."""
    binding = _make_binding(tty="/dev/ttysA")
    store = ExternalBindingStore(data_dir=tmp_path)
    store.save_binding(binding)

    def tty_now() -> str | None:
        got = store.get_binding("session-target")
        assert got is not None
        return got.tty

    # Empty tty MUST NOT wipe existing value.
    store.touch_activity("session-target", utc_now(), tty="")
    assert tty_now() == "/dev/ttysA"

    # None tty MUST NOT wipe existing value.
    store.touch_activity("session-target", utc_now(), tty=None)
    assert tty_now() == "/dev/ttysA"

    # Non-empty tty updates.
    store.touch_activity("session-target", utc_now(), tty="/dev/ttysB")
    assert tty_now() == "/dev/ttysB"


def test_touch_activity_tty_backfills_unset_binding(tmp_path: Path) -> None:
    """A hook that carries a TTY back-fills a binding that had no tty yet."""
    binding = _make_binding(tty=None)
    store = ExternalBindingStore(data_dir=tmp_path)
    store.save_binding(binding)

    store.touch_activity("session-target", utc_now(), tty="/dev/ttys007")
    got = store.get_binding("session-target")
    assert got is not None and got.tty == "/dev/ttys007"


# --- discovery / binder tty propagation -------------------------------------


def test_discovery_records_and_updates_tty() -> None:
    """ExternalSessionDiscoveryService records event.tty on first sighting and
    refreshes it on subsequent events (non-clobber for None)."""
    from app.services.external_session_discovery import ExternalSessionDiscoveryService

    discovery = ExternalSessionDiscoveryService()

    discovery.record_event(_make_hook("UserPromptSubmit", "/dev/ttys010"))
    unbound = discovery.get("s1")
    assert unbound is not None and unbound.tty == "/dev/ttys010"

    discovery.record_event(_make_hook("Stop", "/dev/ttys011"))
    after = discovery.get("s1")
    assert after is not None and after.tty == "/dev/ttys011"

    # An event without tty MUST NOT wipe the recorded tty.
    discovery.record_event(_make_hook("Stop", None))
    again = discovery.get("s1")
    assert again is not None and again.tty == "/dev/ttys011"


async def test_binder_propagates_unbound_tty_to_binding(tmp_path: Path) -> None:
    """ExternalSessionBinder.bind() copies the unbound session's tty into the
    new ExternalBinding so the trust anchor is available at pairing time."""
    from app.services.external_binding_store import ExternalBindingStore
    from app.services.external_session_binder import ExternalSessionBinder
    from app.services.external_session_discovery import ExternalSessionDiscoveryService

    discovery = ExternalSessionDiscoveryService()
    discovery.record_event(_make_hook("UserPromptSubmit", "/dev/ttys021"))

    store = ExternalBindingStore(data_dir=tmp_path)
    unbound = discovery.get("s1")
    assert unbound is not None and unbound.tty == "/dev/ttys021"
    binder = ExternalSessionBinder(
        discovery=discovery,
        binding_store=store,
        projects_dir=tmp_path / ".claude" / "projects",
    )
    result = await binder.bind(user_id=42, session_id="s1")
    assert result.success
    binding = store.get_binding("s1")
    assert binding is not None and binding.tty == "/dev/ttys021"
