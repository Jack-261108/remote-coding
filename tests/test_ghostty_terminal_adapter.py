"""Unit tests for GhosttyTerminalAdapter.

Covers the security contract (argv-only payload, fixed script) and the
outcome classification (not-unique, TCC, Ghostty-down, timeout,
indeterminate, os-error) without depending on a real GUI or TCC grant. The
``osascript`` subprocess is faked by monkeypatching
``asyncio.create_subprocess_exec`` in the adapter module.
"""

from __future__ import annotations

import asyncio

import pytest

from app.adapters.process import ghostty_terminal_adapter as gta
from app.adapters.process.ghostty_terminal_adapter import (
    ExternalTerminal,
    GhosttyTerminalAdapter,
    TerminalOutcome,
)
from tests.fakes.subprocess_process import FakeSubprocessProcess

_FakeProc = FakeSubprocessProcess


def _patch_exec(monkeypatch: pytest.MonkeyPatch, factory) -> list[list[object]]:
    calls: list[list[object]] = []

    async def fake(*args: object, **_kwargs: object) -> object:
        calls.append(list(args))
        return factory(list(args))

    monkeypatch.setattr(gta.asyncio, "create_subprocess_exec", fake)
    return calls


@pytest.mark.asyncio
async def test_is_available_requires_darwin_and_osascript(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")
    assert GhosttyTerminalAdapter().is_available()

    monkeypatch.setattr(gta.sys, "platform", "linux")
    assert not GhosttyTerminalAdapter().is_available()

    monkeypatch.setattr(gta.sys, "platform", "darwin")
    monkeypatch.setattr(gta.shutil, "which", lambda _b: None)
    monkeypatch.setattr(gta.os, "access", lambda _path, _mode: False)
    assert not GhosttyTerminalAdapter().is_available()

    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    assert not GhosttyTerminalAdapter(enable_applescript=False).is_available()


@pytest.mark.asyncio
async def test_list_terminals_falls_back_to_system_osascript_when_path_omits_usr_bin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gta.sys, "platform", "darwin")
    monkeypatch.setattr(gta.shutil, "which", lambda _b: None)
    monkeypatch.setattr(gta.os, "access", lambda path, mode: path == gta._SYSTEM_OSASCRIPT and mode == gta.os.X_OK)
    calls = _patch_exec(monkeypatch, lambda _a: _FakeProc(stdout=b"", returncode=0))

    terminals, err = await GhosttyTerminalAdapter().list_terminals()

    assert err is None
    assert terminals == []
    assert calls[0][0] == gta._SYSTEM_OSASCRIPT


@pytest.mark.asyncio
async def test_list_terminals_parses_windows_tabs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")
    rows = "uuid-1\tclaude — project\t/home/u/project\nuuid-2\tshell\t/home/u\n"
    calls = _patch_exec(monkeypatch, lambda _a: _FakeProc(stdout=rows.encode(), returncode=0))
    adapter = GhosttyTerminalAdapter()
    terminals, err = await adapter.list_terminals()
    assert err is None
    assert [t.terminal_id for t in terminals] == ["uuid-1", "uuid-2"]
    assert terminals[0].name == "claude — project"
    assert terminals[0].cwd == "/home/u/project"
    assert terminals[1].cwd == "/home/u"
    # Script source is the fixed _LIST_SCRIPT and argv is empty for listing.
    assert calls[0][0] == "osascript"
    assert calls[0][1] == "-e"
    assert calls[0][2] == gta._LIST_SCRIPT
    assert calls[0][3] == "--"


@pytest.mark.asyncio
async def test_list_terminals_unavailable_returns_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gta.sys, "platform", "linux")
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    terminals, reason = await GhosttyTerminalAdapter().list_terminals()
    assert terminals is None
    assert reason == "non_darwin"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        ("not allowed to send Apple events", TerminalOutcome.TCC_DENIED),
        ("Ghostty got an error: Can't get application", TerminalOutcome.GHOSTTY_NOT_RUNNING),
        ("timed out", TerminalOutcome.TIMEOUT),
        ("some osascript error", TerminalOutcome.OS_ERROR),
    ],
)
async def test_list_terminals_classifies_failures(
    monkeypatch: pytest.MonkeyPatch,
    stderr: str,
    expected: str,
) -> None:
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")
    _patch_exec(monkeypatch, lambda _a: _FakeProc(stderr=stderr.encode(), returncode=1))
    terminals, reason = await GhosttyTerminalAdapter().list_terminals()
    assert terminals is None
    assert reason == expected


@pytest.mark.asyncio
async def test_validate_terminal_matches_full_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")
    rows = "uuid-A\tt1\t/c\nuuid-B\tt2\t/d\n"
    _patch_exec(monkeypatch, lambda _a: _FakeProc(stdout=rows.encode(), returncode=0))
    ok, term, err = await GhosttyTerminalAdapter().validate_terminal("uuid-B")
    assert ok and err is None
    assert term is not None and term.terminal_id == "uuid-B" and term.name == "t2"


@pytest.mark.asyncio
async def test_validate_terminal_zero_match_is_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")
    _patch_exec(monkeypatch, lambda _a: _FakeProc(stdout=b"uuid-A\tt1\t/c\n", returncode=0))
    ok, term, err = await GhosttyTerminalAdapter().validate_terminal("missing")
    assert not ok and err == TerminalOutcome.NOT_FOUND and term is None


@pytest.mark.asyncio
async def test_inject_text_passes_payload_only_via_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """The user text and UUID must be argv items, never part of the script
    source — assert the script argument is the constant and the payload rides
    after ``--``."""
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")
    calls = _patch_exec(monkeypatch, lambda _a: _FakeProc(returncode=0))
    payload = "hello; rm -rf / && `whoami`"
    outcome = await GhosttyTerminalAdapter().inject_text("uuid-1", payload)
    assert outcome == TerminalOutcome.OK
    args = calls[0]
    assert args[0] == "osascript" and args[1] == "-e"
    assert args[2] == gta._INJECT_SCRIPT, "script source MUST be the constant"
    assert args[3] == "--"
    assert args[4] == "uuid-1"
    assert args[5] == payload
    # Ghostty queues ``input text`` asynchronously; delay before Enter so the
    # key cannot overtake the pasted payload and submit an empty prompt.
    assert gta._INJECT_SCRIPT.index("input text payload") < gta._INJECT_SCRIPT.index("delay 0.1")
    assert gta._INJECT_SCRIPT.index("delay 0.1") < gta._INJECT_SCRIPT.index('send key "enter"')
    # The payload must NOT appear inside the literal script source.
    assert payload not in gta._INJECT_SCRIPT


@pytest.mark.asyncio
async def test_inject_text_unicode_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")
    calls = _patch_exec(monkeypatch, lambda _a: _FakeProc(returncode=0))
    payload = "你好\n第二行\twith tab"
    outcome = await GhosttyTerminalAdapter().inject_text("uuid-9", payload)
    assert outcome == TerminalOutcome.OK
    assert calls[0][5] == payload


@pytest.mark.asyncio
async def test_user_question_select_uses_fixed_script_and_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")
    calls = _patch_exec(monkeypatch, lambda _a: _FakeProc(returncode=0))

    outcome = await GhosttyTerminalAdapter().select_user_question_option(
        "uuid-select",
        option_count=3,
        option_index=2,
        submit_after=True,
    )

    assert outcome == TerminalOutcome.OK
    assert calls[0][2] == gta._QUESTION_ACTION_SCRIPT
    assert calls[0][3:] == ["--", "uuid-select", "select", "3", "2", "1", ""]
    assert "uuid-select" not in gta._QUESTION_ACTION_SCRIPT
    # Ghostty's `Ghostty.Input.Key` String enum only accepts the camelCase
    # arrow names (arrowUp/arrowDown/...); the bare "up"/"down"/"right" names
    # are NOT valid and raise errAECoercionFail (-1700) "Unknown key name".
    assert 'send key "up"' not in gta._QUESTION_ACTION_SCRIPT
    assert 'send key "down"' not in gta._QUESTION_ACTION_SCRIPT
    assert 'send key "right"' not in gta._QUESTION_ACTION_SCRIPT
    assert 'send key "arrowUp"' in gta._QUESTION_ACTION_SCRIPT


@pytest.mark.asyncio
async def test_user_question_text_preserves_unicode_and_multiline_in_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")
    calls = _patch_exec(monkeypatch, lambda _a: _FakeProc(returncode=0))
    answer = "  你好\n第二行 'quoted'  "

    outcome = await GhosttyTerminalAdapter().answer_user_question_with_text(
        "uuid-text",
        option_count=2,
        text=answer,
        submit_after=False,
    )

    assert outcome == TerminalOutcome.OK
    assert calls[0][3:] == ["--", "uuid-text", "answer_text", "2", "-1", "0", answer]
    assert answer not in gta._QUESTION_ACTION_SCRIPT
    input_pos = gta._QUESTION_ACTION_SCRIPT.index("input text answerText")
    assert gta._QUESTION_ACTION_SCRIPT.index('send key "enter"') < input_pos
    assert input_pos < gta._QUESTION_ACTION_SCRIPT.index("delay 0.1", input_pos)


@pytest.mark.asyncio
async def test_user_question_multi_advance_resets_then_moves_right(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")
    calls = _patch_exec(monkeypatch, lambda _a: _FakeProc(returncode=0))

    outcome = await GhosttyTerminalAdapter().advance_user_question_after_multi_select(
        "uuid-multi",
        option_count=4,
        final_question=True,
    )

    assert outcome == TerminalOutcome.OK
    assert calls[0][3:] == ["--", "uuid-multi", "advance_multi", "4", "-1", "1", ""]
    reset_pos = gta._QUESTION_ACTION_SCRIPT.index("repeat (optionCount + 1) times")
    right_pos = gta._QUESTION_ACTION_SCRIPT.index('send key "arrowRight"')
    assert reset_pos < right_pos


@pytest.mark.asyncio
async def test_user_question_timeout_is_indeterminate_and_child_is_reaped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")

    class _HangingProc(_FakeProc):
        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(10)
            return b"", b""

    proc = _HangingProc(returncode=0)

    async def fake(*_args: object, **_kwargs: object) -> _HangingProc:
        return proc

    monkeypatch.setattr(gta.asyncio, "create_subprocess_exec", fake)
    outcome = await GhosttyTerminalAdapter(timeout_sec=0.01).select_user_question_option(
        "uuid",
        option_count=2,
        option_index=0,
        submit_after=False,
    )

    assert outcome == TerminalOutcome.INDETERMINATE
    assert proc.killed and proc.waited


@pytest.mark.asyncio
async def test_user_question_unknown_error_is_indeterminate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")
    _patch_exec(monkeypatch, lambda _a: _FakeProc(stderr=b"partial action failed", returncode=1))

    assert (
        await GhosttyTerminalAdapter().select_user_question_option(
            "uuid",
            option_count=2,
            option_index=0,
            submit_after=False,
        )
        == TerminalOutcome.INDETERMINATE
    )


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("select_user_question_option", {"option_count": 0, "option_index": 0, "submit_after": False}),
        ("select_user_question_option", {"option_count": 2, "option_index": 2, "submit_after": False}),
        ("answer_user_question_with_text", {"option_count": -1, "text": "x", "submit_after": False}),
        ("answer_user_question_with_text", {"option_count": 2, "text": "", "submit_after": False}),
        ("advance_user_question_after_multi_select", {"option_count": 0, "final_question": False}),
    ],
)
async def test_user_question_action_rejects_invalid_arguments(method: str, kwargs: dict[str, object]) -> None:
    adapter = GhosttyTerminalAdapter(platform_name="darwin")
    with pytest.raises(ValueError):
        await getattr(adapter, method)("uuid", **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        ("terminal not unique", TerminalOutcome.NOT_UNIQUE),
        ("Can't get terminal", TerminalOutcome.NOT_FOUND),
        ("not allowed to send Apple events", TerminalOutcome.TCC_DENIED),
        ('Can\'t get application "Ghostty"', TerminalOutcome.GHOSTTY_NOT_RUNNING),
    ],
)
async def test_inject_pre_flight_failures(
    monkeypatch: pytest.MonkeyPatch,
    stderr: str,
    expected: str,
) -> None:
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")
    _patch_exec(monkeypatch, lambda _a: _FakeProc(stderr=stderr.encode(), returncode=1))
    assert await GhosttyTerminalAdapter().inject_text("uuid-1", "x") == expected


@pytest.mark.asyncio
async def test_inject_ambiguous_failure_is_indeterminate_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure text that is not a recognized pre-flight reason is treated as
    INDETERMINATE (text may already be pasted); the caller must not retry."""
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")
    _patch_exec(monkeypatch, lambda _a: _FakeProc(stderr=b"some unexpected applescript error", returncode=1))
    assert await GhosttyTerminalAdapter().inject_text("uuid-1", "x") == TerminalOutcome.INDETERMINATE


@pytest.mark.asyncio
async def test_inject_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gta.sys, "platform", "linux")
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    assert await GhosttyTerminalAdapter().inject_text("uuid-1", "x") == "non_darwin"


@pytest.mark.asyncio
async def test_inject_timeout_kills_child(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")

    class _HangingProc(_FakeProc):
        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(10)
            return b"", b""

    proc = _HangingProc(returncode=0)

    async def fake(*_args: object, **_kwargs: object) -> _HangingProc:
        return proc

    monkeypatch.setattr(gta.asyncio, "create_subprocess_exec", fake)
    adapter = GhosttyTerminalAdapter(timeout_sec=0.01)
    assert await adapter.inject_text("uuid-1", "x") == TerminalOutcome.TIMEOUT
    assert proc.killed, "timeout SHALL kill the osascript child"
    assert proc.waited, "timeout SHALL reap the osascript child"


@pytest.mark.asyncio
async def test_inject_cancellation_kills_and_reaps_child(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")
    entered = asyncio.Event()

    class _HangingProc(_FakeProc):
        async def communicate(self) -> tuple[bytes, bytes]:
            entered.set()
            await asyncio.Event().wait()
            return b"", b""

    proc = _HangingProc(returncode=0)

    async def fake(*_args: object, **_kwargs: object) -> _HangingProc:
        return proc

    monkeypatch.setattr(gta.asyncio, "create_subprocess_exec", fake)
    task = asyncio.create_task(GhosttyTerminalAdapter().inject_text("uuid-1", "x"))
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert proc.killed
    assert proc.waited


@pytest.mark.asyncio
async def test_kill_abandons_reap_when_child_wedges(monkeypatch: pytest.MonkeyPatch) -> None:
    """#11: a wedged osascript (D-state / grandchild holding the pipe) makes
    ``proc.wait()`` never return. ``_kill`` must bound that reap so the caller's
    input-lock await context is released instead of pinning the whole session
    behind an unreachable lock. The injected child is still ``kill()``-ed; the
    reap is simply abandoned once it times out.
    """
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")

    class _WedgedProc(_FakeProc):
        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(10)
            return b"", b""

        async def wait(self) -> int:
            # Simulates an osascript stuck in uninterruptible state: wait()
            # never reaps, no matter how long we hold the input lock.
            await asyncio.Event().wait()  # pragma: no cover - test relies on timeout
            return self.returncode  # pragma: no cover - unreachable

    proc = _WedgedProc(returncode=0)

    async def fake(*_args: object, **_kwargs: object) -> _WedgedProc:
        return proc

    monkeypatch.setattr(gta.asyncio, "create_subprocess_exec", fake)
    # Tiny command timeout so communicate() aborts quickly; the reap budget is
    # the same value, so the wedged wait() is abandoned within ~0.01s.
    adapter = GhosttyTerminalAdapter(timeout_sec=0.01)
    # Must return promptly (not hang); classifies as TIMEOUT.
    assert await adapter.inject_text("uuid-1", "x") == TerminalOutcome.TIMEOUT
    assert proc.killed, "timeout SHALL still SIGKILL the wedged child"
    assert not proc.waited, "wedged child is NOT reaped — _kill abandoned the bounded wait"


@pytest.mark.asyncio
async def test_inject_osascript_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gta.shutil, "which", lambda _b: "/usr/bin/osascript")
    monkeypatch.setattr(gta.sys, "platform", "darwin")

    async def fake(*_args: object, **_kwargs: object) -> _FakeProc:
        raise FileNotFoundError

    monkeypatch.setattr(gta.asyncio, "create_subprocess_exec", fake)
    # FileNotFoundError from create_subprocess_exec is caught by _run_script,
    # yielding a non-empty stderr so inject classifies os_error / not available.
    outcome = await GhosttyTerminalAdapter().inject_text("uuid-1", "x")
    assert outcome in {TerminalOutcome.OS_ERROR, "osascript_missing"}


def test_parse_terminals_handles_empty_and_partial() -> None:
    terminals = gta._parse_terminals("uuid-1\tname\t/c\n\nbad\nuuid-2\t\t\n")
    assert [t.terminal_id for t in terminals] == ["uuid-1", "uuid-2"]
    assert terminals[0].name == "name" and terminals[0].cwd == "/c"
    assert terminals[1].name is None and terminals[1].cwd is None


def test_ghostty_terminal_is_frozen() -> None:
    import dataclasses

    t = ExternalTerminal(kind="ghostty", terminal_id="x", name=None, cwd=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.terminal_id = "y"  # type: ignore[misc]
