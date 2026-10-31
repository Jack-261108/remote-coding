"""Unit tests for KittyTerminalAdapter.

Covers the ``kitten @`` transport contract (socket address via ``--to``,
payload via ``--stdin``), the ``ls`` JSON flattening, and outcome
classification. No real kitty instance is required: ``asyncio``
``create_subprocess_exec`` is monkeypatched in the adapter module.
"""

from __future__ import annotations

import json

import pytest

from app.adapters.process import kitty_terminal_adapter as kta
from app.adapters.process.kitty_terminal_adapter import (
    KittyTerminalAdapter,
    TerminalOutcome,
    _parse_ls_json,
)
from tests.fakes.subprocess_process import FakeSubprocessProcess

_SOCKET = "unix:/tmp/kitty-1234"


def _patch_exec(monkeypatch: pytest.MonkeyPatch, factory) -> list[list[object]]:
    calls: list[list[object]] = []

    async def fake(*args: object, **_kwargs: object) -> object:
        calls.append(list(args))
        return factory(list(args))

    monkeypatch.setattr(kta.asyncio, "create_subprocess_exec", fake)
    return calls


def _keys_sent(calls: list[list[object]]) -> list[object]:
    """Extract the key name from each ``kitten @ send-key --match id:N <key>`` call."""
    return [call[7] for call in calls if call[4] == "send-key"]


def _adapter(**kwargs: object) -> KittyTerminalAdapter:
    params: dict[str, object] = {
        "listen_on": _SOCKET,
        "platform_name": "darwin",
    }
    params.update(kwargs)
    return KittyTerminalAdapter(**params)  # type: ignore[arg-type]


def _ls_payload() -> str:
    return json.dumps(
        [
            {
                "id": 1,
                "tabs": [
                    {
                        "id": 1,
                        "windows": [
                            {
                                "id": 11,
                                "title": "claude — project",
                                "cwd": "/project",
                                "pid": 1234,
                                "foreground_processes": [{"pid": 1234, "cmdline": ["claude"], "cwd": "/project"}],
                            },
                            {"id": 12, "title": "shell", "cwd": "/tmp", "pid": 999, "foreground_processes": []},
                        ],
                    }
                ],
            },
            {
                "id": 2,
                "tabs": [
                    {
                        "id": 2,
                        "windows": [
                            {"id": 21, "title": "other", "cwd": "/other", "pid": 555, "foreground_processes": []},
                        ],
                    }
                ],
            },
        ]
    )


# ─── availability ───────────────────────────────────────────────────


def test_is_available_requires_socket_and_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")
    assert _adapter().is_available()

    assert not _adapter(listen_on="").is_available(), "no socket address -> unavailable"
    assert not _adapter(enable_remote_control=False).is_available()

    monkeypatch.setattr(kta.shutil, "which", lambda _b: None)
    monkeypatch.setattr(kta.os, "access", lambda _p, _m: False)
    assert not _adapter().is_available()


def test_kitten_falls_back_to_app_bundle_on_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    """GUI launchers hand a PATH without /opt/homebrew/bin; the bundle path saves us."""
    monkeypatch.setattr(kta.shutil, "which", lambda _b: None)
    monkeypatch.setattr(kta.os, "access", lambda _p, _m: True)
    assert _adapter().is_available()

    # A custom binary name never falls back — only the default ``kitten`` does.
    assert not _adapter(kitten_bin="/custom/kitten").is_available()


def test_listen_on_falls_back_to_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")
    assert _adapter(listen_on="", env_listen_on=_SOCKET).is_available()
    assert not _adapter(listen_on="", env_listen_on="   ").is_available()


def test_unavailable_reason_distinguishes_socket_from_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")
    assert _adapter(listen_on="")._unavailable_reason() == TerminalOutcome.KITTY_REMOTE_CONTROL_DISABLED

    monkeypatch.setattr(kta.shutil, "which", lambda _b: None)
    monkeypatch.setattr(kta.os, "access", lambda _p, _m: False)
    assert _adapter()._unavailable_reason() == TerminalOutcome.KITTY_BIN_MISSING


# ─── ls / enumeration ───────────────────────────────────────────────


def test_parse_ls_json_flattens_every_window_across_os_windows() -> None:
    terminals = _parse_ls_json(_ls_payload())
    assert [t.terminal_id for t in terminals] == ["11", "12", "21"]
    assert all(t.kind == "kitty" for t in terminals)

    first = terminals[0]
    assert first.name == "claude — project"
    assert first.cwd == "/project"
    assert first.pid == 1234
    assert first.foreground_pids == (1234,)


def test_parse_ls_json_skips_malformed_entries_without_hiding_valid_ones() -> None:
    raw = json.dumps(
        [
            "not-a-dict",
            {"tabs": "not-a-list"},
            {"tabs": [{"windows": [{"id": "not-an-int"}]}]},
            {"tabs": [{"windows": [{"id": 7, "title": "", "cwd": None}]}]},
        ]
    )
    terminals = _parse_ls_json(raw)
    assert [t.terminal_id for t in terminals] == ["7"]
    # Empty title / non-str cwd become None, never empty strings.
    assert terminals[0].name is None
    assert terminals[0].cwd is None
    assert terminals[0].pid is None
    assert terminals[0].foreground_pids == ()


def test_parse_ls_json_rejects_non_list_payload() -> None:
    with pytest.raises(TypeError):
        _parse_ls_json('{"id": 1}')


async def test_list_terminals_runs_ls_against_configured_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess(stdout=_ls_payload().encode()))
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    terminals, reason = await _adapter().list_terminals()

    assert reason is None
    assert terminals is not None and len(terminals) == 3
    assert calls[0][:4] == ["kitten", "@", "--to", _SOCKET]
    assert calls[0][4:] == ["ls"]


async def test_list_terminals_classifies_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess(returncode=1, stderr=b"Failed to connect to socket"))
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    terminals, reason = await _adapter().list_terminals()

    assert terminals is None
    assert reason == TerminalOutcome.KITTY_NOT_RUNNING


async def test_list_terminals_reports_os_error_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess(stdout=b"<html>nope</html>"))
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    terminals, reason = await _adapter().list_terminals()

    assert terminals is None
    assert reason == TerminalOutcome.OS_ERROR


async def test_list_terminals_short_circuits_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess())
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    terminals, reason = await _adapter(listen_on="").list_terminals()

    assert terminals is None
    assert reason == TerminalOutcome.KITTY_REMOTE_CONTROL_DISABLED
    assert calls == [], "unavailable adapter must not spawn subprocesses"


# ─── validate ───────────────────────────────────────────────────────


async def test_validate_terminal_returns_unique_match(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess(stdout=_ls_payload().encode()))
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    ok, terminal, reason = await _adapter().validate_terminal("11")

    assert ok is True
    assert reason is None
    assert terminal is not None and terminal.terminal_id == "11" and terminal.kind == "kitty"


async def test_validate_terminal_reports_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess(stdout=_ls_payload().encode()))
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    ok, terminal, reason = await _adapter().validate_terminal("404")

    assert (ok, terminal, reason) == (False, None, TerminalOutcome.NOT_FOUND)


async def test_validate_terminal_reports_duplicate_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    duplicated = json.dumps(
        [{"tabs": [{"windows": [{"id": 5}, {"id": 5}]}]}],
    )
    _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess(stdout=duplicated.encode()))
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    ok, terminal, reason = await _adapter().validate_terminal("5")

    assert (ok, terminal, reason) == (False, None, TerminalOutcome.NOT_UNIQUE)


# ─── injection ──────────────────────────────────────────────────────


async def test_inject_text_pipes_payload_through_stdin_then_presses_enter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    procs: list[FakeSubprocessProcess] = []

    def factory(_args: list[object]) -> FakeSubprocessProcess:
        proc = FakeSubprocessProcess()
        procs.append(proc)
        return proc

    calls = _patch_exec(monkeypatch, factory)
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    outcome = await _adapter().inject_text("11", "hello \\e world")

    assert outcome == TerminalOutcome.OK
    assert calls[0][4:] == ["send-text", "--match", "id:11", "--stdin"]
    assert procs[0].stdin_data == b"hello \\e world", "payload must be raw bytes, never argv-interpreted"
    assert calls[1][4:] == ["send-key", "--match", "id:11", "enter"]
    assert procs[1].stdin_data is None


async def test_inject_text_rejects_non_numeric_id_before_spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess())
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    with pytest.raises(ValueError):
        await _adapter().inject_text("id:11 or 12", "x")
    assert calls == []


async def test_inject_text_paste_failure_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess(returncode=1, stderr=b"Failed to connect to socket"))
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    assert await _adapter().inject_text("11", "x") == TerminalOutcome.KITTY_NOT_RUNNING


async def test_inject_text_enter_failure_is_indeterminate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The paste already landed: retrying would duplicate the text."""
    state = {"n": 0}

    def factory(_args: list[object]) -> FakeSubprocessProcess:
        state["n"] += 1
        if state["n"] == 1:
            return FakeSubprocessProcess()
        return FakeSubprocessProcess(returncode=1, stderr=b"unexpected failure")

    _patch_exec(monkeypatch, factory)
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    assert await _adapter().inject_text("11", "x") == TerminalOutcome.INDETERMINATE


async def test_run_timeout_kills_child_and_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = FakeSubprocessProcess()

    async def never(*_args: object, **_kwargs: object) -> tuple[bytes, bytes]:
        await kta.asyncio.sleep(10)
        return b"", b""

    monkeypatch.setattr(proc, "communicate", never)
    _patch_exec(monkeypatch, lambda _a: proc)
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    outcome = await _adapter(timeout_sec=0.01).inject_text("11", "x")

    assert outcome == TerminalOutcome.TIMEOUT
    assert proc.killed is True


async def test_run_reports_missing_binary_without_spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess())
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    adapter = _adapter()
    monkeypatch.setattr(adapter, "_resolved_kitten_bin", lambda: None)

    assert await adapter.inject_text("11", "x") == TerminalOutcome.KITTY_BIN_MISSING
    assert calls == []


async def test_run_swallows_spawn_oserror_and_reports_bin_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A custom KITTY_BIN without +x raises PermissionError (an OSError), which
    must be classified as a retryable preflight failure — never propagated."""
    calls = _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess())

    async def permission_denied(*_args: object, **_kwargs: object) -> object:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(kta.asyncio, "create_subprocess_exec", permission_denied)
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/custom/kitten")

    outcome = await _adapter().inject_text("11", "x")

    assert outcome == TerminalOutcome.KITTY_BIN_MISSING
    assert calls == []


# ─── AskUserQuestion key sequences ──────────────────────────────────


async def test_select_option_walks_up_then_down_then_enter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess())
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    outcome = await _adapter(key_delay_sec=0).select_user_question_option(
        "11",
        option_count=3,
        option_index=1,
        submit_after=True,
    )

    assert outcome == TerminalOutcome.OK
    # up x (option_count+1) -> down x option_index -> enter, then a submit enter.
    assert _keys_sent(calls) == ["up", "up", "up", "up", "down", "enter", "enter"]


async def test_select_option_skips_final_enter_when_not_submitting(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess())
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    outcome = await _adapter(key_delay_sec=0).select_user_question_option(
        "11",
        option_count=2,
        option_index=0,
        submit_after=False,
    )

    assert outcome == TerminalOutcome.OK
    assert _keys_sent(calls) == ["up", "up", "up", "enter"]


async def test_answer_with_text_moves_below_options_then_pastes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess())
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    outcome = await _adapter(key_delay_sec=0).answer_user_question_with_text(
        "11",
        option_count=2,
        text="custom answer",
        submit_after=True,
    )

    assert outcome == TerminalOutcome.OK
    # up × (option_count+1) resets the cursor, down × option_count selects the
    # Other option, enter opens the editor; the paste is then confirmed with
    # its own enter before the submit enter.
    assert _keys_sent(calls) == ["up", "up", "up", "down", "down", "enter", "enter", "enter"]
    assert calls[6][4:] == ["send-text", "--match", "id:11", "--stdin"], "paste sits between the selection enter and the confirm enter"


async def test_answer_text_without_submit_still_confirms_paste_with_enter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-final question: the paste MUST still be confirmed with enter, or the
    text stays in the editor and the multi-question flow stalls."""
    calls = _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess())
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    outcome = await _adapter(key_delay_sec=0).answer_user_question_with_text(
        "11",
        option_count=1,
        text="answer",
        submit_after=False,
    )

    assert outcome == TerminalOutcome.OK
    # up × 2 -> down × 1 -> enter (open editor) -> [paste] -> enter (confirm).
    # No trailing submit enter because submit_after is False.
    assert _keys_sent(calls) == ["up", "up", "down", "enter", "enter"]
    assert calls[4][4:] == ["send-text", "--match", "id:11", "--stdin"]


async def test_advance_multi_select_presses_right(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess())
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    outcome = await _adapter(key_delay_sec=0).advance_user_question_after_multi_select(
        "11",
        option_count=2,
        final_question=False,
    )

    assert outcome == TerminalOutcome.OK
    # The cursor reset (up × (option_count+1)) must precede the right key, or
    # the form controls never receive it.
    assert _keys_sent(calls) == ["up", "up", "up", "right"]


async def test_question_action_failure_after_first_key_is_indeterminate(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"n": 0}

    def factory(_args: list[object]) -> FakeSubprocessProcess:
        state["n"] += 1
        return FakeSubprocessProcess() if state["n"] == 1 else FakeSubprocessProcess(returncode=1, stderr=b"boom")

    _patch_exec(monkeypatch, factory)
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    outcome = await _adapter(key_delay_sec=0).select_user_question_option(
        "11",
        option_count=2,
        option_index=0,
        submit_after=True,
    )

    assert outcome == TerminalOutcome.INDETERMINATE


async def test_question_action_first_key_failure_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_exec(monkeypatch, lambda _a: FakeSubprocessProcess(returncode=1, stderr=b"Connection refused"))
    monkeypatch.setattr(kta.shutil, "which", lambda _b: "/usr/local/bin/kitten")

    outcome = await _adapter(key_delay_sec=0).select_user_question_option(
        "11",
        option_count=2,
        option_index=0,
        submit_after=True,
    )

    assert outcome == TerminalOutcome.KITTY_NOT_RUNNING


async def test_question_actions_validate_arguments() -> None:
    adapter = _adapter()
    with pytest.raises(ValueError):
        await adapter.select_user_question_option("11", option_count=0, option_index=0, submit_after=True)
    with pytest.raises(ValueError):
        await adapter.select_user_question_option("11", option_count=2, option_index=2, submit_after=True)
    with pytest.raises(ValueError):
        await adapter.answer_user_question_with_text("11", option_count=-1, text="x", submit_after=True)
    with pytest.raises(ValueError):
        await adapter.answer_user_question_with_text("11", option_count=1, text="", submit_after=True)
    with pytest.raises(ValueError):
        await adapter.advance_user_question_after_multi_select("11", option_count=0, final_question=True)


# ─── failure classification ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("kitten missing", TerminalOutcome.KITTY_BIN_MISSING),
        ("kitten spawn failed", TerminalOutcome.KITTY_BIN_MISSING),
        ("no listen_on configured", TerminalOutcome.KITTY_REMOTE_CONTROL_DISABLED),
        ("Failed to connect to socket", TerminalOutcome.KITTY_NOT_RUNNING),
        ("No such file or directory", TerminalOutcome.KITTY_NOT_RUNNING),
        ("Connection refused", TerminalOutcome.KITTY_NOT_RUNNING),
        ("Remote control is not allowed", TerminalOutcome.KITTY_REMOTE_CONTROL_DISABLED),
        ("Permission denied", TerminalOutcome.KITTY_REMOTE_CONTROL_DISABLED),
        ("timeout", TerminalOutcome.TIMEOUT),
        ("something else", TerminalOutcome.OS_ERROR),
    ],
)
def test_list_failure_classification(message: str, expected: str) -> None:
    assert KittyTerminalAdapter._classify_failure(message, 1) == expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("kitten missing", TerminalOutcome.KITTY_BIN_MISSING),
        ("kitten spawn failed", TerminalOutcome.KITTY_BIN_MISSING),
        ("no listen_on configured", TerminalOutcome.KITTY_REMOTE_CONTROL_DISABLED),
        ("Failed to connect to socket", TerminalOutcome.KITTY_NOT_RUNNING),
        ("Remote control is not allowed", TerminalOutcome.KITTY_REMOTE_CONTROL_DISABLED),
        ("timeout", TerminalOutcome.TIMEOUT),
        # kitty gives no delivery signal for anything else -> never retry.
        ("something else", TerminalOutcome.INDETERMINATE),
    ],
)
def test_inject_failure_classification(message: str, expected: str) -> None:
    assert KittyTerminalAdapter._classify_inject_failure(message, 1) == expected


def test_question_failure_classification_never_returns_os_error() -> None:
    assert KittyTerminalAdapter._classify_question_failure("weird") == TerminalOutcome.INDETERMINATE
    assert KittyTerminalAdapter._classify_question_failure("Connection refused") == TerminalOutcome.KITTY_NOT_RUNNING
    assert KittyTerminalAdapter._classify_question_failure("kitten spawn failed") == TerminalOutcome.KITTY_BIN_MISSING
