from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.adapters.claude.paths import ClaudePaths


@dataclass(frozen=True, order=True)
class ClaudeCodeVersion:
    major: int
    minor: int
    patch: int


class HookInstaller:
    HOOK_SCRIPT_NAME = "remote-coding-hook.py"

    def __init__(
        self,
        *,
        paths: ClaudePaths,
        socket_path: str,
        python_bin: str | None = None,
        claude_bin: str = "claude",
    ) -> None:
        self._paths = paths
        self._socket_path = socket_path
        self._python_bin = python_bin or sys.executable or "python3"
        self._claude_bin = claude_bin

    def install(self, *, version: ClaudeCodeVersion | None = None) -> Path:
        hooks_dir = self._paths.hooks_dir
        hooks_dir.mkdir(parents=True, exist_ok=True)

        script_path = self._paths.hook_script_path(self.HOOK_SCRIPT_NAME)
        script_path.write_text(self._render_hook_script(), encoding="utf-8")
        script_path.chmod(0o755)

        self._update_settings(script_path=script_path, version=version or self.detect_claude_code_version())
        return script_path

    def detect_claude_code_version(self) -> ClaudeCodeVersion | None:
        try:
            completed = subprocess.run(
                [self._claude_bin, "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        return self.parse_claude_code_version(completed.stdout)

    @staticmethod
    def parse_claude_code_version(text: str) -> ClaudeCodeVersion | None:
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
        if match is None:
            return None
        return ClaudeCodeVersion(*(int(part) for part in match.groups()))

    def supported_hook_events(self, version: ClaudeCodeVersion | None) -> list[tuple[str, list[dict[str, object]]]]:
        hook_entry: list[dict[str, object]] = [{"type": "command", "command": self._command()}]
        hook_entry_with_timeout: list[dict[str, object]] = [{"type": "command", "command": self._command(), "timeout": 86400}]
        with_matcher: list[dict[str, object]] = [{"matcher": "*", "hooks": hook_entry}]
        with_matcher_and_timeout: list[dict[str, object]] = [{"matcher": "*", "hooks": hook_entry_with_timeout}]
        without_matcher: list[dict[str, object]] = [{"hooks": hook_entry}]
        pre_compact: list[dict[str, object]] = [
            {"matcher": "auto", "hooks": hook_entry},
            {"matcher": "manual", "hooks": hook_entry},
        ]

        events: list[tuple[str, list[dict[str, object]]]] = [
            ("UserPromptSubmit", without_matcher),
            ("PreToolUse", with_matcher),
            ("PostToolUse", with_matcher),
            ("PermissionRequest", with_matcher_and_timeout),
            ("Notification", with_matcher),
            ("Stop", without_matcher),
            ("SubagentStop", without_matcher),
            ("SessionStart", without_matcher),
            ("SessionEnd", without_matcher),
            ("PreCompact", pre_compact),
        ]
        if version is None:
            return events
        if version >= ClaudeCodeVersion(2, 0, 0):
            events.append(("PostToolUseFailure", with_matcher))
        if version >= ClaudeCodeVersion(2, 0, 43):
            events.append(("SubagentStart", without_matcher))
        if version >= ClaudeCodeVersion(2, 1, 76):
            events.append(("PostCompact", pre_compact))
        if version >= ClaudeCodeVersion(2, 1, 78):
            events.append(("StopFailure", without_matcher))
        if version >= ClaudeCodeVersion(2, 1, 88):
            events.append(("PermissionDenied", with_matcher))
        return events

    def _update_settings(self, *, script_path: Path, version: ClaudeCodeVersion | None) -> None:
        settings_path = self._paths.settings_file
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {}
        if settings_path.exists():
            try:
                payload = json.loads(settings_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}

        hooks = payload.get("hooks")
        if not isinstance(hooks, dict):
            hooks = {}

        cleaned_hooks: dict[str, object] = {}
        for event_name, config in hooks.items():
            if isinstance(config, list):
                cleaned_entries = [
                    entry for entry in (self._clean_entry(item) for item in config if isinstance(item, dict)) if entry is not None
                ]
                if cleaned_entries:
                    cleaned_hooks[event_name] = cleaned_entries
            else:
                cleaned_hooks[event_name] = config

        for event_name, config in self.supported_hook_events(version):
            existing = cleaned_hooks.get(event_name)
            if not isinstance(existing, list):
                existing = []
            cleaned_hooks[event_name] = [*existing, *config]

        payload["hooks"] = cleaned_hooks
        settings_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def _clean_entry(self, entry: dict[str, object]) -> dict[str, object] | None:
        hooks = entry.get("hooks")
        if not isinstance(hooks, list):
            return entry
        filtered_hooks = [hook for hook in hooks if not self._is_remote_coding_hook(hook)]
        if not filtered_hooks:
            return None
        updated = dict(entry)
        updated["hooks"] = filtered_hooks
        return updated

    def _is_remote_coding_hook(self, hook: object) -> bool:
        if not isinstance(hook, dict):
            return False
        command = str(hook.get("command", ""))
        return self.HOOK_SCRIPT_NAME in command

    def _command(self) -> str:
        return f"{shlex.quote(self._python_bin)} {shlex.quote(str(self._paths.hook_script_path(self.HOOK_SCRIPT_NAME)))}"

    def _render_hook_script(self) -> str:
        return f"""#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import sys

DEFAULT_SOCKET_PATH = {self._socket_path!r}

_STATUS_BY_EVENT = {{
    "SessionStart": "starting",
    "UserPromptSubmit": "processing",
    "PreToolUse": "running_tool",
    "PermissionRequest": "waiting_for_approval",
    "PermissionDenied": "processing",
    "PostToolUse": "processing",
    "PostToolUseFailure": "processing",
    "PreCompact": "processing",
    "PostCompact": "processing",
    "SubagentStart": "processing",
    "Stop": "waiting_for_input",
    "SubagentStop": "processing",
    "StopFailure": "waiting_for_input",
    "SessionEnd": "ended",
}}


def _pick(payload: dict[str, object], *names: str) -> object | None:
    for name in names:
        value = payload.get(name)
        if value is not None:
            return value
    return None


def _normalize_message(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _normalize_tool_input(value: object | None) -> dict[str, object] | None:
    return dict(value) if isinstance(value, dict) else None


def _normalize_pid(value: object | None) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        normalized = int(value)
        return normalized if normalized > 0 else None
    if isinstance(value, str) and value.strip():
        try:
            normalized = int(value)
        except ValueError:
            return None
        return normalized if normalized > 0 else None
    return None


def _resolve_pid(value: object | None) -> int | None:
    normalized = _normalize_pid(value)
    if normalized is not None:
        return normalized
    parent_pid = os.getppid()
    return parent_pid if parent_pid > 0 else None


def _status_for_event(event: str, payload: dict[str, object]) -> str:
    explicit = _pick(payload, "status")
    if explicit is not None:
        return str(explicit)

    if event == "Notification":
        notification_type = _pick(payload, "notification_type", "notificationType")
        if notification_type == "idle_prompt":
            return "waiting_for_input"
        if notification_type == "permission_prompt":
            return "waiting_for_approval"
        return "processing"

    return _STATUS_BY_EVENT.get(event, "processing")


def normalize_payload(payload: dict[str, object]) -> dict[str, object]:
    if all(key in payload for key in ("session_id", "cwd", "event", "status")):
        normalized = dict(payload)
        normalized["pid"] = _resolve_pid(normalized.get("pid"))
        if "tool_input" in normalized:
            normalized["tool_input"] = _normalize_tool_input(normalized.get("tool_input"))
        if "message" in normalized:
            normalized["message"] = _normalize_message(normalized.get("message"))
        return normalized

    event = _pick(payload, "event", "hook_event_name", "name")
    session_id = _pick(payload, "session_id", "sessionId")
    cwd = _pick(payload, "cwd", "working_directory")
    tool = _pick(payload, "tool", "tool_name", "toolName")
    tool_input = _pick(payload, "tool_input", "toolInput", "input")
    tool_use_id = _pick(payload, "tool_use_id", "toolUseId", "toolUseID")
    notification_type = _pick(payload, "notification_type", "notificationType")
    message = _pick(payload, "message")
    pid = _pick(payload, "pid")
    tty = _pick(payload, "tty")

    event_name = str(event or "Unknown")
    normalized = {{
        "session_id": str(session_id or "unknown"),
        "cwd": str(cwd or os.getcwd()),
        "event": event_name,
        "status": _status_for_event(event_name, payload),
        "pid": _resolve_pid(pid),
        "tty": str(tty) if tty is not None else None,
        "tool": str(tool) if tool is not None else None,
        "tool_input": _normalize_tool_input(tool_input),
        "tool_use_id": str(tool_use_id) if tool_use_id is not None else None,
        "notification_type": str(notification_type) if notification_type is not None else None,
        "message": _normalize_message(message),
    }}
    return normalized


def _expects_response(payload: dict[str, object]) -> bool:
    return payload.get("event") == "PermissionRequest" and payload.get("status") == "waiting_for_approval"


def main() -> int:
    payload = sys.stdin.buffer.read()
    if not payload:
        return 0

    try:
        raw_payload = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 0
    if not isinstance(raw_payload, dict):
        return 0

    normalized_payload = normalize_payload(raw_payload)
    expects_response = _expects_response(normalized_payload)

    socket_path = os.environ.get("REMOTE_CODING_HOOK_SOCKET_PATH", DEFAULT_SOCKET_PATH)
    response = b""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(socket_path)
            client.sendall(json.dumps(normalized_payload, ensure_ascii=False).encode("utf-8") + b"\\n")
            if not expects_response:
                return 0
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                response += chunk
    except OSError:
        return 0

    if response:
        sys.stdout.write(response.decode("utf-8", errors="replace"))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
