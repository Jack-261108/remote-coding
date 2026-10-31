"""Full-featured asyncio.subprocess fake for monkeypatched ``create_subprocess_exec``.

Superset of the per-file ``_FakeProcess`` / ``_FakeProc`` copies that lived in
the pty-injector / tmux-runner / ghostty-adapter tests.
"""

from __future__ import annotations


class FakeSubprocessProcess:
    def __init__(self, *, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.killed = False
        self.waited = False
        self.stdin_data: bytes | None = None

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        self.stdin_data = input
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        self.waited = True
        return self.returncode
