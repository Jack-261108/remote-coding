"""Domain-layer protocols (interfaces) for cross-layer dependencies.

Adapters should depend on these protocols instead of concrete service classes.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from app.domain.external_terminal_models import ExternalTerminal, TerminalKind
    from app.domain.models import CLIEvent, ExecutionTask
    from app.domain.session_models import SessionEvent, SessionPhase, SessionState
    from app.domain.user_question_models import ExternalQuestionActionResult, ExternalUserQuestionContext


@dataclass(frozen=True)
class AdapterCapabilities:
    """声明 CLI adapter/runtime 可用能力的轻量视图。

    静态位（persistent_terminal / interactive_input / claude_resume /
    user_question_tui / session_state）表达"该 provider 类别支持某能力"，
    与运行环境无关——claude_code 恒真，codex/gemini 恒假。动态位
    persistent_terminal_active 表达"持久终端后端此刻可用"（tmux 已开启并
    完成初始化），随 settings.claude_tmux_mode 变化。
    """

    run_task: bool = True
    cancel_task: bool = True
    persistent_terminal: bool = False
    interactive_input: bool = False
    claude_resume: bool = False
    user_question_tui: bool = False
    session_state: bool = False
    persistent_terminal_active: bool = False


@runtime_checkable
class CLIAdapterProtocol(Protocol):
    """一次性或交互式任务执行 adapter。"""

    provider: str

    def run(
        self,
        task: ExecutionTask,
        *,
        terminal_key: str | None = None,
        interactive: bool = False,
        claude_session_id: str | None = None,
    ) -> AsyncGenerator[CLIEvent, None]: ...

    async def cancel(self, task_id: str) -> bool: ...

    def build_file_args(self, file_paths: list[Path]) -> list[str]: ...


@runtime_checkable
class CLIAdapterRegistryProtocol(Protocol):
    """Provider 归一化与 adapter 查询接口。"""

    def normalize_provider(self, provider: str) -> str: ...

    def get(self, provider: str) -> CLIAdapterProtocol: ...

    def available_providers(self) -> list[str]: ...

    def capabilities(self, provider: str) -> AdapterCapabilities: ...


@runtime_checkable
class ClaudeTerminalRuntimeProtocol(Protocol):
    """Claude 持久终端生命周期和文本输入能力。"""

    async def close_terminal(self, terminal_key: str) -> tuple[bool, str]: ...

    async def ensure_terminal(self, *, terminal_key: str, workdir: str) -> tuple[bool, str]: ...

    async def ensure_interactive_session(self, *, terminal_key: str, workdir: str) -> tuple[bool, str]: ...

    async def ensure_resume_session(self, *, terminal_key: str, workdir: str, session_id: str) -> tuple[bool, str]: ...

    async def reveal_terminal(self, terminal_key: str) -> tuple[bool, str]: ...

    async def send_interactive_input(self, *, terminal_key: str, workdir: str, text: str) -> tuple[bool, str]: ...


@runtime_checkable
class ClaudeUserQuestionTransportProtocol(Protocol):
    """Claude AskUserQuestion 的 managed terminal/TUI 操作能力。"""

    async def select_option(
        self,
        *,
        terminal_key: str,
        workdir: str,
        option_index: int,
        submit_after: bool = False,
    ) -> tuple[bool, str]: ...

    async def answer_with_text(
        self,
        *,
        terminal_key: str,
        workdir: str,
        option_count: int,
        text: str,
        submit_after: bool = False,
    ) -> tuple[bool, str]: ...

    async def advance_after_multi_select(
        self,
        *,
        terminal_key: str,
        workdir: str,
        final_question: bool,
    ) -> tuple[bool, str]: ...


@runtime_checkable
class ExternalTerminalBackendProtocol(Protocol):
    """一个外部终端模拟器（Ghostty / kitty）的输入注入能力。

    后端只负责"在给定 terminal_id 上做事"，不做所有权、绑定代际或进程校验：
    那些由 ``ExternalSessionInputService`` 统一负责。所有方法都返回
    ``TerminalOutcome`` 常量字符串而不是抛异常，让服务端可以区分可重试的
    预检失败与不可重试的 INDETERMINATE。

    设计：``docs/specs/2026-10-29-external-kitty-input-design.md`` §2。
    """

    kind: TerminalKind

    def is_available(self) -> bool: ...

    async def list_terminals(self) -> tuple[list[ExternalTerminal] | None, str | None]: ...

    async def validate_terminal(self, terminal_id: str) -> tuple[bool, ExternalTerminal | None, str | None]: ...

    async def inject_text(self, terminal_id: str, text: str) -> str: ...

    async def select_user_question_option(
        self,
        terminal_id: str,
        *,
        option_count: int,
        option_index: int,
        submit_after: bool,
    ) -> str: ...

    async def answer_user_question_with_text(
        self,
        terminal_id: str,
        *,
        option_count: int,
        text: str,
        submit_after: bool,
    ) -> str: ...

    async def advance_user_question_after_multi_select(
        self,
        terminal_id: str,
        *,
        option_count: int,
        final_question: bool,
    ) -> str: ...


@runtime_checkable
class ExternalClaudeUserQuestionTransportProtocol(Protocol):
    """外部终端 AskUserQuestion 的显式 target 操作能力。"""

    async def select_option(
        self,
        *,
        context: ExternalUserQuestionContext,
        question_index: int,
        option_count: int,
        option_index: int,
        submit_after: bool,
    ) -> ExternalQuestionActionResult: ...

    async def answer_with_text(
        self,
        *,
        context: ExternalUserQuestionContext,
        question_index: int,
        option_count: int,
        text: str,
        submit_after: bool,
    ) -> ExternalQuestionActionResult: ...

    async def advance_after_multi_select(
        self,
        *,
        context: ExternalUserQuestionContext,
        question_index: int,
        option_count: int,
        final_question: bool,
    ) -> ExternalQuestionActionResult: ...

    async def question_completed(self, *, context: ExternalUserQuestionContext) -> None: ...

    async def question_indeterminate(self, *, context: ExternalUserQuestionContext, reason: str) -> None: ...


@runtime_checkable
class SessionStateReaderProtocol(Protocol):
    """读取 adapter/runtime 内部 structured session state 的兼容接口。"""

    def get_session_state(self, terminal_key: str) -> SessionState | None: ...

    def get_claude_session_state(self, session_id: str) -> SessionState | None: ...


@runtime_checkable
class CLIInfrastructureProtocol(CLIAdapterRegistryProtocol, Protocol):
    """组合 registry 与 Claude runtime capability 的兼容入口。"""

    @property
    def claude_terminal_runtime(self) -> ClaudeTerminalRuntimeProtocol: ...

    @property
    def claude_user_question_transport(self) -> ClaudeUserQuestionTransportProtocol | None: ...

    @property
    def session_state_reader(self) -> SessionStateReaderProtocol: ...


@runtime_checkable
class SessionStoreProtocol(Protocol):
    """Interface for session store operations used by adapters."""

    def get_or_create(
        self,
        *,
        session_id: str,
        user_id: int | None = None,
        provider: str = "claude_code",
        workdir: str = ".",
        terminal_id: str | None = None,
        claude_session_id: str | None = None,
    ) -> SessionState: ...

    def get(self, session_id: str) -> SessionState | None: ...

    def process(self, event: SessionEvent) -> SessionState: ...

    def mark_interactive_turn_processing(
        self,
        *,
        terminal_id: str | None,
        workdir: str,
        claude_session_id: str | None = None,
        fallback_session_id: str | None = None,
    ) -> SessionState | None: ...

    def latest_completed_assistant_turn_id(
        self,
        *,
        terminal_id: str | None,
        workdir: str,
        claude_session_id: str | None = None,
        fallback_session_id: str | None = None,
    ) -> str | None: ...

    def resolve_interactive_session_id(
        self,
        *,
        terminal_id: str | None,
        claude_session_id: str | None = None,
        fallback_session_id: str | None = None,
        require_claude_session: bool = False,
    ) -> str | None: ...

    def get_interactive_state(
        self,
        *,
        terminal_id: str | None,
        workdir: str,
        claude_session_id: str | None = None,
        fallback_session_id: str | None = None,
        require_claude_session: bool = False,
    ) -> SessionState | None: ...

    def interactive_completion_phase(
        self,
        *,
        terminal_id: str | None,
        workdir: str,
        claude_session_id: str | None = None,
        fallback_session_id: str | None = None,
    ) -> SessionPhase | None: ...
