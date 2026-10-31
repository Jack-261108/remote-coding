# 外部终端输入支持 kitty

## Summary

把现有「外部 Ghostty 会话输入注入」链路中的终端部分抽象为可插拔后端，新增 kitty 后端，使直接运行在 kitty 中、未经 tmux 的已绑定外部 Claude 会话也能从 Telegram 注入输入并回写 AskUserQuestion。

## Goals

- 引入 `ExternalTerminalBackend` 协议与通用 `ExternalTerminal` / `ExternalTerminalTarget` 模型，Ghostty 与 kitty 各实现一份后端。
- kitty 通过 `kitten @` 远程控制（`allow_remote_control` + `listen_on` socket）实现：枚举窗口、按 window id 校验、文本注入、AskUserQuestion 动作。
- 配对仍是用户从候选列表显式选择；kitty 额外用 binding PID 做候选排序辅助与错选拦截。
- 持久化兼容：旧的 `external_bindings.json`（`ghostty_target` 键、无 `kind` 字段）可正常加载为 ghostty 目标。
- 既有 Ghostty 行为、tmux managed 会话链路、外部回复推送与权限审批全部保持不变。

## Non-goals

- 不支持 kitty 之外的其它终端模拟器（Terminal.app、iTerm2、WezTerm 等）。
- 不实现「按 PID 自动配对」；PID 只用于候选排序，不替代用户显式选择。
- 不支持多 kitty 实例（只接一个 `listen_on` 地址）。
- 不通过 kitty 读取屏幕内容、scrollback 或命令输出。
- 不改造 `TmuxRunner.reveal_terminal`（managed tmux 会话「打开桌面终端」仍是 Terminal.app）。
- 不为 kitty 引入新的配对/发送状态机；复用现有 per-session 输入锁、in-flight 标记、排队与 drain。

## Context

### 现有链路

`docs/specs/2026-08-03-external-ghostty-input-design.md` 建立了完整链路，终端相关部分写死 Ghostty：

| 位置 | 现状 |
|---|---|
| `app/adapters/process/ghostty_terminal_adapter.py` | `GhosttyTerminalAdapter`，AppleScript 固定脚本 + argv 传参 |
| `app/domain/external_session_models.py` | `GhosttyInputTarget`；`ExternalBinding.ghostty_target` |
| `app/domain/user_question_models.py` | `ExternalGhosttyQuestionTarget(kind: Literal["ghostty"])` |
| `app/services/external_binding_store.py` | `set_ghostty_target` / `clear_ghostty_target`，持久化键 `ghostty_target` |
| `app/services/external_session_input_service.py` | 直接持有 `GhosttyTerminalAdapter`，17 处 `InjectionOutcome` 分支 |
| `app/services/user_question_callback_registry.py` | `UserQuestionCallbackOrigin.EXTERNAL_GHOSTTY` |
| `app/bot/handlers/session_actions.py` | 文案写死「Ghostty 终端」 |

### kitty 能力（本机 0.49.2 + 官方文档/源码核对）

- 无 AppleScript 字典，唯一控制通道是 `kitten @` 远程控制；服务进程在 kitty 之外，必须 `--to unix:<socket>` 显式寻址（`KITTY_LISTEN_ON` 环境变量只对 kitty 子进程有效）。
- `kitten @ ls` 输出 JSON 树：`[{id, tabs:[{id, windows:[{id, title, cwd, pid, cmdline, foreground_processes:[{pid, cmdline, cwd}], is_focused, is_active}]}]}]`。window id 为整数、单调递增不复用。
- `kitten @ send-text --stdin` 从 stdin 读**原始字节**，不解释 Python 转义（argv 形式会解释 `\e`、`⇺`），因此 payload 必须走 stdin。
- `kitten @ send-key --match id:<n> enter|up|down|right` 发送按键。
- **关键约束**：`send-text` / `send-key` 不回报匹配失败，窗口不存在时同样退出 0。所以错误分类不能依赖它们；必须紧邻注入前重新 `validate_terminal`，并把窗口在极窄窗口期内关闭视为已知静默丢弃风险。

## Proposed design

### 1. 通用模型（`app/domain/external_terminal_models.py` 新建）

```python
TerminalKind = Literal["ghostty", "kitty"]

class TerminalOutcome:            # 原 InjectionOutcome，值保持稳定
    OK = "ok"
    NOT_FOUND = "not_found"
    NOT_UNIQUE = "not_unique"
    # ghostty 专属
    APPLESCRIPT_DISABLED = "applescript_disabled"
    GHOSTTY_NOT_RUNNING = "ghostty_not_running"
    NON_DARWIN = "non_darwin"
    OSASCRIPT_MISSING = "osascript_missing"
    TCC_DENIED = "tcc_denied"
    # kitty 专属
    KITTY_NOT_RUNNING = "kitty_not_running"                  # socket 连不上
    KITTY_REMOTE_CONTROL_DISABLED = "kitty_remote_control_disabled"  # 未配置 listen_on
    KITTY_BIN_MISSING = "kitty_bin_missing"
    # 通用
    TIMEOUT = "timeout"
    INDETERMINATE = "indeterminate"
    OS_ERROR = "os_error"

@dataclass(frozen=True, slots=True)
class ExternalTerminal:
    kind: TerminalKind
    terminal_id: str            # ghostty: surface UUID；kitty: window id 的十进制字符串
    name: str | None
    cwd: str | None
    pid: int | None = None               # 仅 kitty 提供
    foreground_pids: tuple[int, ...] = ()  # 仅 kitty 提供，用于 PID 辅助排序
```

`GhosttyTerminal` 与 `InjectionOutcome` 删除，全仓改名为 `ExternalTerminal` / `TerminalOutcome`。

### 2. 后端协议（`app/domain/protocols.py`）

```python
@runtime_checkable
class ExternalTerminalBackendProtocol(Protocol):
    kind: TerminalKind
    def is_available(self) -> bool: ...
    async def list_terminals(self) -> tuple[list[ExternalTerminal] | None, str | None]: ...
    async def validate_terminal(self, terminal_id: str) -> tuple[bool, ExternalTerminal | None, str | None]: ...
    async def inject_text(self, terminal_id: str, text: str) -> str: ...
    async def select_user_question_option(self, terminal_id: str, *, option_count: int, option_index: int, submit_after: bool) -> str: ...
    async def answer_user_question_with_text(self, terminal_id: str, *, option_count: int, text: str, submit_after: bool) -> str: ...
    async def advance_user_question_after_multi_select(self, terminal_id: str, *, option_count: int, final_question: bool) -> str: ...
```

`GhosttyTerminalAdapter` 实现该协议（逻辑不变，仅把返回值类型换成 `ExternalTerminal(kind="ghostty", ...)`）。

### 3. kitty 后端（`app/adapters/process/kitty_terminal_adapter.py` 新建）

构造参数：`kitten_bin="kitten"`、`listen_on=""`、`timeout_sec=5.0`、`key_delay_sec=0.05`、`platform_name=None`、`enable_remote_control=True`。

- **地址解析**：`listen_on` 非空则用之；否则回退 `os.environ.get("KITTY_LISTEN_ON")`。仍为空 → 不可用，原因 `KITTY_REMOTE_CONTROL_DISABLED`。
- **二进制解析**：`shutil.which(kitten_bin)`；默认名 `kitten` 且 darwin 上找不到时回退 `/Applications/kitty.app/Contents/MacOS/kitten`（与 osascript 回退同一模式）。
- `is_available()` = `enable_remote_control` 且地址可解析 且 二进制可解析。不做联网探测（socket 可达性只在调用时判定）。
- `list_terminals()`：`kitten @ --to <addr> ls`，JSON 解析并拍平 os_windows → tabs → windows，`terminal_id=str(window["id"])`，`pid=window.get("pid")`，`foreground_pids` 取 `foreground_processes[*].pid`。解析失败/超时/非 0 → `(None, reason)`。
- `validate_terminal()`：list 后按 `terminal_id` 精确匹配，必须恰好 1 个，否则 `NOT_FOUND` / `NOT_UNIQUE`（与 Ghostty 同一契约）。
- `inject_text()`：`send-text --match id:<n> --stdin`（payload 走 stdin，原始字节）→ 成功后再 `send-key --match id:<n> enter`。第一步失败按分类表返回；第一步成功、第二步失败 → `INDETERMINATE`（文本可能已粘贴，禁止重试）。
- AskUserQuestion：逐键调用 `send-key --match id:<n> <key>`，每次之间 `await asyncio.sleep(key_delay_sec)`，对齐 Ghostty 脚本的 `delay 0.05` 语义。
  - `select`：`up` × (option_count+1) → `down` × option_index → `enter`；`submit_after` 时再 `enter`。
  - `answer_text`：`down` × option_count → `enter` → `send-text --stdin <text>` → `enter`。
  - `advance_multi`：`right`，`final_question` 时再 `enter`。
  - 首个键发送前失败 → 归类为预检失败；首个键之后任何失败 → `INDETERMINATE`。
- 超时与 kill 复用 Ghostty 适配器的写法（`wait_for` + kill + 有界 reap）。

**失败分类**（基于 stderr 文本 + 退出码）：

| 特征 | 结果 |
|---|---|
| `Failed to connect` / `no such file or directory`（socket） | `KITTY_NOT_RUNNING` |
| `Permission denied` / `allow_remote_control` | `KITTY_REMOTE_CONTROL_DISABLED` |
| 二进制缺失 | `KITTY_BIN_MISSING` |
| 超时 / 退出码 -1 | `TIMEOUT`（注入路径）/ `INDETERMINATE`（问题动作路径） |
| JSON 解析失败或其它非 0 | `OS_ERROR` |

### 4. 领域模型改造

`app/domain/external_session_models.py`：

```python
@dataclass
class ExternalTerminalTarget:
    kind: TerminalKind          # 新增
    terminal_id: str
    paired_tty: str
    paired_at: datetime
    binding_id: str
    name: str | None = None
    cwd: str | None = None

@dataclass
class ExternalBinding:
    ...
    terminal_target: ExternalTerminalTarget | None = None   # 原 ghostty_target
```

`app/domain/user_question_models.py`：`ExternalGhosttyQuestionTarget` → `ExternalTerminalQuestionTarget(kind: TerminalKind, binding_id, terminal_id, paired_tty, paired_at)`；`ExternalUserQuestionTarget = ExternalTmuxQuestionTarget | ExternalTerminalQuestionTarget`。

### 5. 持久化兼容（`app/services/external_binding_store.py`）

- 写出键改为 `terminal_target`，内容增加 `"kind"`。
- 读取时先取 `terminal_target`，缺失则回退旧键 `ghostty_target`；条目无 `kind` 时默认 `"ghostty"`。畸形条目仍返回 `None`（降级为未配对），不使整份 binding 加载失败。
- 方法改名 `set_terminal_target(..., kind=...)` / `clear_terminal_target(...)`；`set_terminal_target` 额外接受 `kind` 并写入目标。

### 6. 输入服务（`app/services/external_session_input_service.py`）

- 构造参数 `ghostty_adapter` → `backends: dict[TerminalKind, ExternalTerminalBackendProtocol]`；`enabled` 由 `EXTERNAL_INPUT_ENABLED` 控制。
- 新增 `_backend_for(kind) -> backend | None`；所有原 `self._adapter.*` 调用改为 `backend = self._backend_for(target.kind)`，`None` 或 `is_available()` 为假时按原语义返回 `ADAPTER_UNAVAILABLE`。
- `pair_candidates()`：遍历所有已启用后端，合并候选；全部后端不可用 → `ADAPTER_UNAVAILABLE`；合并为空 → `NO_TERMINALS`。排序键：

  ```
  0  binding.pid 命中 terminal.foreground_pids      （kitty 强匹配）
  1  binding.pid == terminal.pid                    （kitty 弱匹配）
  2  terminal.cwd == binding.cwd                    （原有 cwd 提示）
  3  其余
  ```

  排序只在展示层生效，绝不自动选中。
- `register_pair_token` / `consume_pair_token` 携带 `kind`：`PairingTokenRecord` 增加 `terminal_kind` 字段，`consume` 后按该 kind 取后端校验。`terminal_id` 在不同后端可能撞值（kitty 是数字串），kind 是必需消歧项。
- `_RETRYABLE_PREFLIGHT_OUTCOMES` 增加 `KITTY_NOT_RUNNING`、`KITTY_REMOTE_CONTROL_DISABLED`、`KITTY_BIN_MISSING`。
- `send_text` / drain / 问题动作的其余逻辑（锁、in-flight、ABA 校验、排队、通知）完全不动。

### 7. 配置（`app/config/settings.py`）

共享设置改名，并用 `AliasChoices` 保留旧名，避免破坏既有 `.env`：

| 新字段 | 别名（新 → 旧） |
|---|---|
| `external_input_enabled` | `EXTERNAL_INPUT_ENABLED` → `GHOSTTY_INPUT_ENABLED` |
| `external_pairing_token_ttl_sec` | `EXTERNAL_PAIRING_TOKEN_TTL_SEC` → `GHOSTTY_PAIRING_TOKEN_TTL_SEC` |
| `external_input_queue_max_size` | `EXTERNAL_INPUT_QUEUE_MAX_SIZE` → `GHOSTTY_INPUT_QUEUE_MAX_SIZE` |
| `external_input_queue_ttl_sec` | `EXTERNAL_INPUT_QUEUE_TTL_SEC` → `GHOSTTY_INPUT_QUEUE_TTL_SEC` |
| `external_drain_publish_wait_timeout_sec` | `EXTERNAL_DRAIN_PUBLISH_WAIT_TIMEOUT_SEC` → `GHOSTTY_DRAIN_PUBLISH_WAIT_TIMEOUT_SEC` |

kitty 专属新增：

```python
kitty_input_enabled: bool = Field(True, alias="KITTY_INPUT_ENABLED")
kitty_bin: str = Field("kitten", alias="KITTY_BIN")
kitty_listen_on: str = Field("", alias="KITTY_LISTEN_ON")
kitty_timeout_sec: float = Field(5.0, alias="KITTY_TIMEOUT_SEC", gt=0)
kitty_key_delay_sec: float = Field(0.05, alias="KITTY_KEY_DELAY_SEC", gt=0)
```

`ghostty_applescript_enabled` 保持原名（确属 ghostty 专属）。

### 8. 装配（`app/bootstrap.py`）

```python
self.ghostty_adapter = GhosttyTerminalAdapter(enable_applescript=settings.ghostty_applescript_enabled)
self.kitty_adapter = KittyTerminalAdapter(
    kitten_bin=settings.kitty_bin,
    listen_on=settings.kitty_listen_on,
    timeout_sec=settings.kitty_timeout_sec,
    key_delay_sec=settings.kitty_key_delay_sec,
)
backends = {"ghostty": self.ghostty_adapter}
if settings.kitty_input_enabled:
    backends["kitty"] = self.kitty_adapter
```

`bootstrap_base.py` 的 `ghostty_adapter` 注解旁新增 `kitty_adapter`。

### 9. Hook 侧（`app/bootstrap_mixins.py`）

`_try_ghostty_user_question` → `_try_external_terminal_user_question`：读 `binding.terminal_target`，构造 `ExternalTerminalQuestionTarget(kind=target.kind, ...)`，推送 origin 改为 `UserQuestionCallbackOrigin.EXTERNAL_TERMINAL`。

### 10. 用户可见文案

- `_pair_outcome_text`：去掉写死的「Ghostty」，改为「终端」；`ADAPTER_UNAVAILABLE` → 「终端适配器不可用（请检查终端远程控制配置）」。
- 配对候选标题：「选择要配对的终端：」。
- 候选按钮标签追加 PID 命中标记：`配对: <name> · <cwd> · <id尾8位> · PID✓`（仅命中时）。
- 回调前缀 `ghpair:` → `tpair:`（token 生命周期 180s，切换无迁移负担）。
- `UserQuestionCallbackOrigin.EXTERNAL_GHOSTTY` → `EXTERNAL_TERMINAL`（值 `external_ghostty` → `external_terminal`）。`ask:` 回调前缀保持不变。

## Alternatives considered

- **平行 kitty 分支**（保留 `ghostty_target`，另加 `kitty_target` + if/else）：改动局部，但配对、发送、排队、问题动作要复制一份；第三个终端时必然崩坏。放弃。
- **仅文本注入**：不做 AskUserQuestion 回写，kitty 会话里的选择题只能去终端答。范围小但功能残缺，放弃。
- **自动读 `~/.config/kitty/kitty.conf` 的 `listen_on`**：可能含环境变量插值，解析易碎。改为显式配置 + `KITTY_LISTEN_ON` 环境变量回退。
- **按 PID 自动配对**：kitty 能拿到 window pid，但 PID 复用与 cmdline 匹配误差会注入错窗口。改为排序辅助 + 用户显式选择。
- **多 socket 地址**：单实例已覆盖当前需求，多地址只增加部分实例不可达的处理分支。

## Edge cases and risks

| 场景 | 处理 |
|---|---|
| kitty 未配置 `listen_on` | 适配器不可用，`KITTY_REMOTE_CONTROL_DISABLED`；其它链路不受影响 |
| socket 存在但 kitty 已退出 | `KITTY_NOT_RUNNING`，属可重试预检失败，drain 保留队列头并等待 |
| 窗口在 validate 与 inject 之间关闭 | `send-text` 静默成功、消息丢失。已知限制，靠紧邻注入前的 `validate_terminal` 收窄窗口；文档与代码注释标明 |
| kitty window id 被复用 | kitty 保证 id 单调递增不复用；仍以 `validate_terminal` 恰好唯一为唯一寻址依据 |
| 同一 binding 同时配过 ghostty 和 kitty | `terminal_target` 单值，重新配对覆盖旧目标并触发 `invalidate_terminal_target` 与问题回调失效 |
| 旧 `external_bindings.json` | 读 `ghostty_target` 回退、`kind` 缺省 `"ghostty"`；写回时升级为新格式 |
| 两个后端都不可用 | `pair_candidates` 返回 `ADAPTER_UNAVAILABLE`，不误报 `NO_TERMINALS` |
| 问题动作中途失败 | 首键前失败 = 预检失败可重试；首键后失败 = `INDETERMINATE`，不重试 |
| 文本含 `\n`、`\e`、中文 | 走 `--stdin` 原始字节，不做转义解释；`_normalise_text` 仍只做 CRLF→LF |

## Test / acceptance plan

新增：

- `tests/test_kitty_terminal_adapter.py`
  - `is_available`：无 listen_on / 无二进制 / 关闭开关 → False；地址来自 `KITTY_LISTEN_ON` 回退。
  - `ls` JSON 拍平：多 OS window / 多 tab / 多 window，`pid` 与 `foreground_pids` 正确。
  - `validate_terminal`：唯一命中 / 零命中 / 重复 id。
  - `inject_text`：payload 走 stdin（断言 argv 不含文本）、随后发 `enter`；`send-text` 失败分类；`send-text` 成功但 `send-key` 失败 → `INDETERMINATE`。
  - 问题动作：`select` 的 up/down/enter 序列与次数、`answer_text` 走 stdin、`advance_multi` 的 `right`。
  - socket 不可达 / 权限拒绝 / 超时 的 reason 分类。
- `tests/test_external_input_kitty_dispatch.py`
  - 服务按 `target.kind` 派发到 kitty 后端；ghostty 后端不被调用。
  - `pair_candidates` 合并两个后端并按 PID/cwd 排序。
  - 只有 kitty 可用时返回 `NEEDS_PAIRING`；全不可用 → `ADAPTER_UNAVAILABLE`。
- `tests/fakes/kitty.py`：`FakeKittyTerminalAdapter`。

改造：

- `tests/test_ghostty_terminal_adapter.py`、`tests/fakes/ghostty.py`、`tests/fakes/external_input_harness.py`、`tests/test_external_session_input_service.py`、`tests/test_external_binding_ghostty_target.py`、`tests/test_bootstrap_hooks_external_ghostty_question.py`、`tests/test_external_question_lifecycle_invalidation.py`、`tests/test_user_question_service_external_dispatch.py`、`tests/test_external_input_assembly.py`：跟随改名（`InjectionOutcome`→`TerminalOutcome`、`GhosttyTerminal`→`ExternalTerminal`、`ghostty_target`→`terminal_target`、`EXTERNAL_GHOSTTY`→`EXTERNAL_TERMINAL`）。
- 新增持久化兼容用例：含旧 `ghostty_target` 键且无 `kind` 的 JSON 能加载为 ghostty 目标，再次写盘后升级为新键。

验收：

```bash
python -m pytest tests/test_kitty_terminal_adapter.py tests/test_external_input_kitty_dispatch.py -v
python -m pytest -q
./scripts/quality_check.sh
```

人工验收（本机 kitty 0.49.2）：`kitty.conf` 加 `allow_remote_control yes` 与 `listen_on unix:/tmp/kitty-$USER`，重启 kitty；在 kitty 中启动 Claude 并绑定；`/list` 选择该会话 → 出现候选且 Claude 所在窗口排在首位并带 PID✓ → 配对 → 发普通文本，确认进入 Claude TUI；触发 AskUserQuestion，确认 Telegram 按钮能回写。

## Open questions

None
