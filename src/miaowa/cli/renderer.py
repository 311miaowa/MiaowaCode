"""终端渲染器 — 基于 Rich 库的格式化输出。

所有渲染方法遵循"静默失败"原则：
Rich 渲染异常时自动回退到纯文本 print，绝不向上抛出异常中断 REPL。

Phase 2 流式优化 (§3.1):
    - 50ms 缓冲节流（throttle）：累积 chunk 到内部缓冲区，使用 Rich Live 原地刷新
    - 代码块状态机（CodeBlockState）：跟踪 ``` 围栏边界，代码块内禁用 Markdown 渲染
    - 长行截断：超过 3 倍终端宽度的行自动截断并保留 ANSI 颜色码闭合
    - 调试模式（--debug-render）：渲染帧时显示帧率与缓冲区大小叠加层
"""

from __future__ import annotations

import json
import re
import shutil
import time
import unicodedata
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

from miaowa.core.logger import get_logger

if TYPE_CHECKING:
    from miaowa.core.config import UIConfig
    from miaowa.core.types import ProjectCache

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 工具调用参数截断长度（字符数）
_TOOL_ARGS_MAX_LENGTH: int = 200

# 工具结果摘要截断长度（字符数）
_TOOL_RESULT_MAX_LENGTH: int = 300

# 流式输出 flush 间隔（字符数），每累积此数量强制刷新
_STREAM_FLUSH_THRESHOLD: int = 80

# ANSI 转义序列正则（用于过滤 LLM 输出中的终端控制字符）
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# Markdown 渲染最大长度（字符数），超出截断并警告
_MARKDOWN_MAX_LENGTH: int = 50_000

# 代码块渲染最大长度（字符数），超出截断并警告
_CODE_MAX_LENGTH: int = 50_000

# 错误/警告/信息消息最大长度（字符数），超出截断
_MESSAGE_MAX_LENGTH: int = 2_000

# 窄终端阈值（列数），低于此值启用简化渲染
_NARROW_TERMINAL_THRESHOLD: int = 50

# 项目上下文字段截断长度
_PROJECT_CONTEXT_TRUNCATE: int = 200

# -- Phase 2: 流式优化常量 --

# 流式缓冲区刷新节流间隔（秒）
_STREAM_THROTTLE_INTERVAL: float = 0.050  # 50ms

# 长行宽度倍数（相对于终端宽度）
_LONG_LINE_WIDTH_MULTIPLIER: int = 3

# 长行截断后缀
_LONG_LINE_TRUNCATION_SUFFIX: str = "..."

# Live 刷新率上限（帧/秒），仅在 auto_refresh=True 时生效；
# 当前使用手动更新模式（auto_refresh=False），此参数作为显式文档保留。
_LIVE_REFRESH_RATE: int = 20

# 流式缓冲区最大字节数（防止超大输出 OOM），超出时强制刷新
_STREAM_BUFFER_MAX_SIZE: int = 500_000  # 500KB

# ---------------------------------------------------------------------------
# 色彩 / 关键字常量
# ---------------------------------------------------------------------------

_COLORS = {
    "error": "red",
    "warning": "yellow",
    "info": "blue",
}

_MESSAGE_PREFIXES = {
    "error": "Error:",
    "warning": "Warning:",
    "info": "Info:",
}

# ---------------------------------------------------------------------------
# 代码块状态机（Phase 2 §3.1.2）
# ---------------------------------------------------------------------------


class _CodeBlockStateEnum(Enum):
    """代码块状态机内部状态枚举。"""
    NORMAL = auto()
    IN_CODE_BLOCK = auto()


class CodeBlockState:
    """轻量级代码块状态机 — 跟踪围栏代码块的开启/关闭状态。

    用于流式输出场景中正确识别 `` ``` `` 和 `` ~~~ `` 围栏边界，
    确保代码块内的内容不被 Markdown 渲染器误解析。

    支持 CommonMark 围栏规范：
        - 3 个及以上反引号（`` ``` ``）或波浪线（`` ~~~ ``）
        - 开启围栏支持语言标识符（如 `` ```python ``）
        - 关闭围栏匹配相同字符类型，长度 ≥ 开启围栏长度

    状态转换::

        NORMAL ── ```lang / ~~~lang ──> IN_CODE_BLOCK
        IN_CODE_BLOCK ── ``` / ~~~ ──> NORMAL

    Attributes:
        state: 当前状态（NORMAL 或 IN_CODE_BLOCK）。
        language: 当前代码块的语言标识符（NORMAL 状态下为空字符串）。
    """

    def __init__(self) -> None:
        self.state: _CodeBlockStateEnum = _CodeBlockStateEnum.NORMAL
        self.language: str = ""
        self._fence_char: str = ""  # '`' 或 '~'
        self._fence_length: int = 0  # 开启围栏的字符数

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def process_line(self, line: str) -> tuple[_CodeBlockStateEnum, str]:
        """处理一行文本，返回 (新状态, 处理后的行)。

        识别 `` ``` `` / `` ~~~ `` 围栏标记并切换状态。
        开启围栏支持语言标识符（如 `` ```python `` 或 `` ~~~rust ``）。
        关闭围栏要求行仅包含 ≥ 开启长度的同类型围栏字符。

        Args:
            line: 单行文本（不含行尾换行符）。

        Returns:
            ``(state, processed_line)`` 元组。
        """
        stripped = line.strip()

        if self.state == _CodeBlockStateEnum.NORMAL:
            fence_info = self._detect_fence_open(stripped)
            if fence_info is not None:
                self.state = _CodeBlockStateEnum.IN_CODE_BLOCK
                self._fence_char, self._fence_length, self.language = fence_info
                return (self.state, line)
            return (self.state, line)
        else:
            if self._is_fence_close(stripped):
                self.state = _CodeBlockStateEnum.NORMAL
                self.language = ""
                self._fence_char = ""
                self._fence_length = 0
                return (self.state, line)
            return (self.state, line)

    def reset(self) -> None:
        """重置状态机到初始 NORMAL 状态。"""
        self.state = _CodeBlockStateEnum.NORMAL
        self.language = ""
        self._fence_char = ""
        self._fence_length = 0

    def is_in_code_block(self) -> bool:
        """返回当前是否处于代码块内部。

        Returns:
            True 当状态为 IN_CODE_BLOCK 时。
        """
        return self.state == _CodeBlockStateEnum.IN_CODE_BLOCK

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_fence_open(stripped: str) -> tuple[str, int, str] | None:
        """检测开启围栏。

        识别以 ≥3 个连续相同反引号或波浪线开头的行，
        提取围栏字符类型、长度和语言标识符。

        Args:
            stripped: 已去除首尾空白的行。

        Returns:
            ``(fence_char, fence_length, language)`` 或 None。
        """
        for char in ("`", "~"):
            if stripped.startswith(char):
                count = len(stripped) - len(stripped.lstrip(char))
                if count >= 3:
                    # 验证前 count 个字符全部为相同围栏字符
                    if all(c == char for c in stripped[:count]):
                        language = stripped[count:].strip()
                        return (char, count, language)
        return None

    def _is_fence_close(self, stripped: str) -> bool:
        """检查当前行是否为匹配的关闭围栏。

        条件：
            1. 行仅包含与开启围栏相同类型的字符
            2. 字符数量 ≥ 开启围栏长度（CommonMark 规范）
            3. 无额外内容（语言标识符等）

        Returns:
            True 当该行为匹配的关闭围栏时。
        """
        if not self._fence_char:
            return False
        if not stripped or not stripped.startswith(self._fence_char):
            return False
        if not all(c == self._fence_char for c in stripped):
            return False
        return len(stripped) >= self._fence_length


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class Renderer:
    """终端渲染器 — 基于 Rich 库提供统一的格式化输出。

    所有公开方法都遵循"静默失败"原则：
    当 Rich 渲染抛出异常时，自动回退到 ``print()`` 纯文本输出，
    确保渲染错误不会中断 REPL 循环。

    Phase 2 流式优化 (§3.1):
        - 50ms 缓冲节流：累积 chunk 到内部缓冲区，按时钟节拍刷新
        - 代码块状态机：跟踪 `` ``` `` 围栏边界，代码块内禁用 Markdown 渲染
        - 长行截断：超过 3 倍终端宽度的行自动截断，保留 ANSI 颜色码闭合
        - 调试模式：通过 ``debug_render`` 属性启停帧率/缓冲区大小叠加层

    Attributes:
        config: UI 配置（主题、语法高亮样式等）。
        console: Rich Console 实例。
        debug_render: 是否启用调试渲染叠加层（可读写属性）。
    """

    def __init__(self, config: UIConfig) -> None:
        """初始化渲染器。

        Args:
            config: UIConfig 实例，提供 theme、syntax_theme 等配置。
        """
        self.config = config
        self.console = Console(
            highlight=True,
            soft_wrap=True,
        )
        # 流式输出 flush chars计数器（每 _STREAM_FLUSH_THRESHOLD chars强制刷新）
        self._flush_counter: int = 0

        # -- Phase 2: 流式优化状态 --
        self._stream_buffer: str = ""
        self._last_flush_time: float = 0.0
        self._throttle_interval: float = _STREAM_THROTTLE_INTERVAL
        self._live: Live | None = None
        self._code_block_state = CodeBlockState()
        self._debug_render: bool = False
        self._frame_count: int = 0
        self._stream_start_time: float = 0.0

        # -- 性能优化：终端宽度缓存 + 缓冲区处理记忆化 --
        self._cached_terminal_width: int = 0
        self._cached_buffer: str = ""
        self._cached_processed: str = ""

    # -- debug_render property -----------------------------------------------

    @property
    def debug_render(self) -> bool:
        """是否启用调试渲染叠加层（帧率 + 缓冲区大小）。"""
        return self._debug_render

    @debug_render.setter
    def debug_render(self, value: bool) -> None:
        self._debug_render = value

    # ------------------------------------------------------------------
    # Markdown 渲染
    # ------------------------------------------------------------------

    def render_markdown(self, text: str) -> None:
        """渲染 Markdown 文本，代码块自动语法高亮。

        使用 Rich Markdown 渲染器，支持：
        - 标题、列表、粗体、斜体等标准 Markdown 语法
        - 围栏代码块自动语法高亮（基于语言标识符）
        - 内联代码高亮

        超长内容（>50,000 chars）将被截断并显示警告。

        Args:
            text: Markdown 格式文本。
        """
        try:
            if len(text) > _MARKDOWN_MAX_LENGTH:
                text = _truncate_markdown(text, _MARKDOWN_MAX_LENGTH)
                self.console.print(
                    Text(
                        f"[WARNING] Markdown too long, truncated to {_MARKDOWN_MAX_LENGTH} chars",
                        style="yellow",
                    )
                )
            md = Markdown(text, code_theme=self.config.syntax_theme)
            self.console.print(md)
        except Exception:
            _fallback_print(text)

    # ------------------------------------------------------------------
    # 流式输出（Phase 2 优化：50ms 节流 + Live 原地刷新）
    # ------------------------------------------------------------------

    def render_stream_chunk(
        self, chunk: str, finish_reason: str | None = None
    ) -> None:
        """渲染流式输出的文本块 — Phase 2 缓冲节流优化。

        与 Phase 1 的即时逐字输出不同，Phase 2 将 chunk
        累积到内部缓冲区，仅在以下条件之一满足时才刷新到终端：

        1. 距上次刷新已超过 50ms throttle 间隔
        2. chunk 中包含换行符（保证行完整性）
        3. 接收到 finish_reason（流终止信号）

        使用 Rich ``Live`` 实现原地更新，避免终端 I/O 抖动。

        Args:
            chunk: LLM 流式响应的单个文本增量。
            finish_reason: 可选，LLM 返回的 finish_reason；
                非 None 时立即刷新缓冲区。
        """
        try:
            # 过滤 ANSI 转义序列，防止 LLM 输出中的终端注入
            chunk = _ANSI_RE.sub("", chunk)
            current_time = time.monotonic()

            # 首个 chunk — 启动 Live 上下文
            if self._live is None:
                try:
                    self._live = Live(
                        console=self.console,
                        auto_refresh=False,
                        transient=False,
                        vertical_overflow="visible",
                        refresh_per_second=_LIVE_REFRESH_RATE,
                    )
                    self._live.start()
                except Exception:
                    # Live 启动失败（如非 TTY 环境），回退到直接打印模式
                    self._live = None
                    self.console.print(
                        chunk, end="", markup=False, highlight=False
                    )
                    self._flush_counter += len(chunk)
                    if (
                        self._flush_counter >= _STREAM_FLUSH_THRESHOLD
                        or "\n" in chunk
                    ):
                        self.console.file.flush()
                        self._flush_counter = 0
                    return

                self._last_flush_time = current_time
                self._stream_start_time = current_time

            # 累积到缓冲区
            self._stream_buffer += chunk

            # 缓冲区上限保护：防止超大 LLM 输出导致 OOM
            if len(self._stream_buffer) > _STREAM_BUFFER_MAX_SIZE:
                logger.warning(
                    "Stream buffer exceeded %d bytes, forcing flush",
                    _STREAM_BUFFER_MAX_SIZE,
                )
                self._do_flush()
                self._last_flush_time = time.monotonic()

            # 判断是否需要刷新
            has_newline = "\n" in chunk
            time_since_flush = current_time - self._last_flush_time
            reached_throttle = time_since_flush >= self._throttle_interval
            is_finish = finish_reason is not None

            if reached_throttle or has_newline or is_finish:
                self._do_flush()
                self._last_flush_time = time.monotonic()

        except Exception:
            _fallback_print(chunk)
            self._flush_counter = 0

    def flush_stream(self) -> None:
        """强制刷新流式输出缓冲区并停止 Live 显示。

        应在流式响应结束后调用，确保：
        - 所有缓冲内容刷新到终端
        - Live 上下文正确关闭（渲染内容持久化到终端）
        - 所有流式状态重置为空闲
        - 异常时通过 _fallback_print 兜底，确保缓冲区内容不丢失
        """
        try:
            self._do_flush()
        except Exception:
            # _do_flush 异常时兜底输出，防止缓冲区内容丢失
            if self._stream_buffer:
                logger.warning(
                    "_do_flush failed, falling back to plain print (%d bytes)",
                    len(self._stream_buffer),
                )
                _fallback_print(self._stream_buffer)
        finally:
            try:
                if self._live is not None:
                    try:
                        self._live.stop()
                    except Exception:
                        pass
            finally:
                self._live = None
                self._stream_buffer = ""
                self._last_flush_time = 0.0
                self._flush_counter = 0
                self._code_block_state.reset()
                self._frame_count = 0
                self._stream_start_time = 0.0
                self._cached_buffer = ""
                self._cached_processed = ""

    # ------------------------------------------------------------------
    # Phase 2: 流式内部方法
    # ------------------------------------------------------------------

    def _do_flush(self) -> None:
        """将累积的流式缓冲区渲染到 Live 显示。

        执行流程:
            1. 通过代码块状态机处理缓冲区（带记忆化缓存）
            2. 对超长行应用截断
            3. 创建 Markdown 可渲染对象
            4. 可选附加调试叠加层
            5. 调用 Live.update() 刷新显示
        """
        if not self._stream_buffer:
            return

        # 处理缓冲区：代码块状态机 + 长行截断（记忆化缓存加速）
        processed = self._process_stream_buffer(self._stream_buffer)

        # 创建 Markdown 可渲染对象
        renderable: Any = Markdown(
            processed, code_theme=self.config.syntax_theme
        )

        # 调试模式叠加层
        if self._debug_render:
            renderable = self._build_debug_overlay(renderable)
            logger.debug(
                "Stream flush: frame=%d buf_size=%d fps=%.1f",
                self._frame_count,
                len(self._stream_buffer),
                self._frame_count / max(time.monotonic() - self._stream_start_time, 0.001),
            )

        # 刷新到 Live 或回退到直接打印
        if self._live is not None:
            try:
                self._live.update(renderable, refresh=True)
                self._frame_count += 1
            except Exception:
                # Live.update 失败时回退到 console.print
                logger.debug("Live.update failed, falling back to console.print")
                self.console.print(renderable)
        else:
            self.console.print(renderable)

    def _process_stream_buffer(self, text: str) -> str:
        """处理流式缓冲区：逐行应用代码块状态机和长行截断。

        将整个缓冲区分行处理，通过 CodeBlockState 跟踪代码块边界，
        同时对每一行超过 3 倍终端宽度的应用截断。

        记忆化优化：若缓冲区内容未变化（连续 _do_flush 无新 chunk），
        直接返回上次处理结果，避免重复 split/join/状态机遍历。

        Args:
            text: 原始累积的流式缓冲区内容。

        Returns:
            处理后的文本。
        """
        # 记忆化：缓冲区内容未变化时复用缓存结果
        if text == self._cached_buffer:
            return self._cached_processed

        self._code_block_state.reset()
        max_width = self._get_terminal_width() * _LONG_LINE_WIDTH_MULTIPLIER

        # 最小宽度保护：终端宽度检测失败时使用默认值
        if max_width < _NARROW_TERMINAL_THRESHOLD:
            max_width = 80 * _LONG_LINE_WIDTH_MULTIPLIER

        lines = text.split("\n")
        processed_lines: list[str] = []

        for line in lines:
            self._code_block_state.process_line(line)
            # 长行截断（对所有行生效）
            truncated = _truncate_line_preserve_ansi(line, max_width)
            processed_lines.append(truncated)

        result = "\n".join(processed_lines)
        self._cached_buffer = text
        self._cached_processed = result
        return result

    def _build_debug_overlay(self, renderable: Any) -> Group:
        """构建调试信息叠加层（帧率 + 缓冲区大小）。

        在渲染内容底部附加一行灰色调试信息，
        仅在 ``debug_render`` 为 True 时调用。

        Args:
            renderable: 主渲染对象（Markdown 实例）。

        Returns:
            包含调试叠加层的 Group 可渲染对象。
        """
        elapsed = time.monotonic() - self._stream_start_time
        fps = self._frame_count / elapsed if elapsed > 0 else 0.0
        buf_size = len(self._stream_buffer)

        debug_text = Text(
            f"  [fps={fps:.1f}] [buf={buf_size}B]",
            style="dim",
        )
        debug_text.justify = "right"

        return Group(renderable, debug_text)

    # ------------------------------------------------------------------
    # 欢迎 / 告别界面
    # ------------------------------------------------------------------

    def render_welcome(
        self,
        model: str | None = None,
        provider: str | None = None,
        project_root: str | None = None,
    ) -> None:
        """渲染启动信息 — 简洁信息列表风格。

        Args:
            model: 当前使用的模型名称。
            provider: LLM 提供商名称。
            project_root: 项目根目录路径。
        """
        from miaowa import __version__

        provider_name = provider or "unknown"
        model_name = model or "not configured"
        project_path = project_root or "."

        try:
            text = Text()
            text.append("Miaowa Code ", style="bold")
            text.append(f"v{__version__}", style="dim")
            self.console.print(text)

            text = Text()
            text.append("  Project  : ", style="dim")
            text.append(project_path)
            self.console.print(text)

            text = Text()
            text.append("  Model    : ", style="dim")
            text.append(f"{provider_name} / {model_name}")
            self.console.print(text)

            self.console.print()
        except Exception:
            _fallback_print(
                f"Miaowa Code v{__version__}\n"
                f"  Project  : {project_path}\n"
                f"  Model    : {provider_name} / {model_name}"
            )

    def render_goodbye(self, session_stats: dict) -> None:
        """渲染退出信息 — 简洁统计摘要。

        Args:
            session_stats: 会话统计字典，包含 duration / tokens / cost / turns。
        """
        try:
            duration = session_stats.get("duration", "unknown")
            tokens = session_stats.get("tokens", {})
            cost = session_stats.get("cost", 0.0)
            turns = session_stats.get("turns", 0)

            self.console.print()
            text = Text()
            text.append("  Turns: ", style="dim")
            text.append(str(turns))
            text.append("  ·  ")
            text.append("Tokens: ", style="dim")
            text.append(f"{tokens.get('total', 0):,}")
            text.append("  ·  ")
            text.append("Cost: ", style="dim")
            text.append(f"¥{_safe_format_cost(cost)}")
            text.append("  ·  ")
            text.append("Duration: ", style="dim")
            text.append(str(duration))
            self.console.print(text)
            self.console.print()
        except Exception:
            _fallback_print(
                f"Session ended | turns={session_stats.get('turns', '?')} "
                f"cost=¥{_safe_format_cost(session_stats.get('cost', 0))}"
            )

    # ------------------------------------------------------------------
    # 工具调用渲染
    # ------------------------------------------------------------------

    def render_tool_call(self, tool_name: str, tool_args: dict) -> None:
        """渲染工具调用信息 — 单行简洁风格。

        Args:
            tool_name: 工具名称。
            tool_args: 工具参数字典。
        """
        try:
            args_text = _format_json_compact(tool_args, _TOOL_ARGS_MAX_LENGTH)
            text = Text()
            text.append("● ", style="dim")
            text.append(tool_name, style="bold")
            text.append("  ", style="dim")
            text.append(args_text, style="dim")
            self.console.print(text)
        except Exception:
            _fallback_print(f"● {tool_name} {json.dumps(tool_args, ensure_ascii=False)}")

    def render_tool_result(self, success: bool, summary: str) -> None:
        """渲染工具执行结果 — 单行简洁风格。

        Args:
            success: 工具是否执行成功。
            summary: 结果摘要文本。
        """
        try:
            truncated = _truncate_text(summary, _TOOL_RESULT_MAX_LENGTH)
            text = Text()
            if success:
                text.append("  ", style="dim")
            else:
                text.append("  ! ", style="red")
            text.append(truncated, style="dim")
            self.console.print(text)
        except Exception:
            status = "OK" if success else "FAIL"
            _fallback_print(f"  {status}: {summary}")

    # ------------------------------------------------------------------
    # 错误 / 警告 / 信息（统一消息渲染）
    # ------------------------------------------------------------------

    def render_error(self, message: str) -> None:
        """渲染错误消息（红色 Panel）。

        Args:
            message: 错误描述文本。超长时自动截断。
        """
        self._render_message(message, "error")

    def render_warning(self, message: str) -> None:
        """渲染警告消息（黄色 Panel）。

        Args:
            message: 警告描述文本。超长时自动截断。
        """
        self._render_message(message, "warning")

    def render_info(self, message: str) -> None:
        """渲染信息消息（蓝色 Panel）。

        Args:
            message: 信息描述文本。超长时自动截断。
        """
        self._render_message(message, "info")

    # ------------------------------------------------------------------
    # Token 用量 & 费用
    # ------------------------------------------------------------------

    def render_token_usage(self, usage: dict, cost: float) -> None:
        """渲染 Token 用量与费用 — 紧凑单行。

        输出格式::

            Tokens: 1,234 in + 567 out = 1,801  ·  ¥0.0123

        Args:
            usage: Token 用量字典（prompt / completion / total）。
            cost: API 调用费用（人民币元）。
        """
        try:
            prompt = usage.get("prompt", 0)
            completion = usage.get("completion", 0)
            total = usage.get("total", prompt + completion)

            text = Text()
            text.append("  Tokens: ", style="dim")
            text.append(f"{total:,}")
            text.append("  ·  ", style="dim")
            text.append("Cost: ", style="dim")
            text.append(f"¥{_safe_format_cost(cost)}")
            self.console.print(text)
        except Exception:
            _fallback_print(
                f"Tokens: {usage.get('total', 0) if isinstance(usage, dict) else 0:,}"
                f"  ·  ¥{_safe_format_cost(cost)}"
            )

    # ------------------------------------------------------------------
    # 项目上下文
    # ------------------------------------------------------------------

    def render_project_context(self, cache: ProjectCache) -> None:
        """渲染项目分析上下文 — 简洁列表风格。

        Args:
            cache: ProjectCache 实例，包含 tech_stack / structure / key_files。
        """
        try:
            text = Text()
            text.append("── Project context ──", style="dim")
            self.console.print(text)

            if cache.tech_stack:
                tech_items = [f"{k}: {v}" for k, v in cache.tech_stack.items()]
                tech_str = _truncate_text(", ".join(tech_items), _PROJECT_CONTEXT_TRUNCATE)
                line = Text()
                line.append("  Stack: ", style="dim")
                line.append(tech_str)
                self.console.print(line)

            if cache.structure:
                counts_parts = []
                module_count = cache.structure.get("module_count")
                file_count = cache.structure.get("file_count")
                if module_count is not None:
                    counts_parts.append(f"{module_count} modules")
                if file_count is not None:
                    counts_parts.append(f"{file_count} files")
                if counts_parts:
                    line = Text()
                    line.append("  Structure: ", style="dim")
                    line.append("，".join(counts_parts))
                    self.console.print(line)

                if "tree" in cache.structure:
                    tree_text = str(cache.structure["tree"])
                    tree_truncated = _truncate_text(tree_text, _PROJECT_CONTEXT_TRUNCATE)
                    line = Text()
                    line.append(tree_truncated, style="dim")
                    self.console.print(line)

            if cache.key_files:
                files_str = _truncate_text(", ".join(cache.key_files), _PROJECT_CONTEXT_TRUNCATE)
                line = Text()
                line.append("  Key files: ", style="dim")
                line.append(files_str)
                self.console.print(line)
        except Exception:
            _fallback_print(
                f"[PROJECT] tech_stack={cache.tech_stack}, "
                f"key_files={cache.key_files}"
            )

    # ------------------------------------------------------------------
    # 代码语法高亮（独立片段）
    # ------------------------------------------------------------------

    def render_code(self, code: str, language: str = "python") -> None:
        """渲染独立的代码块（带语法高亮和行号）。

        用于工具调用返回的源码片段展示。
        超长代码（>50,000 chars）将被截断并显示警告。

        Args:
            code: 源代码文本。
            language: 编程语言标识符（如 "python"、"javascript"、"rust"）。
        """
        try:
            if len(code) > _CODE_MAX_LENGTH:
                code = _truncate_code(code, _CODE_MAX_LENGTH)
                self.console.print(
                    Text(
                        f"[WARNING] Code too long, truncated to {_CODE_MAX_LENGTH} chars",
                        style="yellow",
                    )
                )
            syntax = Syntax(
                code,
                language,
                theme=self.config.syntax_theme,
                line_numbers=True,
                word_wrap=True,
            )
            self.console.print(syntax)
        except Exception:
            _fallback_print(code)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_terminal_width(self) -> int:
        """获取当前终端宽度（列数）。

        尝试顺序:
            1. ``self.console.width`` — Rich 实时检测结果（优先，自动响应 SIGWINCH）
            2. 缓存值 — shutil 兜底路径的缓存（避免重复系统调用）
            3. ``shutil.get_terminal_size()`` — 跨平台终端尺寸检测
            4. 默认值 80 — 所有检测均失败时的最后防线

        Rich 路径不经过缓存，确保终端 resize 时截断阈值实时更新。
        仅在 Rich 检测失败时才使用 shutil 并缓存其结果。

        Returns:
            终端宽度；若无法获取则默认 80。
        """
        # Rich 路径：不缓存（Rich 内部维护 SIGWINCH 响应）
        try:
            if self.console.width and self.console.width > 0:
                return self.console.width
        except Exception:
            pass

        # shutil 兜底路径：结果缓存
        if self._cached_terminal_width > 0:
            return self._cached_terminal_width

        try:
            size = shutil.get_terminal_size()
            if size.columns > 0:
                self._cached_terminal_width = size.columns
                return self._cached_terminal_width
        except Exception:
            pass

        self._cached_terminal_width = 80
        return 80

    def _render_message(self, message: str, level: str) -> None:
        """统一的消息渲染方法 — 简洁前缀风格（无 Panel 边框）。

        Args:
            message: 消息文本。
            level: 消息级别，取 "error"、"warning"、"info" 之一。
        """
        if not message:
            return

        if len(message) > _MESSAGE_MAX_LENGTH:
            message = message[:_MESSAGE_MAX_LENGTH] + "…"

        style = _COLORS[level]
        prefix = _MESSAGE_PREFIXES.get(level, level)

        try:
            text = Text()
            text.append(prefix, style=f"bold {style}")
            text.append(" ")
            text.append(message, style=style)
            self.console.print(text)
        except Exception:
            _fallback_print(f"{prefix} {message}")


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


def _display_width(text: str) -> int:
    """计算文本在终端中的显示宽度（列数）。

    CJK 字符（East Asian Wide/Fullwidth）占 2 列，其他字符占 1 列。
    用于精确的终端宽度截断判断。

    Args:
        text: 待计算宽度的文本（应为去除 ANSI 转义序列后的纯文本）。

    Returns:
        终端显示列数。
    """
    total = 0
    for ch in text:
        w = unicodedata.east_asian_width(ch)
        total += 2 if w in ("W", "F") else 1
    return total


def _strip_ansi(text: str) -> str:
    """移除字符串中的所有 ANSI 转义序列，返回纯可见文本。

    用于精确计算文本在终端中的显示宽度（如长行截断判断）。

    Args:
        text: 可能包含 ANSI 转义序列的字符串。

    Returns:
        不含任何 ANSI 转义序列的纯文本。
    """
    return _ANSI_RE.sub("", text)


def _truncate_line_preserve_ansi(line: str, max_width: int) -> str:
    """截断超长行，保留 ANSI 颜色码并正确闭合。

    按终端显示宽度计算是否需要截断（CJK 字符计 2 列）：
    若可见部分显示宽度 ≤ max_width 则原样返回；
    否则在 max_width 处截断可见字符，保留截断点
    之前的所有 ANSI 转义序列，追加 ``\\x1b[0m`` 重置颜色 +
    截断后缀 ``...``。

    实现方式：逐字符遍历原始行，跟踪 ANSI 转义序列的起止边界，
    按终端显示宽度累加可见字符，达到 max_width 时停止。

    Args:
        line: 原始行文本（可能包含 ANSI 颜色码）。
        max_width: 最大终端显示列数。

    Returns:
        截断后的行文本，ANSI 序列已正确闭合。
    """
    visible = _strip_ansi(line)
    if _display_width(visible) <= max_width:
        return line

    result_chars: list[str] = []
    vis_width = 0
    i = 0
    n = len(line)

    while i < n and vis_width < max_width:
        ch = line[i]
        if ch == "\x1b":
            # 定位 ANSI 转义序列的完整范围
            j = i
            while j < n and line[j] != "m":
                j += 1
            # 包含终止字符（通常为 'm'，也可能是其他字母）
            if j < n:
                # 跳过直到字母或 ~（完整转义序列）
                seq_end = j
                while seq_end < n and not (
                    line[seq_end].isalpha() or line[seq_end] == "~"
                ):
                    seq_end += 1
                if seq_end < n:
                    seq_end += 1  # 包含终止字符
                result_chars.append(line[i:seq_end])
                i = seq_end
            else:
                # 异常：转义序列未闭合，原样保留
                result_chars.append(line[i:])
                break
        else:
            result_chars.append(ch)
            vis_width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            i += 1

    # 重置 ANSI 属性以闭合所有已开启的颜色码
    result_chars.append("\x1b[0m")
    result_chars.append(_LONG_LINE_TRUNCATION_SUFFIX)
    return "".join(result_chars)


def _fallback_print(text: str) -> None:
    """Rich 渲染失败时的纯文本回退输出。

    Args:
        text: 要输出的纯文本。
    """
    try:
        print(text)
    except Exception:
        # 极端情况：print 也失败了（如管道断开），静默忽略
        pass


def _safe_format_cost(cost: Any) -> str:
    """安全地格式化费用值为 4 位小数的人民币字符串。

    即使传入非数字类型也不会崩溃，确保 fallback 路径的安全性。

    Args:
        cost: 费用值（期望为 float/int，但不做类型假设）。

    Returns:
        格式化后的字符串，如 "0.0123"；无法格式化时回退到 str()。
    """
    try:
        return f"{float(cost):,.4f}"
    except (ValueError, TypeError):
        return str(cost)


def _format_json_compact(data: dict, max_length: int = _TOOL_ARGS_MAX_LENGTH) -> str:
    """将字典格式化为紧凑 JSON chars串，超长时截断。

    Args:
        data: 待格式化的字典。
        max_length: 最大字符长度，超出后截断并添加 "…"。

    Returns:
        格式化的 JSON chars串。
    """
    try:
        formatted = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        formatted = str(data)

    if len(formatted) > max_length:
        formatted = formatted[:max_length] + "…"
    return formatted


def _truncate_text(text: str, max_length: int) -> str:
    """截断文本，超长时添加省略号。

    Args:
        text: 原始文本。
        max_length: 最大字符长度。

    Returns:
        截断后的文本。若未超出 max_length 则原样返回；
        超出时截断并附加 "…"。
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "…"


def _truncate_markdown(text: str, max_length: int) -> str:
    """截断 Markdown 文本，尽可能在段落/代码块边界处断开。

    避免字符级硬截断破坏围栏代码块、表格等 Markdown 结构。
    回退到最后一个 ``\\n\\n`` 或 ``\\n``` `` / ``\\n~~~ `` 边界；
    若找不到合适的边界则退回字符级截断。

    Args:
        text: Markdown 格式文本。
        max_length: 最大字符长度。

    Returns:
        截断后的文本，末尾附加截断提示。
    """
    if len(text) <= max_length:
        return text

    # 在截断点附近查找安全边界
    truncated = text[:max_length]

    # 优先回退到段落边界（双换行）
    last_para = truncated.rfind("\n\n")
    if last_para > max_length // 2:
        return truncated[:last_para] + "\n\n...(truncated)"

    # 其次回退到代码块结束边界（反引号或波浪线围栏）
    last_fence_backtick = truncated.rfind("\n```")
    last_fence_tilde = truncated.rfind("\n~~~")
    last_fence = max(last_fence_backtick, last_fence_tilde)
    if last_fence > max_length // 2:
        fence_marker = truncated[last_fence + 1 : last_fence + 4]  # ``` 或 ~~~
        return truncated[:last_fence] + f"\n{fence_marker}\n...(truncated)"

    # 无合适边界，字符级截断
    return truncated + "\n\n...(truncated)"


def _truncate_code(code: str, max_length: int) -> str:
    """截断代码文本，在行边界处断开以避免破坏语法高亮。

    与 Markdown 截断不同，代码截断更关注保留完整行，
    避免在字符串字面量或表达式中间切断导致 Pygments lexer 错乱。

    Args:
        code: 源代码文本。
        max_length: 最大字符长度。

    Returns:
        截断后的代码文本，末尾附加截断提示。
    """
    if len(code) <= max_length:
        return code

    truncated = code[:max_length]

    # 回退到最后一个完整行
    last_newline = truncated.rfind("\n")
    if last_newline > max_length // 2:
        return truncated[:last_newline] + "\n(truncated)"

    # 无合适行边界，字符级截断
    return truncated + "\n(truncated)"
