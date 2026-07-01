"""终端渲染器 — 基于 Rich 库的格式化输出。

所有渲染方法遵循"静默失败"原则：
Rich 渲染异常时自动回退到纯文本 print，绝不向上抛出异常中断 REPL。
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

if TYPE_CHECKING:
    from miaowa.core.config import UIConfig
    from miaowa.core.types import ProjectCache

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
# Renderer
# ---------------------------------------------------------------------------


class Renderer:
    """终端渲染器 — 基于 Rich 库提供统一的格式化输出。

    所有公开方法都遵循"静默失败"原则：
    当 Rich 渲染抛出异常时，自动回退到 ``print()`` 纯文本输出，
    确保渲染错误不会中断 REPL 循环。

    Attributes:
        config: UI 配置（主题、语法高亮样式等）。
        console: Rich Console 实例。
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
    # 流式输出（打字机效果）
    # ------------------------------------------------------------------

    def render_stream_chunk(self, chunk: str) -> None:
        """渲染流式输出的文本块。

        采用即时输出策略：每个 chunk 直接写入终端并立即 flush，
        实现打字机逐字输出效果。每累积 ``_STREAM_FLUSH_THRESHOLD``
        chars或遇到换行时强制刷新，确保输出及时可见。

        Args:
            chunk: LLM 流式响应的单个文本增量。
        """
        try:
            # 过滤 ANSI 转义序列，防止 LLM 输出中的终端注入
            chunk = _ANSI_RE.sub("", chunk)
            self.console.print(chunk, end="", markup=False, highlight=False)
            self._flush_counter += len(chunk)

            # 累积到阈值或遇到换行时刷新
            if (
                self._flush_counter >= _STREAM_FLUSH_THRESHOLD
                or "\n" in chunk
            ):
                self.console.file.flush()
                self._flush_counter = 0
        except Exception:
            _fallback_print(chunk)
            self._flush_counter = 0

    def flush_stream(self) -> None:
        """强制刷新流式输出缓冲区。

        应在流式响应结束后调用，确保所有缓冲内容写入终端。
        """
        try:
            self.console.file.flush()
        except Exception:
            pass
        finally:
            self._flush_counter = 0

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

        Returns:
            终端宽度；若无法获取则默认 80。
        """
        try:
            return self.console.width or 80
        except Exception:
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
    回退到最后一个 ``\\n\\n`` 或 ``\\n``` `` 边界；
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

    # 其次回退到代码块结束边界
    last_fence = truncated.rfind("\n```")
    if last_fence > max_length // 2:
        return truncated[:last_fence] + "\n```\n...(truncated)"

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
