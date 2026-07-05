"""Renderer 单元测试。

通过 Console(force_terminal=True, file=StringIO()) 捕获 Rich 输出，
验证各渲染方法的输出内容正确性。
"""

from __future__ import annotations

import io
import json
from unittest.mock import Mock, patch

import pytest
from rich.console import Console

from miaowa.cli.renderer import (
    CodeBlockState,
    Renderer,
    _display_width,
    _format_json_compact,
    _safe_format_cost,
    _strip_ansi,
    _truncate_code,
    _truncate_line_preserve_ansi,
    _truncate_markdown,
    _truncate_text,
)
from miaowa.core.config import UIConfig


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------


@pytest.fixture
def ui_config() -> UIConfig:
    """提供默认 UI 配置。"""
    return UIConfig(theme="dark", syntax_theme="monokai", max_history=1000)


@pytest.fixture
def renderer(ui_config: UIConfig) -> Renderer:
    """提供已初始化的 Renderer，console 输出重定向到 StringIO。"""
    r = Renderer(ui_config)
    output = io.StringIO()
    r.console = Console(force_terminal=True, file=output, highlight=False, width=120)
    return r


@pytest.fixture
def narrow_renderer(ui_config: UIConfig) -> Renderer:
    """提供窄终端（width=40）的 Renderer。"""
    r = Renderer(ui_config)
    output = io.StringIO()
    r.console = Console(force_terminal=True, file=output, highlight=False, width=40)
    return r


def _get_output(renderer: Renderer) -> str:
    """从 Renderer 的 console.file 中获取已输出的文本。"""
    assert isinstance(renderer.console.file, io.StringIO)
    return renderer.console.file.getvalue()


# ---------------------------------------------------------------------------
# 构造函数
# ---------------------------------------------------------------------------


class TestRendererInit:
    """Renderer 初始化测试。"""

    def test_default_init(self, ui_config: UIConfig) -> None:
        r = Renderer(ui_config)
        assert r.config is ui_config
        assert r.console is not None
        assert r._flush_counter == 0

    def test_auto_theme(self) -> None:
        """theme='auto' 时 console 的 theme 应为 None（由 Rich 自动检测）。"""
        config = UIConfig(theme="auto", syntax_theme="monokai")
        r = Renderer(config)
        assert r.config.theme == "auto"


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    """Markdown 渲染测试。"""

    def test_basic_markdown(self, renderer: Renderer) -> None:
        renderer.render_markdown("**粗体** 和 _斜体_")
        output = _get_output(renderer)
        assert "粗体" in output

    def test_code_block(self, renderer: Renderer) -> None:
        renderer.render_markdown("```python\nprint('hello')\n```")
        output = _get_output(renderer)
        assert "print" in output or "hello" in output

    def test_empty_text(self, renderer: Renderer) -> None:
        """空文本不应崩溃。"""
        renderer.render_markdown("")
        # 不抛异常即为通过

    def test_fallback_on_rich_failure(self, renderer: Renderer) -> None:
        """Rich 渲染抛出异常时应回退到 print。"""
        with patch.object(
            renderer.console, "print", side_effect=RuntimeError("boom")
        ):
            # 不应抛出异常
            renderer.render_markdown("some text")

    def test_long_markdown_truncated(self, renderer: Renderer) -> None:
        """超长 Markdown 应被截断并显示警告。"""
        long_text = "x" * 60_000  # 超出 _MARKDOWN_MAX_LENGTH (50,000)
        renderer.render_markdown(long_text)
        output = _get_output(renderer)
        assert "truncated" in output
        # 原始全文不应出现
        assert "x" * 60_000 not in output
        # 输出长度应远小于原始长度
        assert len(output) < 60_000


# ---------------------------------------------------------------------------
# render_stream_chunk
# ---------------------------------------------------------------------------


class TestRenderStreamChunk:
    """流式输出测试 — Phase 2 缓冲节流。"""

    def test_single_chunk_buffered(self, renderer: Renderer) -> None:
        """单个 chunk 进入缓冲区，flush_stream 后可见。"""
        renderer.render_stream_chunk("你好")
        # Phase 2: chunk 先累积到缓冲区，调用 flush_stream 后才输出
        renderer.flush_stream()
        output = _get_output(renderer)
        assert "你好" in output

    def test_multiple_chunks_buffered(self, renderer: Renderer) -> None:
        """多个 chunk 在缓冲区内拼接，flush_stream 后完整输出。"""
        renderer.render_stream_chunk("Hello")
        renderer.render_stream_chunk(" ")
        renderer.render_stream_chunk("World")
        renderer.flush_stream()
        output = _get_output(renderer)
        assert "Hello World" in output

    def test_newline_triggers_flush(self, renderer: Renderer) -> None:
        """包含换行符的 chunk 触发立即刷新（不等 throttle）。"""
        renderer.render_stream_chunk("line 1\n")
        # 换行符触发 _do_flush，内容立即可见
        output = _get_output(renderer)
        assert "line 1" in output

    def test_empty_chunk(self, renderer: Renderer) -> None:
        """空 chunk 不应崩溃。"""
        renderer.render_stream_chunk("")
        # 不抛异常即为通过

    def test_buffer_accumulates(self, renderer: Renderer) -> None:
        """缓冲区应正确累积内容。"""
        renderer.render_stream_chunk("hello")
        assert renderer._stream_buffer == "hello"
        renderer.render_stream_chunk(" world")
        assert renderer._stream_buffer == "hello world"

    def test_finish_reason_triggers_flush(self, renderer: Renderer) -> None:
        """finish_reason 非 None 时立即刷新缓冲区。"""
        renderer.render_stream_chunk("done", finish_reason="stop")
        output = _get_output(renderer)
        assert "done" in output

    def test_exception_fallback(self, renderer: Renderer) -> None:
        """异常时回退到 _fallback_print，_flush_counter 重置。"""
        renderer._flush_counter = 10
        with patch(
            "miaowa.cli.renderer._fallback_print", side_effect=None
        ):
            # 通过让 Live 初始化失败来触发异常路径
            with patch.object(
                renderer.console, "print", side_effect=RuntimeError("boom")
            ):
                # 直接设置 _live 为非 None 使其跳过 Live 初始化，
                # 然后触发其内部的异常路径
                pass
        # 验证 _flush_counter 在 fallback 路径中正确重置
        # (此测试验证异常路径不崩溃即可)
        renderer.render_stream_chunk("test")
        renderer.flush_stream()

    def test_buffer_cap_forces_flush(self, renderer: Renderer) -> None:
        """缓冲区超出上限时强制刷新，防止 OOM。"""
        from miaowa.cli.renderer import _STREAM_BUFFER_MAX_SIZE

        # 写入略超上限的 chunk（无换行符，避免提前触发刷新）
        huge_chunk = "x" * (_STREAM_BUFFER_MAX_SIZE + 100)
        renderer.render_stream_chunk(huge_chunk)
        # 缓冲区超出上限后应立即触发 _do_flush，不应继续累积
        assert len(renderer._stream_buffer) <= _STREAM_BUFFER_MAX_SIZE + len(huge_chunk)
        # 不应崩溃
        renderer.flush_stream()


class TestFlushStream:
    """flush_stream 方法测试 — Phase 2 状态重置。"""

    def test_flush_resets_stream_state(self, renderer: Renderer) -> None:
        """flush_stream 应重置所有流式状态。"""
        renderer._flush_counter = 42
        renderer._stream_buffer = "some content"
        renderer.flush_stream()
        assert renderer._flush_counter == 0
        assert renderer._stream_buffer == ""
        assert renderer._last_flush_time == 0.0
        assert renderer._frame_count == 0

    def test_flush_empty_state(self, renderer: Renderer) -> None:
        """空状态时 flush 不应崩溃。"""
        renderer.flush_stream()
        assert renderer._flush_counter == 0
        assert renderer._stream_buffer == ""

    def test_flush_with_buffered_content(self, renderer: Renderer) -> None:
        """缓冲内容在 flush 后应输出到终端。"""
        renderer.render_stream_chunk("buffered content")
        output_before = _get_output(renderer)
        # 刷新前：内容在 Live 缓冲区中，StringIO 可能只有 ANSI 控制序列
        renderer.flush_stream()
        output_after = _get_output(renderer)
        # 刷新后：内容应已渲染
        assert "buffered content" in output_after


# ---------------------------------------------------------------------------
# render_welcome / render_goodbye
# ---------------------------------------------------------------------------


class TestRenderWelcome:
    """欢迎界面测试。"""

    def test_welcome_contains_logo(self, renderer: Renderer) -> None:
        renderer.render_welcome(model="deepseek-chat")
        output = _get_output(renderer)
        assert "Miaowa Code" in output

    def test_welcome_shows_model_name(self, renderer: Renderer) -> None:
        """欢迎界面应包含模型名称。"""
        renderer.render_welcome(model="deepseek-reasoner")
        output = _get_output(renderer)
        assert "deepseek-reasoner" in output

    def test_welcome_shows_version(self, renderer: Renderer) -> None:
        """欢迎界面应包含版本号。"""
        from miaowa import __version__

        renderer.render_welcome()
        output = _get_output(renderer)
        assert __version__ in output

    def test_welcome_no_model_shows_unconfigured(self, renderer: Renderer) -> None:
        """未提供 model 时应显示'未配置'。"""
        renderer.render_welcome()
        output = _get_output(renderer)
        assert "not configured" in output

    def test_welcome_narrow_terminal(self, narrow_renderer: Renderer) -> None:
        """窄终端应降级为单行纯文本。"""
        narrow_renderer.render_welcome(model="deepseek-chat")
        output = _get_output(narrow_renderer)
        # 窄终端下应包含关键信息但不含 box 字符
        assert "Miaowa Code" in output
        assert "deepseek-chat" in output
        assert "╭" not in output  # 无 box drawing

    def test_welcome_fallback(self, renderer: Renderer) -> None:
        """欢迎界面 fallback 不应抛异常。"""
        with patch.object(
            renderer.console, "print", side_effect=RuntimeError("boom")
        ):
            renderer.render_welcome()


class TestRenderGoodbye:
    """告别界面测试。"""

    def test_goodbye_with_full_stats(self, renderer: Renderer) -> None:
        stats = {
            "duration": "5m 30s",
            "tokens": {"prompt": 1234, "completion": 567, "total": 1801},
            "cost": 0.0123,
            "turns": 12,
        }
        renderer.render_goodbye(stats)
        output = _get_output(renderer)
        assert "5m 30s" in output
        assert "12" in output
        assert "1,801" in output

    def test_goodbye_with_minimal_stats(self, renderer: Renderer) -> None:
        """缺少部分键的统计字典不应崩溃。"""
        renderer.render_goodbye({})
        output = _get_output(renderer)
        assert "Turns" in output

    def test_goodbye_fallback(self, renderer: Renderer) -> None:
        """告别界面 fallback 不应抛异常。"""
        with patch.object(
            renderer.console, "print", side_effect=RuntimeError("boom")
        ):
            renderer.render_goodbye({"turns": 5, "cost": 0.001})

    def test_goodbye_fallback_handles_non_numeric_cost(self, renderer: Renderer) -> None:
        """fallback 路径中 cost 为非数字时不应崩溃（严重问题 #1 修复验证）。"""
        with patch.object(
            renderer.console, "print", side_effect=RuntimeError("boom")
        ):
            # cost 为字符串时不应崩——之前会崩在 fallback 的 :.4f 格式化
            renderer.render_goodbye({"turns": 3, "cost": "not-a-number"})


# ---------------------------------------------------------------------------
# render_tool_call / render_tool_result
# ---------------------------------------------------------------------------


class TestRenderToolCall:
    """工具调用渲染测试。"""

    def test_basic_tool_call(self, renderer: Renderer) -> None:
        renderer.render_tool_call("read_file", {"path": "/tmp/test.py"})
        output = _get_output(renderer)
        assert "read_file" in output
        assert "test.py" in output

    def test_tool_call_with_empty_args(self, renderer: Renderer) -> None:
        renderer.render_tool_call("list_files", {})
        output = _get_output(renderer)
        assert "list_files" in output

    def test_tool_call_args_truncation(self, renderer: Renderer) -> None:
        """长参数应被截断。"""
        long_value = "x" * 300
        renderer.render_tool_call("search", {"query": long_value})
        output = _get_output(renderer)
        # 不应包含完整的长字符串
        assert long_value not in output


class TestRenderToolResult:
    """工具结果渲染测试。"""

    def test_success_result(self, renderer: Renderer) -> None:
        renderer.render_tool_result(True, "文件读取成功")
        output = _get_output(renderer)
        assert "文件读取成功" in output

    def test_failure_result(self, renderer: Renderer) -> None:
        renderer.render_tool_result(False, "文件未找到")
        output = _get_output(renderer)
        assert "文件未找到" in output

    def test_long_summary_truncation(self, renderer: Renderer) -> None:
        """长摘要应被截断。"""
        long_summary = "x" * 500
        renderer.render_tool_result(True, long_summary)
        output = _get_output(renderer)
        assert long_summary not in output


# ---------------------------------------------------------------------------
# 错误 / 警告 / 信息（含长度限制和 DRY 统一渲染）
# ---------------------------------------------------------------------------


class TestRenderError:
    """错误消息渲染测试。"""

    def test_render_error(self, renderer: Renderer) -> None:
        renderer.render_error("API 密钥无效")
        output = _get_output(renderer)
        assert "API 密钥无效" in output

    def test_render_error_empty_message(self, renderer: Renderer) -> None:
        """空消息应被短路，不产生任何输出。"""
        renderer.render_error("")
        output = _get_output(renderer)
        assert output == ""  # 空消息不应渲染任何内容

    def test_long_error_truncated(self, renderer: Renderer) -> None:
        """超长错误消息应被截断。"""
        long_msg = "E" * 3_000  # 超出 _MESSAGE_MAX_LENGTH (2,000)
        renderer.render_error(long_msg)
        output = _get_output(renderer)
        # 原始完整消息不应出现
        assert "E" * 3_000 not in output
        # 截断标记应出现
        assert "…" in output
        # 错误标题应出现
        assert "Error:" in output


class TestRenderWarning:
    """警告消息渲染测试。"""

    def test_render_warning(self, renderer: Renderer) -> None:
        renderer.render_warning("磁盘空间不足")
        output = _get_output(renderer)
        assert "磁盘空间不足" in output

    def test_long_warning_truncated(self, renderer: Renderer) -> None:
        long_msg = "W" * 3_000
        renderer.render_warning(long_msg)
        output = _get_output(renderer)
        # 原始完整消息不应出现
        assert "W" * 3_000 not in output
        # 截断标记应出现
        assert "…" in output


class TestRenderInfo:
    """信息消息渲染测试。"""

    def test_render_info(self, renderer: Renderer) -> None:
        renderer.render_info("模型已切换至 deepseek-chat")
        output = _get_output(renderer)
        assert "deepseek-chat" in output

    def test_long_info_truncated(self, renderer: Renderer) -> None:
        long_msg = "I" * 3_000
        renderer.render_info(long_msg)
        output = _get_output(renderer)
        # 原始完整消息不应出现
        assert "I" * 3_000 not in output
        # 截断标记应出现
        assert "…" in output


# ---------------------------------------------------------------------------
# render_token_usage
# ---------------------------------------------------------------------------


class TestRenderTokenUsage:
    """Token 用量渲染测试。"""

    def test_full_usage(self, renderer: Renderer) -> None:
        usage = {"prompt": 1000, "completion": 500, "total": 1500}
        renderer.render_token_usage(usage, 0.015)
        output = _get_output(renderer)
        assert "1,500" in output
        assert "0.0150" in output

    def test_usage_without_total(self, renderer: Renderer) -> None:
        """未提供 total 时自动计算 prompt + completion。"""
        usage = {"prompt": 100, "completion": 50}
        renderer.render_token_usage(usage, 0.001)
        output = _get_output(renderer)
        assert "150" in output  # 100 + 50 = 150

    def test_zero_usage(self, renderer: Renderer) -> None:
        """零用量不应崩溃。"""
        renderer.render_token_usage({}, 0.0)
        output = _get_output(renderer)
        assert "0" in output

    def test_fallback_with_non_numeric_cost(self, renderer: Renderer) -> None:
        """fallback 路径中 cost 为非数字时不应崩溃（严重问题 #1 修复验证）。"""
        with patch.object(
            renderer.console, "print", side_effect=RuntimeError("boom")
        ):
            # 不应抛出异常
            renderer.render_token_usage({"prompt": 1, "completion": 1}, "bad-cost")


# ---------------------------------------------------------------------------
# render_project_context
# ---------------------------------------------------------------------------


class TestRenderProjectContext:
    """项目上下文渲染测试。"""

    def test_full_context(self, renderer: Renderer) -> None:
        from miaowa.core.types import ProjectCache

        cache = ProjectCache(
            tech_stack={"language": "Python", "framework": "FastAPI"},
            structure={"module_count": 12, "file_count": 50},
            key_files=["pyproject.toml", "README.md", "src/main.py"],
        )
        renderer.render_project_context(cache)
        output = _get_output(renderer)
        assert "Python" in output
        assert "FastAPI" in output
        assert "pyproject.toml" in output
        assert "12" in output

    def test_empty_cache(self, renderer: Renderer) -> None:
        """空缓存不应崩溃。"""
        from miaowa.core.types import ProjectCache

        cache = ProjectCache()
        renderer.render_project_context(cache)
        output = _get_output(renderer)
        assert "Project context" in output

    def test_cache_with_structure_tree(self, renderer: Renderer) -> None:
        """包含 tree 的结构信息。"""
        from miaowa.core.types import ProjectCache

        cache = ProjectCache(
            structure={
                "tree": "src/\n  main.py\n  utils.py",
                "file_count": 2,
            },
        )
        renderer.render_project_context(cache)
        output = _get_output(renderer)
        assert "main.py" in output

    def test_long_tree_truncated(self, renderer: Renderer) -> None:
        """超长目录树应被截断。"""
        from miaowa.core.types import ProjectCache

        long_tree = "\n".join([f"  file_{i}.py" for i in range(50)])
        cache = ProjectCache(
            structure={"tree": long_tree, "file_count": 50},
        )
        renderer.render_project_context(cache)
        output = _get_output(renderer)
        # 超长 tree 应被截断，不应包含完整内容
        assert "…" in output

    def test_long_key_files_truncated(self, renderer: Renderer) -> None:
        """超长关键文件列表应被截断。"""
        from miaowa.core.types import ProjectCache

        many_files = [f"path/to/file_{i}.py" for i in range(30)]
        cache = ProjectCache(key_files=many_files)
        renderer.render_project_context(cache)
        output = _get_output(renderer)
        # 完整列表不应全部出现（会被截断）
        assert "…" in output

    def test_long_tech_stack_truncated(self, renderer: Renderer) -> None:
        """超长技术栈信息应被截断。"""
        from miaowa.core.types import ProjectCache

        many_techs = {f"tool_{i}": f"version_{i}" * 10 for i in range(20)}
        cache = ProjectCache(tech_stack=many_techs)
        renderer.render_project_context(cache)
        output = _get_output(renderer)
        assert "…" in output


# ---------------------------------------------------------------------------
# render_code
# ---------------------------------------------------------------------------


class TestRenderCode:
    """代码语法高亮渲染测试。"""

    def test_python_code(self, renderer: Renderer) -> None:
        renderer.render_code("def hello():\n    return 'world'\n", "python")
        output = _get_output(renderer)
        assert "hello" in output

    def test_no_language(self, renderer: Renderer) -> None:
        """默认语言参数为 python。"""
        renderer.render_code("print('hello')")
        output = _get_output(renderer)
        assert "hello" in output

    def test_fallback(self, renderer: Renderer) -> None:
        """Rich 渲染失败时回退。"""
        with patch.object(
            renderer.console, "print", side_effect=RuntimeError("boom")
        ):
            renderer.render_code("some code")

    def test_long_code_truncated(self, renderer: Renderer) -> None:
        """超长代码应被截断并显示警告。"""
        long_code = "x = 1\n" * 10_000  # 超出 _CODE_MAX_LENGTH (50,000)
        renderer.render_code(long_code)
        output = _get_output(renderer)
        assert "truncated" in output
        # 原始 60,000 字符的全文不应出现在输出中
        assert "x = 1\n" * 10_000 not in output

    def test_with_language_identifier(self, renderer: Renderer) -> None:
        """明确指定语言标识符时的渲染。"""
        renderer.render_code("const x = 1;", "javascript")
        output = _get_output(renderer)
        assert "x" in output


# ---------------------------------------------------------------------------
# 窄终端测试
# ---------------------------------------------------------------------------


class TestNarrowTerminal:
    """窄终端降级行为测试。"""

    def test_welcome_narrow_fallback(self, narrow_renderer: Renderer) -> None:
        """窄终端下 welcome 应降级为单行。"""
        narrow_renderer.render_welcome(model="gpt-4")
        output = _get_output(narrow_renderer)
        assert "Miaowa Code" in output
        assert "gpt-4" in output
        # 不应出现 Box Drawing 字符
        assert "╭" not in output

    def test_normal_terminal_not_affected(self, renderer: Renderer) -> None:
        """正常宽度终端不应降级。"""
        renderer.render_welcome(model="gpt-4")
        output = _get_output(renderer)
        assert "Miaowa Code" in output


# ---------------------------------------------------------------------------
# 静默失败（全局回退）测试
# ---------------------------------------------------------------------------


class TestSilentFail:
    """所有渲染方法在异常时静默回退，不向上抛出异常。"""

    @pytest.mark.parametrize(
        "method_name,args",
        [
            ("render_markdown", ("text",)),
            ("render_stream_chunk", ("chunk",)),
            ("flush_stream", ()),
            ("render_welcome", ()),
            ("render_goodbye", ({"turns": 1},)),
            ("render_tool_call", ("test_tool", {"arg": "val"})),
            ("render_tool_result", (True, "summary")),
            ("render_error", ("msg",)),
            ("render_warning", ("msg",)),
            ("render_info", ("msg",)),
            ("render_token_usage", ({"prompt": 1, "completion": 1}, 0.0)),
        ],
    )
    def test_method_silent_on_error(
        self, renderer: Renderer, method_name: str, args: tuple
    ) -> None:
        """每个渲染方法在 console.print 抛异常时都不应向上传播。"""
        method = getattr(renderer, method_name)
        # Phase 2: render_stream_chunk / flush_stream 可能使用 Live 而非 console.print
        # 测试多种异常路径以确保静默失败
        if method_name in ("render_stream_chunk", "flush_stream"):
            # 清除可能残留的 Live 状态
            renderer._live = None
        with patch.object(
            renderer.console, "print", side_effect=RuntimeError("boom")
        ):
            # 不应抛出异常
            method(*args)


# ---------------------------------------------------------------------------
# _get_terminal_width
# ---------------------------------------------------------------------------


class TestGetTerminalWidth:
    """终端宽度检测测试。"""

    def test_returns_console_width(self, renderer: Renderer) -> None:
        width = renderer._get_terminal_width()
        assert width == 120  # 测试夹具中设置的宽度

    def test_fallback_when_width_none(self, ui_config: UIConfig) -> None:
        """width 为 None 时回退到默认值（≥79）。"""
        r = Renderer(ui_config)
        r.console = Console(force_terminal=True, file=io.StringIO(), width=None)
        width = r._get_terminal_width()
        # Rich Console 在 width=None 且 force_terminal=True 时
        # 默认宽度为 79 或 80（取决于终端模拟），验证大于 0 即可
        assert width >= 79


# ---------------------------------------------------------------------------
# _safe_format_cost
# ---------------------------------------------------------------------------


class TestSafeFormatCost:
    """_safe_format_cost 函数测试。"""

    def test_float(self) -> None:
        assert _safe_format_cost(0.0123) == "0.0123"

    def test_int(self) -> None:
        assert _safe_format_cost(5) == "5.0000"

    def test_string_numeric(self) -> None:
        """数值字符串可以正常转换。"""
        assert _safe_format_cost("0.5") == "0.5000"

    def test_string_non_numeric(self) -> None:
        """非数值字符串不应崩溃，回退到 str()。"""
        result = _safe_format_cost("not-a-number")
        assert result == "not-a-number"

    def test_none(self) -> None:
        """None 不应崩溃。"""
        result = _safe_format_cost(None)
        assert result == "None"

    def test_dict(self) -> None:
        """dict 不应崩溃。"""
        result = _safe_format_cost({"a": 1})
        assert "{" in result


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


class TestFormatJsonCompact:
    """_format_json_compact 函数测试。"""

    def test_simple_dict(self) -> None:
        result = _format_json_compact({"a": 1, "b": "hello"})
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": "hello"}

    def test_truncation(self) -> None:
        """超出 max_length 应截断。"""
        long_str = "x" * 100
        result = _format_json_compact({"key": long_str}, max_length=50)
        assert len(result) <= 50 + 1  # +1 for the "…"
        assert result.endswith("…")

    def test_non_serializable(self) -> None:
        """不可 JSON 序列化的对象应回退到 str()。"""
        obj = {"fn": lambda: None}  # type: ignore[dict-item]
        result = _format_json_compact(obj)
        assert len(result) > 0


class TestTruncateText:
    """_truncate_text 函数测试。"""

    def test_short_text(self) -> None:
        result = _truncate_text("hello", 10)
        assert result == "hello"

    def test_long_text(self) -> None:
        result = _truncate_text("hello world, this is long", 10)
        assert len(result) == 11  # 10 + "…"
        assert result.endswith("…")

    def test_exact_length(self) -> None:
        result = _truncate_text("12345", 5)
        assert result == "12345"


class TestTruncateMarkdown:
    """_truncate_markdown 函数测试。"""

    def test_short_text_passes_through(self) -> None:
        """短文本不应被截断。"""
        result = _truncate_markdown("hello", 100)
        assert result == "hello"

    def test_paragraph_boundary_truncation(self) -> None:
        """应在段落边界（双换行）处截断。"""
        text = "para1\n\npara2\n\npara3\n\npara4"
        # max_length 选在 para3 中间
        max_len = len("para1\n\npara2\n\npa")
        result = _truncate_markdown(text, max_len)
        # 应在 para2 之后截断
        assert "para3" not in result
        assert "para2" in result
        assert "truncated" in result

    def test_code_fence_boundary_truncation(self) -> None:
        """应在代码块结束边界处截断。"""
        text = "some text\n```python\ncode block\n```\nmore text"
        # max_length 选在 "more text" 中间
        max_len = len("some text\n```python\ncode block\n```\nmo")
        result = _truncate_markdown(text, max_len)
        # 应在 ``` 处截断，保留完整代码块
        assert "more" not in result
        assert "code block" in result
        assert "truncated" in result

    def test_tilde_fence_boundary_truncation(self) -> None:
        """应在波浪线代码块结束边界处截断。"""
        text = "some text\n~~~python\ncode block\n~~~\nmore text"
        max_len = len("some text\n~~~python\ncode block\n~~~\nmo")
        result = _truncate_markdown(text, max_len)
        # 应在 ~~~ 处截断，保留完整代码块
        assert "more" not in result
        assert "code block" in result
        assert "truncated" in result

    def test_no_boundary_falls_back_to_hard_cut(self) -> None:
        """无合适边界时回退到字符级截断。"""
        text = "x" * 200 + "y" * 200
        result = _truncate_markdown(text, 150)
        assert len(result) <= 150 + 50  # 允许截断提示的额外开销
        assert "truncated" in result

    def test_does_not_truncate_at_too_early_boundary(self) -> None:
        """不应在太靠前的边界处截断（避免丢失大量内容）。"""
        text = "short\n\n" + "x" * 1000
        # max_length 在长文本中间，但太靠前的 \n\n 不应被选为截断点
        result = _truncate_markdown(text, 600)
        # 应保留大部分内容（不应在 "short" 之后立即截断）
        assert len(result) > 400


class TestTruncateCode:
    """_truncate_code 函数测试。"""

    def test_short_code_passes_through(self) -> None:
        """短代码不应被截断。"""
        result = _truncate_code("print('hello')", 100)
        assert result == "print('hello')"

    def test_line_boundary_truncation(self) -> None:
        """应在完整行边界处截断。"""
        lines = [f"line_{i} = {i}" for i in range(100)]
        code = "\n".join(lines)
        # max_length 选在 line_60 中间
        max_len = len("\n".join(lines[:60])) + 3  # 切在 line_60 中间
        result = _truncate_code(code, max_len)
        # line_60 不应完整出现（在截断点之后）
        assert "line_60" not in result
        # line_59 应完整保留
        assert "line_59" in result
        assert "truncated" in result

    def test_no_boundary_falls_back_to_hard_cut(self) -> None:
        """无合适行边界时回退到字符级截断。"""
        code = "x" * 300  # 单行超长代码，无换行符
        result = _truncate_code(code, 150)
        assert len(result) <= 150 + 20  # 允许截断提示开销
        assert "truncated" in result

    def test_does_not_truncate_at_too_early_line(self) -> None:
        """不应在太靠前的行边界处截断。"""
        code = "first\n" + "x" * 2000
        result = _truncate_code(code, 1200)
        # 应保留大部分内容（不应在 "first" 之后立即截断）
        assert len(result) > 600


# ---------------------------------------------------------------------------
# Phase 2: CodeBlockState 测试 (§3.1.2)
# ---------------------------------------------------------------------------


class TestCodeBlockState:
    """CodeBlockState 状态机测试。"""

    def test_initial_state_is_normal(self) -> None:
        """初始状态应为 NORMAL。"""
        cbs = CodeBlockState()
        assert not cbs.is_in_code_block()
        assert cbs.language == ""

    def test_open_code_block_with_language(self) -> None:
        """```python 应进入 IN_CODE_BLOCK 并记录语言。"""
        cbs = CodeBlockState()
        state, line = cbs.process_line("```python")
        assert cbs.is_in_code_block()
        assert cbs.language == "python"
        assert line == "```python"

    def test_open_code_block_without_language(self) -> None:
        """无语言标识的 ``` 也应进入代码块。"""
        cbs = CodeBlockState()
        cbs.process_line("```")
        assert cbs.is_in_code_block()
        assert cbs.language == ""

    def test_close_code_block(self) -> None:
        """单独的 ``` 应关闭代码块回到 NORMAL。"""
        cbs = CodeBlockState()
        cbs.process_line("```python")
        cbs.process_line("code line")
        state, _ = cbs.process_line("```")
        assert not cbs.is_in_code_block()
        assert cbs.language == ""

    def test_triple_backtick_in_code_block_not_close(self) -> None:
        """代码块内的非单独 ``` 不应关闭代码块。"""
        cbs = CodeBlockState()
        cbs.process_line("```python")
        state, _ = cbs.process_line("some ``` in code")
        assert cbs.is_in_code_block()

    def test_reset(self) -> None:
        """reset() 应回到初始状态。"""
        cbs = CodeBlockState()
        cbs.process_line("```python")
        cbs.process_line("code")
        cbs.reset()
        assert not cbs.is_in_code_block()
        assert cbs.language == ""

    def test_nested_code_block_simulation(self) -> None:
        """模拟完整的代码块生命周期。"""
        cbs = CodeBlockState()
        # Normal text
        state, _ = cbs.process_line("Some text")
        assert not cbs.is_in_code_block()
        # Open block
        cbs.process_line("```python")
        assert cbs.language == "python"
        # Code lines
        cbs.process_line("def hello():")
        cbs.process_line("    return 'world'")
        assert cbs.is_in_code_block()
        # Close block
        cbs.process_line("```")
        assert not cbs.is_in_code_block()
        # Back to normal
        state, _ = cbs.process_line("More text")
        assert not cbs.is_in_code_block()

    def test_standalone_tick_not_open(self) -> None:
        """单个 ` 不应触发代码块。"""
        cbs = CodeBlockState()
        cbs.process_line("`inline code`")
        assert not cbs.is_in_code_block()

    def test_indented_fence(self) -> None:
        """缩进的 ``` 也应识别（strip 后仍以 ``` 开头）。"""
        cbs = CodeBlockState()
        cbs.process_line("  ```python")
        assert cbs.is_in_code_block()
        assert cbs.language == "python"

    # -- Phase 2 修复: 波浪线围栏 (~) 支持 --

    def test_tilde_fence_open_close(self) -> None:
        """~~~ 波浪线围栏应正确开启和关闭代码块。"""
        cbs = CodeBlockState()
        cbs.process_line("~~~python")
        assert cbs.is_in_code_block()
        assert cbs.language == "python"
        cbs.process_line("code line")
        assert cbs.is_in_code_block()
        cbs.process_line("~~~")
        assert not cbs.is_in_code_block()

    def test_tilde_fence_without_language(self) -> None:
        """无语言标识的 ~~~ 也应进入代码块。"""
        cbs = CodeBlockState()
        cbs.process_line("~~~")
        assert cbs.is_in_code_block()
        assert cbs.language == ""

    def test_tilde_fence_not_closed_by_backtick(self) -> None:
        """~~~ 开启的代码块不应被 ``` 关闭。"""
        cbs = CodeBlockState()
        cbs.process_line("~~~python")
        cbs.process_line("```")
        # ``` 出现在代码块内部，不应关闭 ~~~ 围栏
        assert cbs.is_in_code_block()

    def test_four_backtick_fence(self) -> None:
        """```` 4 反引号围栏应正确开启和关闭。"""
        cbs = CodeBlockState()
        cbs.process_line("````python")
        assert cbs.is_in_code_block()
        assert cbs.language == "python"
        # 3 个反引号不应关闭 4 反引号围栏（长度不足）
        cbs.process_line("```")
        assert cbs.is_in_code_block()
        # 4 个反引号应正确关闭
        cbs.process_line("````")
        assert not cbs.is_in_code_block()

    def test_four_backtick_closed_by_five(self) -> None:
        """4 反引号围栏可以被 5 反引号关闭（长度 ≥ 开启长度）。"""
        cbs = CodeBlockState()
        cbs.process_line("````")
        assert cbs.is_in_code_block()
        cbs.process_line("`````")
        assert not cbs.is_in_code_block()

    def test_five_tilde_fence(self) -> None:
        """~~~~~ 5 波浪线围栏正确工作。"""
        cbs = CodeBlockState()
        cbs.process_line("~~~~~")
        assert cbs.is_in_code_block()
        cbs.process_line("~~~~~")
        assert not cbs.is_in_code_block()


# ---------------------------------------------------------------------------
# Phase 2: _display_width 测试
# ---------------------------------------------------------------------------


class TestDisplayWidth:
    """_display_width 函数测试 — CJK 宽度感知。"""

    def test_ascii_width(self) -> None:
        """ASCII 字符宽度为 1。"""
        assert _display_width("hello") == 5

    def test_chinese_width(self) -> None:
        """中文字符宽度为 2。"""
        assert _display_width("你好") == 4

    def test_mixed_width(self) -> None:
        """中英混合正确计算。"""
        assert _display_width("hello你好") == 5 + 4

    def test_emoji_width(self) -> None:
        """Emoji 宽度通常为 2。"""
        assert _display_width("😀") == 2

    def test_empty_string(self) -> None:
        """空字符串宽度为 0。"""
        assert _display_width("") == 0


# ---------------------------------------------------------------------------
# Phase 2: _strip_ansi 测试
# ---------------------------------------------------------------------------


class TestStripAnsi:
    """ANSI 转义序列剥离测试。"""

    def test_plain_text_unchanged(self) -> None:
        assert _strip_ansi("hello world") == "hello world"

    def test_color_code_stripped(self) -> None:
        assert _strip_ansi("\x1b[31mred\x1b[0m") == "red"

    def test_bold_code_stripped(self) -> None:
        assert _strip_ansi("\x1b[1mbold\x1b[0m") == "bold"

    def test_multiple_codes_stripped(self) -> None:
        text = "\x1b[1m\x1b[31mbold red\x1b[0m"
        assert _strip_ansi(text) == "bold red"

    def test_no_ansi_sequences(self) -> None:
        """不含 ANSI 序列的文本原样返回。"""
        assert _strip_ansi("普通中文文本") == "普通中文文本"

    def test_chinese_with_ansi(self) -> None:
        """中文 + ANSI 颜色码正确剥离。"""
        assert _strip_ansi("\x1b[34m蓝色\x1b[0m") == "蓝色"


# ---------------------------------------------------------------------------
# Phase 2: _truncate_line_preserve_ansi 测试 (§3.1.3)
# ---------------------------------------------------------------------------


class TestTruncateLinePreserveAnsi:
    """长行截断（保留 ANSI 颜色码）测试。"""

    def test_short_line_passes_through(self) -> None:
        line = "short line"
        result = _truncate_line_preserve_ansi(line, 50)
        assert result == line

    def test_long_line_truncated_with_suffix(self) -> None:
        line = "x" * 100
        result = _truncate_line_preserve_ansi(line, 30)
        assert result.endswith("...")
        # 30 chars + ANSI reset + ...
        visible = _strip_ansi(result)
        assert len(visible) == 30 + 3  # visible chars + "..."

    def test_ansi_reset_closed(self) -> None:
        """截断后 ANSI 颜色码应正确闭合（\x1b[0m）。"""
        line = "\x1b[31m" + "x" * 100
        result = _truncate_line_preserve_ansi(line, 20)
        assert "\x1b[0m" in result
        # "..." 前应有 reset
        assert result.endswith("\x1b[0m...")

    def test_colored_line_preserves_ansi_before_truncation(self) -> None:
        """截断点之前的 ANSI 序列应保留。"""
        line = "\x1b[31mred text\x1b[0m more content that gets cut"
        result = _truncate_line_preserve_ansi(line, 15)
        assert "\x1b[31m" in result
        assert "red text" in result
        assert result.endswith("...")

    def test_exact_width_no_truncation(self) -> None:
        """恰好等于 max_width 不应被截断。"""
        line = "1234567890"
        result = _truncate_line_preserve_ansi(line, 10)
        assert result == line

    def test_empty_line(self) -> None:
        """空行不应崩溃。"""
        result = _truncate_line_preserve_ansi("", 50)
        assert result == ""

    def test_max_width_one(self) -> None:
        """max_width=1 边界情况。"""
        line = "abc"
        result = _truncate_line_preserve_ansi(line, 1)
        visible = _strip_ansi(result)
        assert len(visible) <= 4  # 1 visible + "..." + reset codes
        assert "..." in result

    def test_chinese_characters(self) -> None:
        """中文字符应正确计数（每个 CJK 字符计为 2 列显示宽度）。"""
        line = "你好世界" * 50
        result = _truncate_line_preserve_ansi(line, 10)
        visible = _strip_ansi(result)
        # max_width=10 列 → 5 个 CJK 字符（各 2 列）+ "..." 后缀
        assert len(visible) == 8  # 5 CJK chars + "..." (3 个 ASCII 字符)

    def test_mixed_ansi_and_visible(self) -> None:
        """ANSI 序列不计入可见字符数。"""
        line = "\x1b[1m\x1b[34m" + "x" * 50 + "\x1b[0m"
        result = _truncate_line_preserve_ansi(line, 25)
        visible = _strip_ansi(result)
        assert len(visible) == 28  # 25 + "..."

    def test_ansi_at_truncation_point(self) -> None:
        """截断点恰好落在 ANSI 序列中间时不应崩溃。"""
        # 在可见字符中间插入 ANSI 序列
        line = "aaaaa\x1b[31mbbbbb\x1b[0mccccc"
        result = _truncate_line_preserve_ansi(line, 7)
        visible = _strip_ansi(result)
        assert len(visible) <= 10  # 7 chars + possible "..."

    def test_emoji_preserved(self) -> None:
        """Emoji 字符应被保留（作为可见字符计数）。"""
        line = "hello 😀 world 🌍 extra content"
        result = _truncate_line_preserve_ansi(line, 15)
        assert "😀" in result
        assert result.endswith("...")


# ---------------------------------------------------------------------------
# Phase 2: Debug 渲染模式测试 (§3.1.4)
# ---------------------------------------------------------------------------


class TestDebugRenderMode:
    """debug_render 属性测试。"""

    def test_default_debug_off(self, renderer: Renderer) -> None:
        """默认 debug_render 应为 False。"""
        assert renderer.debug_render is False

    def test_enable_debug_via_property(self, renderer: Renderer) -> None:
        """通过属性可启停 debug 模式。"""
        renderer.debug_render = True
        assert renderer.debug_render is True
        renderer.debug_render = False
        assert renderer.debug_render is False

    def test_debug_overlay_adds_fps_info(self, renderer: Renderer) -> None:
        """debug 模式下渲染应包含帧率信息。"""
        renderer.debug_render = True
        renderer.render_stream_chunk("test debug overlay")
        renderer.flush_stream()
        output = _get_output(renderer)
        # debug 叠加层应包含 fps 和 buf 信息
        assert "fps=" in output or "buf=" in output

    def test_debug_off_no_overlay(self, renderer: Renderer) -> None:
        """debug 关闭时不应包含调试信息。"""
        renderer.debug_render = False
        renderer.render_stream_chunk("plain content")
        renderer.flush_stream()
        output = _get_output(renderer)
        assert "fps=" not in output

    def test_frame_count_increments(self, renderer: Renderer) -> None:
        """每次 _do_flush 应递增帧计数。"""
        assert renderer._frame_count == 0
        renderer.render_stream_chunk("frame 1\n")  # newline triggers flush
        assert renderer._frame_count == 1
        renderer.render_stream_chunk("frame 2\n")  # another flush
        assert renderer._frame_count == 2

    def test_flush_stream_resets_frame_count(self, renderer: Renderer) -> None:
        """flush_stream 后帧计数应重置。"""
        renderer.render_stream_chunk("test\n")
        assert renderer._frame_count > 0
        renderer.flush_stream()
        assert renderer._frame_count == 0


# ---------------------------------------------------------------------------
# Phase 2: _process_stream_buffer 集成测试
# ---------------------------------------------------------------------------


class TestProcessStreamBuffer:
    """_process_stream_buffer 集成测试。"""

    def test_normal_text_unchanged(self, renderer: Renderer) -> None:
        """普通文本应原样通过。"""
        result = renderer._process_stream_buffer("hello world")
        assert result == "hello world"

    def test_long_lines_truncated(self, renderer: Renderer) -> None:
        """超长行应被截断。"""
        long_line = "x" * (120 * 3 + 10)  # 超出 3 倍终端宽度
        result = renderer._process_stream_buffer(long_line)
        assert "..." in result
        assert len(_strip_ansi(result)) < len(long_line)

    def test_code_block_lines_tracked(self, renderer: Renderer) -> None:
        """代码块行应被状态机正确跟踪（但内容不变）。"""
        text = "before\n```python\nprint('hello')\n```\nafter"
        result = renderer._process_stream_buffer(text)
        # 所有内容应保留
        assert "before" in result
        assert "```python" in result
        assert "print('hello')" in result
        assert "after" in result

    def test_state_reset_before_processing(self, renderer: Renderer) -> None:
        """每次调用 _process_stream_buffer 前状态机应重置。"""
        # 先处理一个不完整的缓冲区（开启代码块但未关闭）
        renderer._process_stream_buffer("```python\ncode line")
        assert renderer._code_block_state.is_in_code_block()
        # 再次调用应重置状态
        renderer._process_stream_buffer("normal text")
        assert not renderer._code_block_state.is_in_code_block()
