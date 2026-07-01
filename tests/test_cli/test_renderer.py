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
    Renderer,
    _format_json_compact,
    _safe_format_cost,
    _truncate_code,
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
    """流式输出测试。"""

    def test_single_chunk(self, renderer: Renderer) -> None:
        renderer.render_stream_chunk("你好")
        output = _get_output(renderer)
        assert "你好" in output

    def test_multiple_chunks(self, renderer: Renderer) -> None:
        renderer.render_stream_chunk("Hello")
        renderer.render_stream_chunk(" ")
        renderer.render_stream_chunk("World")
        output = _get_output(renderer)
        assert "Hello World" in output

    def test_newline_flushes_counter(self, renderer: Renderer) -> None:
        """换行符应触发 flush 并重置计数器。"""
        renderer.render_stream_chunk("line 1\n")
        output = _get_output(renderer)
        assert "line 1" in output
        assert renderer._flush_counter == 0

    def test_empty_chunk(self, renderer: Renderer) -> None:
        """空 chunk 不应崩溃。"""
        renderer.render_stream_chunk("")
        assert renderer._flush_counter == 0

    def test_flush_counter_accumulates(self, renderer: Renderer) -> None:
        """flush 计数器应正确累积。"""
        renderer.render_stream_chunk("hello")
        assert renderer._flush_counter == 5
        renderer.render_stream_chunk(" world")
        assert renderer._flush_counter == 11

    def test_exception_resets_counter(self, renderer: Renderer) -> None:
        """console.print 异常后 _flush_counter 应重置为 0。"""
        renderer._flush_counter = 10
        with patch.object(
            renderer.console, "print", side_effect=RuntimeError("boom")
        ):
            renderer.render_stream_chunk("test")
        assert renderer._flush_counter == 0


class TestFlushStream:
    """flush_stream 方法测试。"""

    def test_flush_with_pending_counter(self, renderer: Renderer) -> None:
        renderer._flush_counter = 42
        renderer.flush_stream()
        assert renderer._flush_counter == 0

    def test_flush_empty_counter(self, renderer: Renderer) -> None:
        """计数器为 0 时 flush 不应崩溃。"""
        renderer.flush_stream()
        assert renderer._flush_counter == 0


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
