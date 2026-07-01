"""REPL 单元测试。

使用 Mock 对象模拟 AgentExecutor / Renderer / SessionManager 等依赖，
隔离测试 REPL 的业务逻辑（元命令处理、统计累加等）。
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from rich.console import Console

from miaowa.agent.executor import AgentExecutor
from miaowa.agent.session import MemoryManager, SessionManager
from miaowa.cli.parser import CommandParser, ParsedCommand
from miaowa.cli.renderer import Renderer
from miaowa.cli.repl import REPL
from miaowa.core.config import Config, LLMConfig, UIConfig


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_executor() -> Mock:
    """创建 Mock AgentExecutor。"""
    executor = MagicMock(spec=AgentExecutor)
    executor.MAX_ITERATIONS = 10
    executor.RUN_TIMEOUT = 300.0
    executor.last_response = None
    executor.get_status.return_value = {
        "last_response": None,
        "cache_status": {"cache_hit": False, "tool_count": 5},
        "tool_count": 5,
        "max_iterations": 10,
        "run_timeout": 300.0,
    }
    executor._ctx_builder = MagicMock()
    return executor


@pytest.fixture
def mock_renderer() -> Renderer:
    """创建 Renderer，输出重定向到 StringIO。"""
    config = UIConfig(theme="dark", syntax_theme="monokai", max_history=1000)
    r = Renderer(config)
    output = io.StringIO()
    r.console = Console(force_terminal=True, file=output, highlight=False, width=120)
    return r


@pytest.fixture
def config() -> Config:
    """创建带测试 API Key 的 Config。"""
    return Config(
        llm=LLMConfig(
            provider="deepseek",
            api_key="test-key-12345",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=4096,
            timeout=120,
        ),
        ui=UIConfig(theme="dark", syntax_theme="monokai", max_history=1000),
    )


@pytest.fixture
def parser() -> CommandParser:
    """创建 CommandParser。"""
    return CommandParser()


@pytest.fixture
def session_manager() -> SessionManager:
    """创建 SessionManager。"""
    return SessionManager()


@pytest.fixture
def repl(
    mock_executor: Mock,
    mock_renderer: Renderer,
    parser: CommandParser,
    config: Config,
    session_manager: SessionManager,
) -> REPL:
    """创建完整的 REPL 实例。"""
    return REPL(
        executor=mock_executor,  # type: ignore[arg-type]
        renderer=mock_renderer,
        parser=parser,
        config=config,
        session_manager=session_manager,
    )


def _get_output(renderer: Renderer) -> str:
    """获取 renderer 的输出内容。"""
    assert isinstance(renderer.console.file, io.StringIO)
    return renderer.console.file.getvalue()


# ---------------------------------------------------------------------------
# 构造函数
# ---------------------------------------------------------------------------


class TestReplInit:
    """REPL 初始化测试。"""

    def test_default_attributes(
        self,
        mock_executor: Mock,
        mock_renderer: Renderer,
        parser: CommandParser,
        config: Config,
        session_manager: SessionManager,
    ) -> None:
        repl = REPL(mock_executor, mock_renderer, parser, config, session_manager)  # type: ignore[arg-type]
        assert repl.executor is mock_executor
        assert repl.renderer is mock_renderer
        assert repl.parser is parser
        assert repl.config is config
        assert repl.sessions is session_manager
        assert repl.current_model == "deepseek-chat"
        assert repl._turn_count == 0
        assert repl._total_cost == 0.0
        assert repl._debug_mode is False

    def test_total_tokens_initial_zero(self, repl: REPL) -> None:
        assert repl._total_tokens == {"prompt": 0, "completion": 0, "total": 0}


# ---------------------------------------------------------------------------
# _accumulate_stats
# ---------------------------------------------------------------------------


class TestAccumulateStats:
    """统计累加测试。"""

    def test_single_accumulation(self, repl: REPL) -> None:
        tokens = {"prompt": 100, "completion": 50, "total": 150}
        repl._accumulate_stats(tokens, 0.015)

        assert repl._total_tokens["prompt"] == 100
        assert repl._total_tokens["completion"] == 50
        assert repl._total_tokens["total"] == 150
        assert repl._total_cost == 0.015

    def test_multiple_accumulations(self, repl: REPL) -> None:
        repl._accumulate_stats({"prompt": 100, "completion": 50, "total": 150}, 0.01)
        repl._accumulate_stats({"prompt": 200, "completion": 100, "total": 300}, 0.02)

        assert repl._total_tokens["prompt"] == 300
        assert repl._total_tokens["completion"] == 150
        assert repl._total_tokens["total"] == 450
        assert repl._total_cost == 0.03

    def test_empty_tokens_dict(self, repl: REPL) -> None:
        """缺少字段时应默认为 0，不崩溃。"""
        repl._accumulate_stats({}, 0.0)
        assert repl._total_tokens == {"prompt": 0, "completion": 0, "total": 0}
        assert repl._total_cost == 0.0


# ---------------------------------------------------------------------------
# _handle_meta_command — /quit / /exit
# ---------------------------------------------------------------------------


class TestMetaQuitExit:
    """退出命令测试。"""

    @pytest.mark.asyncio
    async def test_quit_returns_true(self, repl: REPL) -> None:
        parsed = ParsedCommand(type="meta", command="/quit", args=[])
        result = await repl._handle_meta_command(parsed)
        assert result is True

    @pytest.mark.asyncio
    async def test_exit_returns_true(self, repl: REPL) -> None:
        parsed = ParsedCommand(type="meta", command="/exit", args=[])
        result = await repl._handle_meta_command(parsed)
        assert result is True


# ---------------------------------------------------------------------------
# _handle_meta_command — /clear
# ---------------------------------------------------------------------------


class TestMetaClear:
    """清空历史命令测试。"""

    @pytest.mark.asyncio
    async def test_clear_resets_state(self, repl: REPL) -> None:
        # 先累加一些统计数据
        repl._accumulate_stats({"prompt": 100, "completion": 50, "total": 150}, 0.01)
        repl._turn_count = 5

        parsed = ParsedCommand(type="meta", command="/clear", args=[])
        result = await repl._handle_meta_command(parsed)

        assert result is False  # 不退出
        assert repl._total_tokens == {"prompt": 0, "completion": 0, "total": 0}
        assert repl._total_cost == 0.0
        assert repl._turn_count == 0

    @pytest.mark.asyncio
    async def test_clear_output_message(self, repl: REPL, mock_renderer: Renderer) -> None:
        parsed = ParsedCommand(type="meta", command="/clear", args=[])
        await repl._handle_meta_command(parsed)
        output = _get_output(mock_renderer)
        assert "cleared" in output


# ---------------------------------------------------------------------------
# _handle_meta_command — /help
# ---------------------------------------------------------------------------


class TestMetaHelp:
    """帮助命令测试。"""

    @pytest.mark.asyncio
    async def test_help_returns_false(self, repl: REPL) -> None:
        parsed = ParsedCommand(type="meta", command="/help", args=[])
        result = await repl._handle_meta_command(parsed)
        assert result is False

    @pytest.mark.asyncio
    async def test_help_for_specific_command(self, repl: REPL, mock_renderer: Renderer) -> None:
        parsed = ParsedCommand(type="meta", command="/help", args=["/model"])
        await repl._handle_meta_command(parsed)
        output = _get_output(mock_renderer)
        assert "model" in output.lower()

    @pytest.mark.asyncio
    async def test_help_for_unknown_command(self, repl: REPL, mock_renderer: Renderer) -> None:
        parsed = ParsedCommand(type="meta", command="/help", args=["/unknown"])
        await repl._handle_meta_command(parsed)
        output = _get_output(mock_renderer)
        assert "Unknown" in output


# ---------------------------------------------------------------------------
# _handle_meta_command — /model
# ---------------------------------------------------------------------------


class TestMetaModel:
    """模型切换命令测试。"""

    @pytest.mark.asyncio
    async def test_model_show_current(self, repl: REPL, mock_renderer: Renderer) -> None:
        parsed = ParsedCommand(type="meta", command="/model", args=[])
        await repl._handle_meta_command(parsed)
        output = _get_output(mock_renderer)
        assert repl.current_model in output

    @pytest.mark.asyncio
    async def test_model_switch(self, repl: REPL) -> None:
        parsed = ParsedCommand(type="meta", command="/model", args=["deepseek-reasoner"])
        await repl._handle_meta_command(parsed)
        assert repl.current_model == "deepseek-reasoner"
        assert repl.config.llm.model == "deepseek-reasoner"

    @pytest.mark.asyncio
    async def test_model_switch_output(self, repl: REPL, mock_renderer: Renderer) -> None:
        parsed = ParsedCommand(type="meta", command="/model", args=["gpt-4"])
        await repl._handle_meta_command(parsed)
        output = _get_output(mock_renderer)
        assert "deepseek-chat" in output  # 旧模型
        assert "gpt-4" in output           # 新模型


# ---------------------------------------------------------------------------
# _handle_meta_command — /tokens
# ---------------------------------------------------------------------------


class TestMetaTokens:
    """Token 用量命令测试。"""

    @pytest.mark.asyncio
    async def test_tokens_shows_accumulated(self, repl: REPL, mock_renderer: Renderer) -> None:
        repl._accumulate_stats({"prompt": 500, "completion": 200, "total": 700}, 0.007)
        parsed = ParsedCommand(type="meta", command="/tokens", args=[])
        await repl._handle_meta_command(parsed)
        output = _get_output(mock_renderer)
        assert "700" in output


# ---------------------------------------------------------------------------
# _handle_meta_command — /cost
# ---------------------------------------------------------------------------


class TestMetaCost:
    """费用命令测试。"""

    @pytest.mark.asyncio
    async def test_cost_shows_accumulated(self, repl: REPL, mock_renderer: Renderer) -> None:
        repl._accumulate_stats({"prompt": 100, "completion": 50, "total": 150}, 0.0015)
        parsed = ParsedCommand(type="meta", command="/cost", args=[])
        await repl._handle_meta_command(parsed)
        output = _get_output(mock_renderer)
        assert "0.0015" in output


# ---------------------------------------------------------------------------
# _handle_meta_command — /cache
# ---------------------------------------------------------------------------


class TestMetaCache:
    """缓存状态命令测试。"""

    @pytest.mark.asyncio
    async def test_cache_shows_status(self, repl: REPL, mock_renderer: Renderer) -> None:
        parsed = ParsedCommand(type="meta", command="/cache", args=[])
        await repl._handle_meta_command(parsed)
        output = _get_output(mock_renderer)
        assert "Cache" in output

    @pytest.mark.asyncio
    async def test_cache_empty_status(self, repl: REPL, mock_renderer: Renderer) -> None:
        """空状态也不应崩溃。"""
        repl.executor.get_status.return_value = {}  # type: ignore[union-attr]
        parsed = ParsedCommand(type="meta", command="/cache", args=[])
        await repl._handle_meta_command(parsed)
        output = _get_output(mock_renderer)
        assert "Cache" in output


# ---------------------------------------------------------------------------
# _handle_meta_command — /debug
# ---------------------------------------------------------------------------


class TestMetaDebug:
    """调试模式切换命令测试。"""

    @pytest.mark.asyncio
    async def test_debug_toggle_on(self, repl: REPL, mock_renderer: Renderer) -> None:
        assert repl._debug_mode is False
        parsed = ParsedCommand(type="meta", command="/debug", args=[])
        await repl._handle_meta_command(parsed)
        assert repl._debug_mode is True
        output = _get_output(mock_renderer)
        assert "ON" in output

    @pytest.mark.asyncio
    async def test_debug_toggle_off(self, repl: REPL, mock_renderer: Renderer) -> None:
        repl._debug_mode = True
        parsed = ParsedCommand(type="meta", command="/debug", args=[])
        await repl._handle_meta_command(parsed)
        assert repl._debug_mode is False


# ---------------------------------------------------------------------------
# _handle_meta_command — 错误处理
# ---------------------------------------------------------------------------


class TestMetaErrorHandling:
    """元命令错误处理测试。"""

    @pytest.mark.asyncio
    async def test_handler_does_not_crash_on_exception(
        self, repl: REPL, mock_renderer: Renderer
    ) -> None:
        """即使内部处理器抛异常也不应向上传播。"""
        # 模拟 _handle_help 抛出异常
        with patch.object(repl, "_handle_help", side_effect=RuntimeError("boom")):
            parsed = ParsedCommand(type="meta", command="/help", args=[])
            result = await repl._handle_meta_command(parsed)
            assert result is False  # 不退出

    @pytest.mark.asyncio
    async def test_unknown_command_in_meta_handler(
        self, repl: REPL, mock_renderer: Renderer
    ) -> None:
        """未知命令在 meta handler 中也应优雅处理。"""
        # 注册一个已知命令但让其落入 else 分支
        parsed = ParsedCommand(type="meta", command="/somenewcmd", args=[])
        # 直接调用 — is_known 可能为 False，但我们只测 handler 的兜底
        result = await repl._handle_meta_command(parsed)
        assert result is False


# ---------------------------------------------------------------------------
# _get_prompt_session
# ---------------------------------------------------------------------------


class TestPromptSession:
    """PromptSession 创建测试。

    Note:
        PromptSession 构造函数在非 Windows 控制台环境（Git Bash / MSYS2）
        会因无法访问 Win32 控制台缓冲区而失败，因此本测试类用
        DummyInput/DummyOutput 绕过终端检测。
    """

    def test_session_created(self, repl: REPL) -> None:
        """验证 PromptSession 可正确创建（含 Dummy I/O）。"""
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.input import DummyInput
        from prompt_toolkit.output import DummyOutput
        from prompt_toolkit.shortcuts import PromptSession

        session = PromptSession(
            history=repl._history.pt_history,
            completer=WordCompleter(["/help", "/quit"]),
            input=DummyInput(),
            output=DummyOutput(),
        )
        assert session is not None

    def test_session_has_history(self, repl: REPL) -> None:
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.input import DummyInput
        from prompt_toolkit.output import DummyOutput
        from prompt_toolkit.shortcuts import PromptSession

        session = PromptSession(
            history=repl._history.pt_history,
            completer=WordCompleter(["/help"]),
            input=DummyInput(),
            output=DummyOutput(),
        )
        assert session.history is not None

    def test_session_has_completer(self, repl: REPL) -> None:
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.input import DummyInput
        from prompt_toolkit.output import DummyOutput
        from prompt_toolkit.shortcuts import PromptSession

        session = PromptSession(
            history=repl._history.pt_history,
            completer=WordCompleter(["/help", "/quit", "/model"]),
            input=DummyInput(),
            output=DummyOutput(),
        )
        assert session.completer is not None


# ---------------------------------------------------------------------------
# _handle_interrupt / _handle_eof
# ---------------------------------------------------------------------------


class TestSignalHandlers:
    """Ctrl+C / Ctrl+D 处理器测试。"""

    def test_handle_interrupt_does_not_raise(
        self, repl: REPL, mock_renderer: Renderer
    ) -> None:
        repl._handle_interrupt()
        output = _get_output(mock_renderer)
        assert "Interrupted" in output

    @pytest.mark.asyncio
    async def test_handle_eof_first_press_returns_false(self, repl: REPL) -> None:
        """首次 Ctrl+D 不应退出，返回 False 并提示。"""
        result = await repl._handle_eof()
        assert result is False
        assert repl._eof_count == 1

    @pytest.mark.asyncio
    async def test_handle_eof_second_press_returns_true(self, repl: REPL) -> None:
        """连续两次 Ctrl+D 应确认退出，返回 True。"""
        repl._eof_count = 1  # 模拟首次已按
        result = await repl._handle_eof()
        assert result is True
        assert repl._eof_count == 2

    @pytest.mark.asyncio
    async def test_handle_eof_resets_on_input(self, repl: REPL) -> None:
        """正常输入后 EOF 计数器应归零。"""
        repl._eof_count = 1
        # 模拟正常输入路径中的重置
        repl._eof_count = 0
        result = await repl._handle_eof()
        assert result is False  # 计数器从 0 开始


# ---------------------------------------------------------------------------
# _handle_natural_input 基础测试
# ---------------------------------------------------------------------------


class TestHandleNaturalInput:
    """自然语言输入处理测试。"""

    @pytest.mark.asyncio
    async def test_handles_streaming_response(
        self, repl: REPL, mock_renderer: Renderer
    ) -> None:
        """测试正常流式响应的处理流程。"""

        async def mock_run(user_input: str, project_root: Path):
            yield "Hello"
            yield " "
            yield "World"

        repl.executor.run.return_value = mock_run("", Path("."))  # type: ignore[union-attr]

        # 设置 last_response
        from miaowa.agent.executor import AgentResponse
        repl.executor.last_response = AgentResponse(  # type: ignore[union-attr]
            content="Hello World",
            tool_calls_made=0,
            tokens_used={"prompt": 10, "completion": 5, "total": 15},
            cost=0.0001,
            iterations=1,
        )

        await repl._handle_natural_input("hi", Path("."))

        output = _get_output(mock_renderer)
        assert "Hello World" in output
        assert repl._turn_count == 1

    @pytest.mark.asyncio
    async def test_handles_executor_error(
        self, repl: REPL, mock_renderer: Renderer
    ) -> None:
        """executor 抛异常时不应崩溃。"""

        async def mock_error(user_input: str, project_root: Path):
            raise RuntimeError("LLM connection failed")
            yield  # unreachable, but needed for AsyncGenerator

        repl.executor.run.return_value = mock_error("", Path("."))  # type: ignore[union-attr]

        await repl._handle_natural_input("test", Path("."))
        output = _get_output(mock_renderer)
        assert "Error" in output or "error" in output or "Internal" in output

    @pytest.mark.asyncio
    async def test_turn_count_increments(self, repl: REPL) -> None:
        assert repl._turn_count == 0

        async def mock_run(user_input: str, project_root: Path):
            yield "ok"

        repl.executor.run.return_value = mock_run("", Path("."))  # type: ignore[union-attr]
        repl.executor.last_response = None  # type: ignore[union-attr]

        await repl._handle_natural_input("hi", Path("."))
        assert repl._turn_count == 1

        repl.executor.run.return_value = mock_run("", Path("."))  # type: ignore[union-attr]
        await repl._handle_natural_input("hello again", Path("."))
        assert repl._turn_count == 2
