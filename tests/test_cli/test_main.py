"""CLI 入口点单元测试。

测试 cli_entry / main / _build_arg_parser / _build_cli_overrides 等。
对外部依赖（ConfigManager、AgentExecutor、REPL）使用 mock 隔离。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from miaowa import __version__
from miaowa.main import (
    _build_arg_parser,
    _build_cli_overrides,
    _resolve_project_root,
    _wire_components,
    cli_entry,
    main,
)


# ---------------------------------------------------------------------------
# _build_arg_parser
# ---------------------------------------------------------------------------


class TestBuildArgParser:
    """参数解析器构建测试。"""

    def test_parser_created(self) -> None:
        parser = _build_arg_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_all_arguments_registered(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        # 所有参数应有默认值且不抛出异常
        assert args.model is None
        assert args.api_key is None
        assert args.base_url is None
        assert args.debug is False
        assert args.no_color is False
        assert args.project is None
        assert args.version is False

    def test_model_short_flag(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["-m", "deepseek-reasoner"])
        assert args.model == "deepseek-reasoner"

    def test_model_long_flag(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--model", "deepseek-chat"])
        assert args.model == "deepseek-chat"

    def test_debug_flag(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--debug"])
        assert args.debug is True

    def test_version_flag(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--version"])
        assert args.version is True

    def test_project_flag(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--project", "/tmp/test"])
        assert args.project == "/tmp/test"

    def test_api_key_flag(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--api-key", "sk-test123"])
        assert args.api_key == "sk-test123"

    def test_base_url_flag(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--base-url", "https://custom.api/v1"])
        assert args.base_url == "https://custom.api/v1"


# ---------------------------------------------------------------------------
# _build_cli_overrides
# ---------------------------------------------------------------------------


class TestBuildCliOverrides:
    """CLI 覆盖字典构建测试。"""

    def test_empty_overrides(self) -> None:
        args = argparse.Namespace(
            model=None, api_key=None, base_url=None, debug=False,
        )
        assert _build_cli_overrides(args) == {}

    def test_model_override(self) -> None:
        args = argparse.Namespace(
            model="deepseek-reasoner", api_key=None, base_url=None, debug=False,
        )
        overrides = _build_cli_overrides(args)
        assert overrides["model"] == "deepseek-reasoner"

    def test_api_key_override(self) -> None:
        args = argparse.Namespace(
            model=None, api_key="sk-test", base_url=None, debug=False,
        )
        overrides = _build_cli_overrides(args)
        assert overrides["api_key"] == "sk-test"

    def test_debug_adds_log_level(self) -> None:
        args = argparse.Namespace(
            model=None, api_key=None, base_url=None, debug=True,
        )
        overrides = _build_cli_overrides(args)
        assert overrides["log_level"] == "DEBUG"

    def test_multiple_overrides(self) -> None:
        args = argparse.Namespace(
            model="gpt-4", api_key="sk-abc", base_url="https://x.com/v1", debug=True,
        )
        overrides = _build_cli_overrides(args)
        assert overrides["model"] == "gpt-4"
        assert overrides["api_key"] == "sk-abc"
        assert overrides["base_url"] == "https://x.com/v1"
        assert overrides["log_level"] == "DEBUG"


# ---------------------------------------------------------------------------
# _resolve_project_root
# ---------------------------------------------------------------------------


class TestResolveProjectRoot:
    """项目根目录解析测试。"""

    def test_defaults_to_cwd(self) -> None:
        args = argparse.Namespace(project=None)
        result = _resolve_project_root(args)
        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_explicit_project_dir(self, tmp_path: Path) -> None:
        args = argparse.Namespace(project=str(tmp_path))
        result = _resolve_project_root(args)
        assert result == tmp_path.resolve()

    def test_nonexistent_project_falls_back_to_cwd(self) -> None:
        args = argparse.Namespace(project="/nonexistent/path/12345")
        result = _resolve_project_root(args)
        assert result == Path.cwd().resolve()


# ---------------------------------------------------------------------------
# _wire_components
# ---------------------------------------------------------------------------


class TestWireComponents:
    """组件装配测试。"""

    @pytest.fixture
    def mock_config(self) -> Mock:
        """创建完整的 mock Config 对象。"""
        from miaowa.core.config import (
            Config,
            LLMConfig,
            LoggingConfig,
            ProjectConfig,
            ToolsConfig,
            UIConfig,
        )

        config = MagicMock(spec=Config)
        config.llm = LLMConfig(
            provider="deepseek",
            api_key="sk-test-key-for-wire-test",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=4096,
            timeout=120,
        )
        config.ui = UIConfig(theme="dark", syntax_theme="monokai", max_history=1000)
        config.project = ProjectConfig()
        config.tools = ToolsConfig()
        config.logging = LoggingConfig()
        return config

    @pytest.fixture
    def project_root(self, tmp_path: Path) -> Path:
        """创建临时项目目录。"""
        (tmp_path / "src").mkdir(exist_ok=True)
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        return tmp_path.resolve()

    def test_wire_components_succeeds(
        self, mock_config: Mock, project_root: Path
    ) -> None:
        """完整的组件装配应成功并返回 REPL 实例。"""
        repl = _wire_components(mock_config, project_root)
        assert repl is not None
        assert repl.executor is not None
        assert repl.renderer is not None
        assert repl.parser is not None
        assert repl.config is mock_config

    def test_wire_components_registers_4_tools(
        self, mock_config: Mock, project_root: Path
    ) -> None:
        """应注册恰好 4 个内置工具。"""
        repl = _wire_components(mock_config, project_root)
        tools = repl.executor._tools  # type: ignore[union-attr]
        assert len(tools) == 4, f"期望 4 个工具，实际 {len(tools)} 个"

        tool_names = {t.name for t in tools.list_all()}
        expected = {"read_file", "list_directory", "search_files", "analyze_project"}
        assert tool_names == expected

    def test_wire_components_repl_has_model(
        self, mock_config: Mock, project_root: Path
    ) -> None:
        """REPL 的 current_model 应与配置一致。"""
        repl = _wire_components(mock_config, project_root)
        assert repl.current_model == "deepseek-chat"


# ---------------------------------------------------------------------------
# cli_entry
# ---------------------------------------------------------------------------


class TestCliEntry:
    """cli_entry 入口点测试。"""

    def test_version_flag_prints_and_exits(self, capsys) -> None:
        """--version 应输出版本号并正常退出。"""
        with patch.object(sys, "argv", ["miaowa", "--version"]):
            cli_entry()
        captured = capsys.readouterr()
        assert __version__ in captured.out

    @patch("miaowa.main.asyncio.run")
    @patch("miaowa.main.ConfigManager")
    def test_cli_entry_passes_overrides_to_main(
        self, mock_cfg_mgr: Mock, mock_asyncio_run: Mock
    ) -> None:
        """CLI 参数应正确传递到 main()。"""
        with patch.object(sys, "argv", ["miaowa", "--model", "deepseek-reasoner", "--debug"]):
            cli_entry()

        mock_asyncio_run.assert_called_once()
        # 验证 asyncio.run 的调用参数是一个协程
        called_coro = mock_asyncio_run.call_args[0][0]
        assert asyncio.iscoroutine(called_coro)

    def test_cli_entry_handles_keyboard_interrupt(self) -> None:
        """Ctrl+C 应优雅退出（exit code 0）。"""
        with patch.object(sys, "argv", ["miaowa"]):
            with patch("miaowa.main.asyncio.run", side_effect=KeyboardInterrupt):
                with pytest.raises(SystemExit) as exc_info:
                    cli_entry()
                assert exc_info.value.code == 0

    def test_cli_entry_handles_config_missing_error(self) -> None:
        """配置缺失错误应输出中文提示并以 exit code 1 退出。"""
        from miaowa.core.exceptions import ConfigMissingError

        with patch.object(sys, "argv", ["miaowa"]):
            with patch(
                "miaowa.main.asyncio.run",
                side_effect=ConfigMissingError("缺少 API Key", key_name="api_key"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    cli_entry()
                assert exc_info.value.code == 1

    def test_cli_entry_handles_unexpected_exception(self) -> None:
        """未预期异常应以 exit code 1 退出。"""
        with patch.object(sys, "argv", ["miaowa"]):
            with patch(
                "miaowa.main.asyncio.run",
                side_effect=RuntimeError("unknown error"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    cli_entry()
                assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# main 异步主流程
# ---------------------------------------------------------------------------


class TestMainAsync:
    """main() 异步主流程测试。"""

    @pytest.fixture
    def mock_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            debug=False,
            project=None,
            no_color=False,
        )

    @pytest.fixture
    def mock_config(self) -> Mock:
        from miaowa.core.config import (
            Config,
            LLMConfig,
            LoggingConfig,
            UIConfig,
        )

        config = MagicMock(spec=Config)
        config.llm = LLMConfig(
            provider="deepseek",
            api_key="sk-test-key-for-main-test",
            model="deepseek-chat",
        )
        config.ui = UIConfig()
        config.logging = LoggingConfig()
        return config

    @pytest.mark.asyncio
    async def test_main_loads_config_and_starts_repl(
        self, mock_args: argparse.Namespace, mock_config: Mock
    ) -> None:
        """main() 应加载配置并启动 REPL。"""
        with patch("miaowa.main.ConfigManager") as mock_cfg_mgr:
            mock_cfg_mgr.load.return_value = mock_config
            mock_cfg_mgr.load_default.return_value = mock_config

            with patch("miaowa.main.setup_logger"):
                with patch("miaowa.main._wire_components") as mock_wire:
                    mock_repl = AsyncMock()
                    mock_repl.start = AsyncMock()
                    mock_repl.executor.close = AsyncMock()
                    mock_wire.return_value = mock_repl

                    await main({"model": "deepseek-chat"}, mock_args)

                    mock_wire.assert_called_once()
                    mock_repl.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_handles_wire_failure(
        self, mock_args: argparse.Namespace, mock_config: Mock
    ) -> None:
        """组件装配失败应触发 sys.exit(1)。"""
        with patch("miaowa.main.ConfigManager") as mock_cfg_mgr:
            mock_cfg_mgr.load.return_value = mock_config

            with patch("miaowa.main.setup_logger"):
                with patch(
                    "miaowa.main._wire_components",
                    side_effect=ValueError("LLM failed"),
                ):
                    with pytest.raises(SystemExit) as exc_info:
                        await main({}, mock_args)
                    assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_main_debug_sets_log_level(
        self, mock_args: argparse.Namespace, mock_config: Mock
    ) -> None:
        """--debug 标志应将日志级别设为 DEBUG。"""
        mock_args.debug = True

        with patch("miaowa.main.ConfigManager") as mock_cfg_mgr:
            mock_cfg_mgr.load.return_value = mock_config
            mock_cfg_mgr.load_default.return_value = mock_config

            with patch("miaowa.main.setup_logger") as mock_setup:
                with patch("miaowa.main._wire_components") as mock_wire:
                    mock_repl = AsyncMock()
                    mock_repl.start = AsyncMock()
                    mock_repl.executor.close = AsyncMock()
                    mock_wire.return_value = mock_repl

                    await main({}, mock_args)

                    # 配置的日志级别应被设为 DEBUG
                    assert mock_config.logging.level == "DEBUG"
                    mock_setup.assert_called_once_with(mock_config.logging)
