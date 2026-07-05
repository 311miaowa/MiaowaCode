"""Miaowa Code CLI 入口 — 依赖注入与 REPL 启动。

本模块是 Miaowa Code 的顶层入口点，负责：
    1. 解析命令行参数（argparse）
    2. 加载配置（ConfigManager.load）
    3. 设置日志（setup_logger）
    4. 初始化所有组件（依赖注入）
    5. 启动 REPL 交互循环

Usage::

    # 通过 poetry scripts 入口
    miaowa

    # 直接调用
    python -m miaowa.main

    # 带参数
    miaowa --model deepseek-reasoner --project /path/to/project

Typical startup flow::

    cli_entry()                  # 同步入口（parse args + 异常处理）
      └── asyncio.run(main())    # 异步主流程
            ├── ConfigManager.load(cli_overrides)
            ├── setup_logger(config.logging)
            ├── _wire_components(config)
            │     ├── TokenCounter
            │     ├── MemoryManager → SessionManager
            │     ├── DeepSeekAdapter (LLM)
            │     ├── 4 tools → ToolRegistry
            │     ├── ContextBuilder
            │     ├── AgentExecutor
            │     ├── Renderer + CommandParser
            │     └── REPL
            └── await repl.start(project_root)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from miaowa import __version__
from miaowa.agent.context import ContextBuilder
from miaowa.agent.executor import AgentExecutor
from miaowa.agent.session import MemoryManager, SessionManager
from miaowa.cli.parser import CommandParser
from miaowa.cli.renderer import Renderer
from miaowa.cli.repl import REPL
from miaowa.core.config import ConfigManager
from miaowa.core.exceptions import ConfigFormatError, ConfigMissingError
from miaowa.core.logger import get_logger, setup_logger
from miaowa.llm.deepseek import DeepSeekAdapter
from miaowa.llm.tokenizer import TokenCounter
from miaowa.prompts.manager import PromptManager
from miaowa.tools.analyzer import AnalyzeProjectTool
from miaowa.tools.filesystem import ListDirectoryTool, ReadFileTool
from miaowa.tools.registry import ToolRegistry
from miaowa.tools.search import SearchFilesTool

logger = get_logger(__name__)


# =============================================================================
# cli_entry — 同步入口（poetry console_scripts 调用）
# =============================================================================


def cli_entry() -> None:
    """Miaowa Code 的同步入口点。

    由 ``pyproject.toml`` 的 ``[tool.poetry.scripts]`` 配置调用::

        [tool.poetry.scripts]
        miaowa = "miaowa.main:cli_entry"

    职责：
        1. 解析命令行参数
        2. 处理 --version（输出版本号后退出）
        3. 将 CLI 参数传递给 asyncio.run(main(cli_overrides))
        4. 捕获顶层异常并给出中文错误提示
    """
    # -- Windows 终端强制 UTF-8 编码（修复 GBK 乱码）----------------------
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = _build_arg_parser()
    args = parser.parse_args()

    # --version：输出版本号后立即退出
    if args.version:
        print(f"Miaowa Code v{__version__}")
        return

    # 构建 CLI 覆盖字典（仅包含用户在命令行显式指定的参数）
    cli_overrides = _build_cli_overrides(args)

    try:
        asyncio.run(main(cli_overrides, args))
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    except (ConfigMissingError, ConfigFormatError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        print("Check:", file=sys.stderr)
        print("  1. MIAOWA_API_KEY environment variable is set", file=sys.stderr)
        print("  2. ~/.miaowa/config.yaml is valid", file=sys.stderr)
        print("  3. API key in .env file is valid", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        print("Use --debug for detailed error information.", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# main — 异步主流程
# =============================================================================


async def main(
    cli_overrides: dict[str, str],
    args: argparse.Namespace,
) -> None:
    """Miaowa Code 的异步主入口。

    执行流程::

        1. 加载配置（ConfigManager.load）
        2. 设置日志（setup_logger）
        3. 确定项目根目录
        4. 初始化所有组件（_wire_components）
        5. 启动 REPL 交互循环

    Args:
        cli_overrides: CLI 参数覆盖字典（如 ``{"model": "deepseek-reasoner"}``）。
        args: argparse 解析后的 Namespace 对象（用于获取 --project 等非覆盖参数）。
    """
    # -- 1. 加载配置 -----------------------------------------------------
    try:
        config = ConfigManager.load(cli_overrides=cli_overrides)
    except ConfigFormatError as exc:
        print(f"Warning: config file format error: {exc}", file=sys.stderr)
        print("Using default config...", file=sys.stderr)
        config = ConfigManager.load_default()

    # -- 2. 设置日志 -----------------------------------------------------
    if args.debug:
        config.logging.level = "DEBUG"
    setup_logger(config.logging)
    logger.info(
        f"Miaowa Code v{__version__} starting... "
        f"model={config.llm.model}, "
        f"base_url={config.llm.base_url}"
    )

    # -- 3. 确定项目根目录 -----------------------------------------------
    project_root = _resolve_project_root(args)

    # -- 4. 初始化所有组件（依赖注入）-----------------------------------
    try:
        repl = _wire_components(config, project_root, no_color=args.no_color)
    except ConfigMissingError:
        raise  # 重新抛出，由 cli_entry 统一处理
    except Exception as exc:
        logger.exception("Component initialization failed")
        print(f"\nError: component initialization failed: {exc}", file=sys.stderr)
        print("Check:", file=sys.stderr)
        print("  1. API key is correctly configured", file=sys.stderr)
        print("  2. Network connection is available (if using remote API)", file=sys.stderr)
        print("  3. Project directory exists and is readable", file=sys.stderr)
        sys.exit(1)

    # -- 5. 启动 REPL ----------------------------------------------------
    try:
        await repl.start(project_root)
    except Exception:
        logger.exception("REPL runtime error")
        raise
    finally:
        await _cleanup(repl)


# =============================================================================
# 依赖注入 — _wire_components
# =============================================================================


def _wire_components(config, project_root: Path, *, no_color: bool = False) -> REPL:
    """初始化所有组件并完成依赖注入。

    初始化顺序（按依赖关系自底向上）::

        TokenCounter → MemoryManager → SessionManager
        DeepSeekAdapter → ContextBuilder
        ToolRegistry（4 个工具）→ AgentExecutor
        Renderer → CommandParser → REPL

    每个初始化步骤都有独立的 try/except，失败时给出明确的中文错误提示。

    Args:
        config: Config 应用配置实例。
        project_root: 项目根目录路径。
        no_color: 若为 True，禁用终端彩色输出。

    Returns:
        已装配完成的 REPL 实例。

    Raises:
        ConfigMissingError: API Key 缺失时传递到上层处理。
    """
    # ------------------------------------------------------------------
    # 1. Token 计数器
    # ------------------------------------------------------------------
    try:
        token_counter = TokenCounter(model=config.llm.model)
    except Exception as exc:
        print(f"Error: Token counter init failed: {exc}", file=sys.stderr)
        raise

    # ------------------------------------------------------------------
    # 2. 对话记忆 & 会话管理
    # ------------------------------------------------------------------
    try:
        memory_manager = MemoryManager()
        session_manager = SessionManager()
    except Exception as exc:
        print(f"Error: Session manager init failed: {exc}", file=sys.stderr)
        raise

    # ------------------------------------------------------------------
    # 3. LLM 适配器（DeepSeek）
    # ------------------------------------------------------------------
    try:
        llm_adapter = DeepSeekAdapter(config.llm)
    except Exception as exc:
        print(f"Error: LLM adapter init failed: {exc}", file=sys.stderr)
        print("Check API Key and base_url configuration.", file=sys.stderr)
        raise

    # ------------------------------------------------------------------
    # 4. 上下文构建器
    # ------------------------------------------------------------------
    try:
        context_builder = ContextBuilder(
            config=config,
            token_counter=token_counter,
            prompt_manager=PromptManager,
        )
    except Exception as exc:
        print(f"Error: Context builder init failed: {exc}", file=sys.stderr)
        raise

    # ------------------------------------------------------------------
    # 5. 工具注册中心（注册 4 个内置工具）
    # ------------------------------------------------------------------
    try:
        tool_registry = ToolRegistry()

        # 文件系统工具
        tool_registry.register(ReadFileTool(project_root=project_root, config=config))
        tool_registry.register(ListDirectoryTool(project_root=project_root, config=config))
        # 搜索工具
        tool_registry.register(SearchFilesTool(project_root=project_root, config=config))
        # 项目分析工具
        tool_registry.register(AnalyzeProjectTool(project_root=project_root, config=config))

        logger.info(f"Registered {len(tool_registry)} tools")
    except Exception as exc:
        print(f"Error: Tool registry init failed: {exc}", file=sys.stderr)
        raise

    # ------------------------------------------------------------------
    # 6. Agent 执行器
    # ------------------------------------------------------------------
    try:
        executor = AgentExecutor(
            llm_adapter=llm_adapter,
            tool_manager=tool_registry,
            context_builder=context_builder,
            token_counter=token_counter,
            memory_manager=memory_manager,
            config=config,
        )
    except Exception as exc:
        print(f"Error: Agent executor init failed: {exc}", file=sys.stderr)
        raise

    # ------------------------------------------------------------------
    # 7. 终端渲染器
    # ------------------------------------------------------------------
    try:
        renderer = Renderer(config=config.ui)
        if no_color:
            from rich.console import Console
            renderer.console = Console(
                force_terminal=True, no_color=True, highlight=False
            )
    except Exception as exc:
        print(f"Error: Renderer init failed: {exc}", file=sys.stderr)
        raise

    # Phase 2 §3.2: 将 renderer 注入 executor 以支持加载动画
    executor._renderer = renderer

    # ------------------------------------------------------------------
    # 8. 命令解析器
    # ------------------------------------------------------------------
    try:
        parser = CommandParser()
    except Exception as exc:
        print(f"Error: Command parser init failed: {exc}", file=sys.stderr)
        raise

    # ------------------------------------------------------------------
    # 9. REPL（组装所有组件）
    # ------------------------------------------------------------------
    try:
        repl = REPL(
            executor=executor,
            renderer=renderer,
            parser=parser,
            config=config,
            session_manager=session_manager,
        )
    except Exception as exc:
        print(f"Error: REPL init failed: {exc}", file=sys.stderr)
        raise

    logger.info("All components initialized, ready to start REPL")
    return repl


# =============================================================================
# 清理
# =============================================================================


async def _cleanup(repl: REPL) -> None:
    """清理资源（释放 LLM 连接池等）。

    Args:
        repl: REPL 实例。
    """
    try:
        await repl.executor.close()
    except Exception:
        logger.debug("Resource cleanup error", exc_info=True)


# =============================================================================
# 内部辅助函数
# =============================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        配置好的 ArgumentParser 实例。
    """
    parser = argparse.ArgumentParser(
        prog="miaowa",
        description="Miaowa Code — 轻量级终端 AI Agent 工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  miaowa                             在当前目录启动\n"
            "  miaowa --project /path/to/project   指定项目目录\n"
            "  miaowa --model deepseek-reasoner    指定模型\n"
            "  miaowa --debug                      调试模式启动\n"
            "\n"
            "配置文件加载顺序（优先级从低到高）:\n"
            "  默认值 < config.yaml < .env < 环境变量 < CLI 参数"
        ),
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="指定 LLM 模型名称（如 deepseek-chat、deepseek-reasoner）",
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="DeepSeek API Key（亦可设置环境变量 MIAOWA_API_KEY）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API 基础地址（默认 https://api.deepseek.com/v1）",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        default=False,
        help="启用调试模式（DEBUG 日志级别 + REPL 调试信息）",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="禁用终端彩色输出（Rich 无条件回退为纯文本）",
    )
    parser.add_argument(
        "--project", "-p",
        type=str,
        default=None,
        help="指定项目根目录路径（默认当前目录）",
    )
    parser.add_argument(
        "--version", "-V",
        action="store_true",
        default=False,
        help="显示版本号并退出",
    )

    return parser


def _build_cli_overrides(args: argparse.Namespace) -> dict[str, str]:
    """从 argparse Namespace 构建 ConfigManager.load() 可用的覆盖字典。

    仅包含用户显式指定的参数（非 None），避免用 None 覆盖配置文件中
    的有效值。

    Args:
        args: argparse 解析后的参数。

    Returns:
        CLI 覆盖字典，键与 ConfigManager.load(cli_overrides) 兼容。
    """
    overrides: dict[str, str] = {}

    if args.model is not None:
        overrides["model"] = args.model
    if args.api_key is not None:
        overrides["api_key"] = args.api_key
    if args.base_url is not None:
        overrides["base_url"] = args.base_url
    if args.debug:
        overrides["log_level"] = "DEBUG"
    if getattr(args, "no_color", False):
        overrides["no_color"] = "true"

    return overrides


def _resolve_project_root(args: argparse.Namespace) -> Path:
    """确定项目根目录路径。

    优先级:
        1. --project CLI 参数
        2. 当前工作目录

    Args:
        args: argparse 解析后的参数。

    Returns:
        解析后的绝对路径。
    """
    if args.project:
        project_root = Path(args.project).expanduser().resolve()
    else:
        project_root = Path.cwd().resolve()

    if not project_root.is_dir():
        logger.error(
            f"Project directory not found: {project_root}, falling back to cwd"
        )
        return Path.cwd().resolve()

    logger.info(f"Project root: {project_root}")
    return project_root
