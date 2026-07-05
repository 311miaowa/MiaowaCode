"""REPL 交互循环 — 终端 AI 助手的主界面。

Miaowa Code 的核心交互层：基于 prompt_toolkit 的 PromptSession
提供命令提示、自动补全、多行输入和历史记录，通过 AgentExecutor
执行自然语言交互，通过 CommandParser 处理元命令。

Typical usage::

    from miaowa.cli.repl import REPL

    repl = REPL(executor, renderer, parser, config, session_manager)
    await repl.start(Path.cwd())
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from miaowa.cli.history import CommandHistory
from miaowa.core.logger import get_logger

if TYPE_CHECKING:
    from miaowa.agent.executor import AgentExecutor
    from miaowa.agent.session import SessionManager
    from miaowa.cli.parser import CommandParser
    from miaowa.cli.renderer import Renderer
    from miaowa.core.config import Config
    from miaowa.core.types import ProjectCache

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# prompt_toolkit 样式
_PROMPT_STYLE = Style.from_dict({
    "prompt": "#888888",       # 主提示符 "> "
    "separator": "#555555",    # 续行提示符
})

# 提示符文本
_PROMPT_MESSAGE = [
    ("class:prompt", "> "),
]

# 多行输入模式下的续行提示符
_CONTINUATION_MESSAGE = [
    ("class:separator", "  … "),
]

# 项目缓存后台任务的延迟（s），确保欢迎界面先渲染完成
_CACHE_BUILD_DELAY: float = 0.5

# Ctrl+D 退出前的连续按下计数阈值
_EOF_EXIT_COUNT: int = 2


# ============================================================================
# REPL
# ============================================================================


class REPL:
    """REPL 交互循环 — 终端 AI 助手的主界面。

    协调 CommandParser → AgentExecutor → Renderer 的数据流，
    通过 prompt_toolkit 提供完整的命令行交互体验。

    Attributes:
        executor: Agent 执行器，负责 ReAct 循环和 LLM 交互。
        renderer: 终端渲染器，格式化输出。
        parser: 命令解析器，区分元命令与自然语言。
        config: 应用配置。
        sessions: 会话管理器，维护对话历史。
        current_model: 当前使用的模型名称。
    """

    def __init__(
        self,
        executor: AgentExecutor,
        renderer: Renderer,
        parser: CommandParser,
        config: Config,
        session_manager: SessionManager,
    ) -> None:
        """初始化 REPL。

        Args:
            executor: AgentExecutor 实例。
            renderer: Renderer 实例。
            parser: CommandParser 实例。
            config: Config 应用配置实例。
            session_manager: SessionManager 实例。
        """
        self.executor = executor
        self.renderer = renderer
        self.parser = parser
        self.config = config
        self.sessions = session_manager

        # 运行时状态
        self.current_model: str = config.llm.model
        self._session_start: float = 0.0
        self._total_tokens: dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}
        self._total_cost: float = 0.0
        self._turn_count: int = 0
        self._debug_mode: bool = False
        self._eof_count: int = 0

        # 命令历史
        self._history = CommandHistory(max_size=config.ui.max_history)

        # PromptSession（延迟创建，因为需要 parser 的已知命令列表）
        self._prompt_session: PromptSession | None = None

        logger.info(
            f"REPL initialized: model={self.current_model}, "
            f"max_history={config.ui.max_history}"
        )

    # ------------------------------------------------------------------
    # start — 主入口
    # ------------------------------------------------------------------

    async def start(self, project_root: Path) -> None:
        """启动 REPL 交互循环。

        执行流程::

            1. 渲染欢迎界面
            2. 异步构建项目缓存（后台，不阻塞用户输入）
            3. 主循环：
               a. 显示提示符 ">> "
               b. 解析输入 → meta 命令或 natural 语言
               c. Ctrl+C 中断当前操作 / Ctrl+D 退出
            4. 退出时显示会话统计与告别界面

        Args:
            project_root: 项目根目录的绝对路径。
        """
        self._session_start = time.monotonic()

        # -- 1. 欢迎界面 -----------------------------------------------
        self.renderer.render_welcome(
            model=self.current_model,
            provider=self.config.llm.provider,
            project_root=str(project_root),
        )

        # -- 2. 异步构建项目缓存（后台任务） ---------------------------
        cache_task = asyncio.create_task(
            self._build_project_cache_async(project_root)
        )

        # -- 3. 初始化 PromptSession -----------------------------------
        self._prompt_session = self._get_prompt_session()

        # -- 4. 主循环 -------------------------------------------------
        running = True

        while running:
            try:
                user_input = await self._read_input()
            except EOFError:
                # Ctrl+D → 需连续两次确认退出
                if await self._handle_eof():
                    break
                continue
            except KeyboardInterrupt:
                # Ctrl+C → 中断当前操作（不退出）
                self._handle_interrupt()
                continue

            if user_input is None:
                # prompt_toolkit 返回 None — 视为 EOF
                if await self._handle_eof():
                    break
                continue

            # 正常输入时重置 EOF 计数器
            self._eof_count = 0

            # 空输入跳过
            if not user_input.strip():
                continue

            # 记录到历史
            self._history.append(user_input)

            # 解析输入
            parsed = self.parser.parse(user_input)

            if parsed.parse_error:
                self.renderer.render_warning(parsed.parse_error)

            if parsed.type == "meta":
                if parsed.command is None:
                    continue

                if not parsed.is_known:
                    self.renderer.render_warning(
                        f"Unknown command: {parsed.command}\n"
                        f"Type /help for available commands"
                    )
                    continue

                should_exit = await self._handle_meta_command(parsed)
                if should_exit:
                    running = False
            else:
                await self._handle_natural_input(
                    parsed.content or user_input, project_root
                )

        # -- 5. 等待缓存任务完成 ---------------------------------------
        if not cache_task.done():
            cache_task.cancel()
            try:
                await cache_task
            except asyncio.CancelledError:
                pass

        # -- 6. 会话统计 & 告别界面 ------------------------------------
        self._show_session_stats()

    # ------------------------------------------------------------------
    # 自然语言输入处理
    # ------------------------------------------------------------------

    async def _handle_natural_input(
        self, user_input: str, project_root: Path
    ) -> None:
        """处理自然语言输入：调用 AgentExecutor 并流式渲染回复。

        执行流程:
            1. 调用 ``executor.run()``（AsyncGenerator），逐块接收文本
            2. 每块通过 ``renderer.render_stream_chunk()`` 实时输出
            3. 流结束后调用 ``renderer.flush_stream()`` 确保全部显示
            4. 从 ``executor.last_response`` 获取统计信息并渲染

        Args:
            user_input: 用户输入的自然语言文本。
            project_root: 项目根目录路径。
        """
        self._turn_count += 1

        if self._debug_mode:
            self.renderer.render_info(
                f"[DEBUG] Turn #{self._turn_count} | Model: {self.current_model}"
            )

        try:
            # Phase 2 §3.2: 状态行 — 处理中
            self.renderer.render_status_line(
                f"处理中 (turn #{self._turn_count})..."
            )

            # 流式执行 + 实时渲染
            async for chunk in self.executor.run(user_input, project_root):
                self.renderer.render_stream_chunk(chunk)

            self.renderer.flush_stream()
            self.renderer.console.print()  # 换行

            # Phase 2 §3.2: 清除状态行
            self.renderer.render_status_line("")

            # 统计信息
            response = self.executor.last_response
            if response is not None:
                self._accumulate_stats(response.tokens_used, response.cost)
                self.renderer.render_token_usage(
                    response.tokens_used, response.cost
                )

                if self._debug_mode:
                    self.renderer.render_info(
                        f"[DEBUG] iterations={response.iterations}, "
                        f"tool_calls={response.tool_calls_made}"
                    )

        except asyncio.CancelledError:
            # Ctrl+C 中断时，asyncio 会取消当前任务
            self.renderer.render_warning("Task interrupted")
            self.renderer.flush_stream()
            self.renderer.render_status_line("")
            # 仍尝试显示已有的统计信息
            response = self.executor.last_response
            if response is not None:
                self._accumulate_stats(response.tokens_used, response.cost)
        except Exception:
            logger.exception("Natural language input handling error")
            self.renderer.render_error(
                "Internal error processing request. Check logs for details."
            )
            self.renderer.render_status_line("")

    # ------------------------------------------------------------------
    # 元命令处理
    # ------------------------------------------------------------------

    async def _handle_meta_command(self, parsed: Any) -> bool:
        """处理元命令。

        根据命令名称分发到对应处理器。
        所有处理器返回 False 表示继续运行，仅 /quit 和 /exit 返回 True。

        Args:
            parsed: ParsedCommand 实例。

        Returns:
            True 表示需要退出 REPL，False 表示继续运行。
        """
        command = parsed.command
        args = parsed.args or []

        try:
            if command == "/quit" or command == "/exit":
                return True

            elif command == "/clear":
                self.sessions.current.clear()
                self.renderer.render_info("Conversation history cleared")
                self._total_tokens = {"prompt": 0, "completion": 0, "total": 0}
                self._total_cost = 0.0
                self._turn_count = 0

            elif command == "/help":
                self._handle_help(args)

            elif command == "/model":
                self._handle_model(args)

            elif command == "/tokens":
                self._handle_tokens()

            elif command == "/cost":
                self._handle_cost()

            elif command == "/cache":
                self._handle_cache()

            elif command == "/debug":
                self._debug_mode = not self._debug_mode
                status = "ON" if self._debug_mode else "OFF"
                self.renderer.render_info(f"Debug mode: {status}")

            else:
                self.renderer.render_warning(
                    f"Unknown command: {command}\nType /help for available commands"
                )

        except Exception as exc:
            logger.exception(f"Meta command handling error: {command}")
            self.renderer.render_error(
                f"Error executing {command}: {exc}"
            )

        return False

    # ------------------------------------------------------------------
    # 元命令处理器
    # ------------------------------------------------------------------

    def _handle_help(self, args: list[str]) -> None:
        """显示帮助文本。

        Args:
            args: 命令参数，可指定特定命令名以获取详细帮助。
        """
        if args:
            target = args[0]
            if not target.startswith("/"):
                target = f"/{target}"
            desc = self.parser.get_command_description(target)
            if desc is not None:
                self.renderer.render_info(f"{target} — {desc}")
            else:
                self.renderer.render_warning(f"Unknown command: {target}")
        else:
            help_text = self.parser.get_help_text()
            self.renderer.console.print(help_text)

    def _handle_model(self, args: list[str]) -> None:
        """显示或切换当前模型。

        Args:
            args: 为空时显示当前模型；否则第一个参数为新模型名称。
        """
        if not args:
            self.renderer.render_info(f"Current model: {self.current_model}")
            return

        new_model: str = args[0]
        old_model = self.current_model
        self.current_model = new_model
        self.config.llm.model = new_model
        self.renderer.render_info(
            f"Model: {old_model} -> {new_model}"
        )
        logger.info(f"Model switch: {old_model} -> {new_model}")

    def _handle_tokens(self) -> None:
        """显示当前会话的 Token 用量统计。"""
        self.renderer.render_token_usage(self._total_tokens, self._total_cost)

    def _handle_cost(self) -> None:
        """显示当前会话的 API 费用统计。"""
        self.renderer.render_token_usage(self._total_tokens, self._total_cost)

    def _handle_cache(self) -> None:
        """显示项目缓存状态。"""
        status = self.executor.get_status()
        cache_info = status.get("cache_status", {})
        if cache_info:
            lines = [f"  {k}: {v}" for k, v in cache_info.items()]
            self.renderer.render_info(
                "Cache status:\n" + "\n".join(lines)
            )
        else:
            self.renderer.render_info("Cache: not built or invalidated")

    # ------------------------------------------------------------------
    # 输入读取
    # ------------------------------------------------------------------

    async def _read_input(self) -> str | None:
        """从 prompt_toolkit 读取一行用户输入。

        Returns:
            用户输入字符串，EOF 时返回 None。

        Raises:
            EOFError: 用户按下 Ctrl+D 时。
            KeyboardInterrupt: 用户按下 Ctrl+C 时。
        """
        assert self._prompt_session is not None
        return await self._prompt_session.prompt_async(
            message=_PROMPT_MESSAGE,
            style=_PROMPT_STYLE,
        )

    # ------------------------------------------------------------------
    # PromptSession 工厂
    # ------------------------------------------------------------------

    def _get_prompt_session(self) -> PromptSession:
        """创建 prompt_toolkit PromptSession。

        功能:
            - 元命令自动补全（WordCompleter）
            - 自定义键绑定（Ctrl+J 提交多行输入等）
            - 命令历史
            - 多行输入支持

        Returns:
            配置好的 PromptSession 实例。
        """
        # 自动补全：所有已注册元命令及其描述
        completions = self.parser.get_command_completions()
        completer = WordCompleter(
            list(completions.keys()),
            ignore_case=True,
            sentence=True,
            meta_dict=completions,
        )

        # 键绑定
        kb = KeyBindings()

        @kb.add("c-d", eager=True)
        def _(event: Any) -> None:
            """Ctrl+D: 无条件触发 EOF，优先级高于默认字符删除行为。"""
            event.app.exit(exception=EOFError())

        @kb.add("c-z", eager=True)
        def _(event: Any) -> None:
            """Ctrl+Z (Windows EOF): 无条件触发 EOF。"""
            event.app.exit(exception=EOFError())

        @kb.add("escape", "enter")
        def _(event: Any) -> None:
            """Esc + Enter: 插入换行（多行输入）。"""
            event.current_buffer.insert_text("\n")

        @kb.add("c-j")
        def _(event: Any) -> None:
            """Ctrl+J: 插入换行（替代方案）。"""
            event.current_buffer.insert_text("\n")

        session: PromptSession = PromptSession(
            history=self._history.pt_history,
            completer=completer,
            key_bindings=kb,
            multiline=False,  # Enter 直接提交，Esc+Enter / Ctrl+J 换行
            complete_while_typing=True,
            enable_history_search=True,
        )
        return session

    # ------------------------------------------------------------------
    # Ctrl+C / Ctrl+D 处理
    # ------------------------------------------------------------------

    def _handle_interrupt(self) -> None:
        """处理 Ctrl+C 中断（不退出 REPL）。"""
        self.renderer.console.print()
        self.renderer.render_warning("Interrupted. Type /quit or press Ctrl+D to exit.")

    async def _handle_eof(self) -> bool:
        """处理 Ctrl+D（EOF）。

        需要连续按下两次 Ctrl+D 确认退出，防止误触。
        首次按下显示提示，连续两次确认后才真正退出。

        Returns:
            True 表示确认退出，False 表示仍需继续。
        """
        self._eof_count += 1
        if self._eof_count < _EOF_EXIT_COUNT:
            self.renderer.render_warning(
                "Press Ctrl+D again or type /quit to exit"
            )
            return False
        logger.info("Received consecutive EOF (Ctrl+D), exiting REPL")
        return True

    # ------------------------------------------------------------------
    # 会话统计
    # ------------------------------------------------------------------

    def _show_session_stats(self) -> None:
        """计算并显示会话结束统计信息。"""
        elapsed = time.monotonic() - self._session_start

        # 格式化时长
        if elapsed < 60:
            duration = f"{elapsed:.0f}s"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            duration = f"{minutes}m {seconds}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            duration = f"{hours}h {minutes}m"

        stats = {
            "duration": duration,
            "tokens": self._total_tokens,
            "cost": self._total_cost,
            "turns": self._turn_count,
        }
        self.renderer.render_goodbye(stats)

    # ------------------------------------------------------------------
    # 统计累加
    # ------------------------------------------------------------------

    def _accumulate_stats(
        self, tokens_used: dict[str, int], cost: float
    ) -> None:
        """累加单次交互的 Token 用量和费用到会话总计。

        Args:
            tokens_used: 单次交互的 Token 用量。
            cost: 单次交互的费用。
        """
        for key in ("prompt", "completion", "total"):
            self._total_tokens[key] += tokens_used.get(key, 0)
        self._total_cost += cost

    # ------------------------------------------------------------------
    # 项目缓存异步构建
    # ------------------------------------------------------------------

    async def _build_project_cache_async(self, project_root: Path) -> None:
        """后台异步构建项目缓存。

        延迟一小段时间后执行，避免与欢迎界面渲染竞争终端输出。
        缓存构建成功后将结果送入 Renderer 展示。

        Args:
            project_root: 项目根目录路径。
        """
        await asyncio.sleep(_CACHE_BUILD_DELAY)

        try:
            # 触发 ContextBuilder 的缓存构建（通过 invalidate + get_status）
            self.executor.invalidate_project_cache()

            # 给缓存构建一点时间（同步操作很快，但作为异步占位）
            await asyncio.sleep(0)

            # 从 status 中提取 ProjectCache 信息
            status = self.executor.get_status()
            cache_status = status.get("cache_status", {})

            if cache_status and cache_status.get("cache_hit") is False:
                # 缓存刚构建完成，尝试获取项目上下文
                cache = await self._try_get_project_cache()
                if cache is not None and (
                    cache.tech_stack or cache.key_files or cache.structure
                ):
                    self.renderer.render_project_context(cache)

        except asyncio.CancelledError:
            # 任务被取消（用户可能已退出）
            pass
        except Exception:
            logger.exception("Background project cache build failed")

    async def _try_get_project_cache(self) -> ProjectCache | None:
        """尝试从 AgentExecutor 获取项目缓存。

        Returns:
            ProjectCache 或 None（无法获取时）。
        """
        try:
            return self.executor.get_project_cache()
        except Exception:
            return None
