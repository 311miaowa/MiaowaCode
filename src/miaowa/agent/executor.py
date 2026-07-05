"""Agent Executor — ReAct 循环的核心引擎。

PRD §2.5 & §5.2: AgentExecutor 实现 ReAct (Reasoning + Acting) 循环，
协调 LLM 推理、工具调用和上下文管理，是 Agent 运行时的心脏。

流程::

    User Input → Memory → Context → [LLM Stream ⇄ Tool Calls] → Response
                                  └── ReAct 循环 (≤10 轮) ──┘

流式输出::

    executor = AgentExecutor(llm, tools, ctx)
    async for text_chunk in executor.run("帮我分析项目", project_root):
        print(text_chunk, end="")       # 实时打字效果
    result = executor.last_response      # AgentResponse（含统计信息）

容错原则:
    - LLM 调用失败 → 重试或友好降级
    - 工具执行失败 → 将错误作为 tool result 反馈给 LLM 自行修正
    - Token 超限 → 截断历史后继续
    - 所有异常路径均有日志记录，不静默吞错
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miaowa.agent.context import ContextBuilder
from miaowa.core.config import Config
from miaowa.core.types import ProjectCache
from miaowa.core.exceptions import (
    RetriableError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
)
from miaowa.core.logger import get_logger
from miaowa.llm.base import BaseLLMAdapter
from miaowa.llm.tokenizer import TokenCounter
from miaowa.llm.types import Message, StreamChunk, ToolCall
from miaowa.tools.registry import ToolRegistry
from miaowa.tools.validator import ToolValidator

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 模块常量
# ---------------------------------------------------------------------------

_MAX_RETRIES = 2
"""LLM 调用可重试错误的最大重试次数。"""

_RETRY_DELAY_BASE = 1.5
"""重试退避基础延迟（秒），实际延迟 = base * (attempt + 1)。"""


# ============================================================================
# AgentResponse
# ============================================================================


@dataclass
class AgentResponse:
    """Agent 执行响应。

    在 ``AgentExecutor.run()`` 生成器耗尽后，
    通过 ``executor.last_response`` 获取本实例。

    Attributes:
        content: 最终回复文本（完整内容）。
        tool_calls_made: 本次交互中执行的工具调用总次数。
        tokens_used: Token 用量统计。
        cost: 估算费用（美元）。
        iterations: ReAct 循环轮次。
    """

    content: str
    tool_calls_made: int = 0
    tokens_used: dict[str, int] = field(
        default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0}
    )
    cost: float = 0.0
    iterations: int = 0


# ============================================================================
# _StreamResult — 流式收集结果
# ============================================================================


@dataclass
class _StreamResult:
    """一次 LLM 流式调用的收集结果。

    取代原先的 ``chunk._reconstructed_tool_calls`` monkey-patching，
    提供类型安全的结构化数据传递。

    Attributes:
        content_parts: 流式增量文本片段列表。
        tool_calls: 流结束后重构的完整 ToolCall 列表。
        finish_reason: 流终止原因。
    """

    content_parts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None

    @property
    def full_content(self) -> str:
        """将所有内容片段拼接为完整回复文本。"""
        return "".join(self.content_parts)


# ============================================================================
# AgentExecutor
# ============================================================================


class AgentExecutor:
    """Agent 执行器 — ReAct 循环的核心引擎。

    协调 LLM Reasoning 与 Tool Acting 的交替循环，
    直至模型给出最终回复或达到最大迭代次数。

    .. note:: MVP 技术债务

        - **Planner 缺失**: PRD §5.2.1 的 Planner.think() 未独立实现，
          意图分析 + 计划制定已吸收进本类的 ReAct 循环（隐式规划）。
          V2+ 应抽取独立 Planner 组件以实现显式计划生成。
        - **MemoryManager 接口待约束**: memory_manager 参数类型为 ``Any``，
          MemoryManager 已在 session.py 中实现，但缺少 Protocol 接口契约。
          V2 应引入 ``MemoryManagerProtocol`` 替代 ``Any``。

    Attributes:
        MAX_ITERATIONS: 单次用户请求的最大工具调用轮次（默认 10）。
        RUN_TIMEOUT: 单次 run() 的 wall-clock 超时时间（秒），0 表示禁用（默认 300）。
        last_response: 最近一次 ``run()`` 完成后的 AgentResponse。
    """

    MAX_ITERATIONS: int = 10
    """最大工具调用轮次。超出后注入 stop 消息强制 LLM 给出最终回复。"""

    RUN_TIMEOUT: float = 300.0
    """单次 run() 的 wall-clock 超时（秒）。0 表示禁用超时保护。"""

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        tool_manager: ToolRegistry,
        context_builder: ContextBuilder,
        *,
        token_counter: TokenCounter | None = None,
        memory_manager: Any = None,
        config: Config | None = None,
        renderer: Any = None,
    ) -> None:
        """初始化 Agent 执行器。

        Args:
            llm_adapter: LLM 适配器实例。
            tool_manager: 工具注册中心实例。
            context_builder: 上下文构建器实例。
            token_counter: Token 计数器（用于用量统计和费用估算）。
                若为 None，用量统计和费用将被设为 0。
            memory_manager: 对话记忆管理器（可选，None 时跳过记忆操作）。
            config: Miaowa 应用配置对象（可选）。
            renderer: 终端渲染器（可选，用于加载动画反馈）。
                若为 None，跳过动画渲染。
        """
        self._llm = llm_adapter
        self._tools = tool_manager
        self._ctx_builder = context_builder
        self._token_counter = token_counter
        self._memory = memory_manager
        self._config = config
        self._renderer = renderer
        self._validator = ToolValidator()
        self.last_response: AgentResponse | None = None

        logger.info(
            "AgentExecutor 初始化完成: "
            f"MAX_ITERATIONS={self.MAX_ITERATIONS}, "
            f"RUN_TIMEOUT={self.RUN_TIMEOUT}s, "
            f"tools={len(self._tools)}, "
            f"memory={'enabled' if self._memory else 'disabled'}"
        )

    # ------------------------------------------------------------------
    # run — 主入口（AsyncGenerator）
    # ------------------------------------------------------------------

    async def run(
        self,
        user_input: str,
        project_root: Path,
    ) -> AsyncIterator[str]:
        """执行一次完整的 Agent 交互（ReAct 循环），流式产出回复文本。

        ReAct 循环（MVP 隐式规划版本）::

            PRD §2.5 定义了 Understand → Plan → Execute → Synthesize 四阶段。
            MVP 将 Phase 1-2 合并为 LLM 的单次流式推理（隐式规划），
            Phase 3 对应工具执行，Phase 4 对应无 tool_calls 时的直接输出。

            1. 从 memory 加载历史（若有）
            2. ReAct 循环（最多 MAX_ITERATIONS 轮）：
               a. 构建上下文
               b. 流式调用 LLM，逐块 yield delta_content 给调用方
               c. 无 tool_calls → 跳出循环（最终回复）
               d. 有 tool_calls → 并行执行 → 结果追加到历史 → 继续
            3. 达到 MAX_ITERATIONS 时，注入 stop 消息强制 LLM 最终回复
            4. 保存到 memory（若有）
            5. 统计信息写入 self.last_response

        Args:
            user_input: 用户输入的文本。
            project_root: 项目根目录的绝对路径。

        Yields:
            str: LLM 流式输出的文本增量（delta_content）。
        """
        t_start = time.time()
        logger.info(
            f"======== Agent.run() 开始 ========\n"
            f"  input: {user_input[:100]}{'...' if len(user_input) > 100 else ''}\n"
            f"  project: {project_root}\n"
            f"===================================="
        )

        self.last_response = None

        # 对话历史（不含 system 消息）
        history: list[Message] = []
        if self._memory is not None:
            await self._load_memory(history)

        tool_definitions = self._tools.get_definitions()
        original_input = user_input

        total_prompt_tokens = 0
        total_completion_tokens = 0
        tool_calls_made = 0
        final_content = ""

        # -- ReAct 循环 -------------------------------------------------
        hit_max_iterations = True

        try:
            async with self._make_timeout():
                for iteration in range(1, self.MAX_ITERATIONS + 1):
                    logger.info(f"--- 第 {iteration}/{self.MAX_ITERATIONS} 轮 ---")

                    # a. 构建上下文 + 累计输入 token 用量
                    payload = await self._ctx_builder.build(
                        user_message=user_input,
                        history=history,
                        project_root=project_root,
                        tool_definitions=tool_definitions,
                    )
                    if self._token_counter is not None:
                        total_prompt_tokens += self._token_counter.count_messages(
                            payload.messages
                        )

                    # b. 流式调用 LLM（逐块 yield 文本给调用方）
                    stream_result = _StreamResult()
                    chunk: StreamChunk | None = None  # C4: 预初始化

                    async for chunk in self._stream_with_retry(
                        payload.messages, tool_definitions
                    ):
                        if chunk.delta_content:
                            stream_result.content_parts.append(chunk.delta_content)
                            yield chunk.delta_content

                        if chunk.finish_reason:
                            stream_result.finish_reason = chunk.finish_reason

                        # 终端 chunk 携带 _stream_complete 属性（ToolCall 列表）。
                        # 使用 getattr 而非直接属性访问 — 常规流式 chunk 无此属性。
                        # None 表示非终端 chunk，[] 表示流结束但无工具调用。
                        tc = getattr(chunk, '_stream_complete', None)
                        if tc is not None:
                            stream_result.tool_calls = tc

                    # 输出 token 用量估算
                    if self._token_counter is not None:
                        total_completion_tokens += self._token_counter.count_tokens(
                            stream_result.full_content
                        )

                    logger.info(
                        f"第 {iteration} 轮 LLM 响应: "
                        f"content_len={len(stream_result.full_content)}, "
                        f"tool_calls={len(stream_result.tool_calls)}"
                    )

                    # c. 无 tool_calls → 回复完成
                    if not stream_result.tool_calls:
                        final_content = stream_result.full_content
                        hit_max_iterations = False
                        break

                    # d. 执行工具调用
                    logger.info(
                        f"第 {iteration} 轮: 执行 {len(stream_result.tool_calls)} 个工具 -> "
                        f"{', '.join(tc.name for tc in stream_result.tool_calls)}"
                    )

                    history.append(Message(
                        role="assistant",
                        content=stream_result.full_content or "",
                        tool_calls=stream_result.tool_calls,
                    ))

                    tool_results = await self._execute_tool_calls(
                        stream_result.tool_calls
                    )
                    tool_calls_made += len(stream_result.tool_calls)
                    history.extend(tool_results)

                    # 后续轮次不再重复添加原始用户消息
                    user_input = ""

        except asyncio.TimeoutError:
            logger.warning(
                f"Agent.run() 超时 ({self.RUN_TIMEOUT}s)，"
                f"已执行 {tool_calls_made} 次工具调用，返回当前结果"
            )
            if not final_content:
                timeout_msg = (
                    "执行超时。当前已获取的信息可能已足够回答您的问题，"
                    "请查看以上结果或尝试简化请求。"
                )
                final_content = timeout_msg
                yield timeout_msg

        # -- 达到 MAX_ITERATIONS 的处理 ---------------------------------
        if hit_max_iterations:
            logger.warning(
                f"达到最大迭代次数 {self.MAX_ITERATIONS}，"
                f"已执行 {tool_calls_made} 次工具调用，强制生成最终回复"
            )

            history.append(Message(
                role="user",
                content=(
                    "你已达到最大工具调用次数限制。"
                    "请基于当前已获取的所有信息，直接给出最终回答。"
                    "不要再调用任何工具。"
                ),
            ))

            payload = await self._ctx_builder.build(
                user_message="",
                history=history,
                project_root=project_root,
                tool_definitions=[],
            )
            if self._token_counter is not None:
                total_prompt_tokens += self._token_counter.count_messages(
                    payload.messages
                )

            fc = ""
            async for chunk in self._stream_with_retry(payload.messages, None):
                if chunk.delta_content:
                    fc += chunk.delta_content
                    yield chunk.delta_content
            if self._token_counter is not None:
                total_completion_tokens += self._token_counter.count_tokens(fc)

            if fc:
                final_content = fc
            else:
                final_content = (
                    "已达到最大工具调用次数，但无法生成最终回复。"
                    "请尝试简化您的请求。"
                )
                yield final_content

        # -- 收尾 -------------------------------------------------------
        total_tokens = total_prompt_tokens + total_completion_tokens
        cost = (
            self._token_counter.estimate_cost(
                total_prompt_tokens, total_completion_tokens
            )
            if self._token_counter is not None
            else 0.0
        )
        elapsed = time.time() - t_start

        self.last_response = AgentResponse(
            content=final_content,
            tool_calls_made=tool_calls_made,
            tokens_used={
                "prompt": total_prompt_tokens,
                "completion": total_completion_tokens,
                "total": total_tokens,
            },
            cost=cost,
            iterations=iteration,
        )

        if self._memory is not None:
            await self._save_memory(original_input, final_content)

        logger.info(
            f"======== Agent.run() 完成 ({elapsed:.1f}s) ========\n"
            f"  iterations: {iteration}\n"
            f"  tool_calls: {tool_calls_made}\n"
            f"  tokens: prompt={total_prompt_tokens}, "
            f"completion={total_completion_tokens}\n"
            f"  cost: ${cost:.4f}\n"
            f"  content_len: {len(final_content)}\n"
            f"====================================================="
        )

    # ------------------------------------------------------------------
    # 超时管理
    # ------------------------------------------------------------------

    def _make_timeout(self) -> Any:
        """创建超时上下文管理器。

        ``RUN_TIMEOUT`` 为 0 时返回空的 nullcontext（禁用超时），
        否则返回 ``asyncio.timeout(RUN_TIMEOUT)``。
        """
        if self.RUN_TIMEOUT <= 0:
            from contextlib import nullcontext
            return nullcontext()
        return asyncio.timeout(self.RUN_TIMEOUT)

    # ------------------------------------------------------------------
    # 流式 LLM 调用（带重试）
    # ------------------------------------------------------------------

    async def _stream_with_retry(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
    ) -> AsyncIterator[StreamChunk]:
        """流式调用 LLM，逐块产出 StreamChunk（含自动重试）。

        与直接调用 ``self._llm.stream()`` 的区别：
            - 遇到 ``RetriableError``（超时、限流、连接错误）自动重试
            - 流正常结束后，yield 一个终端 chunk（finish_reason=None），
              其 ``_stream_complete`` 属性携带重构的 ToolCall 列表。
              调用方通过 ``chunk._stream_complete`` 获取。
            - 所有中间 chunk 透传，调用方按需处理 delta_content。

        Args:
            messages: 对话消息列表。
            tools: 工具定义列表，None 表示不启用工具。

        Yields:
            StreamChunk: 透传自 LLM 适配器的流式数据块。
            流结束后 yield 一个终端 chunk，携带 ``_stream_complete`` 属性。
        """
        last_error: Exception | None = None

        # -- 思考中动画（Phase 2 §3.2）: 跨重试复用同一实例 ---------
        _thinking = (
            self._renderer.render_thinking()
            if self._renderer is not None
            else None
        )

        for attempt in range(_MAX_RETRIES + 1):
            try:
                if _thinking is not None:
                    _thinking.__enter__()

                # -- 流式迭代（逐块产出，同时累积 tool call 片段）-----
                tool_call_acc: dict[int, dict[str, Any]] = {}
                finish_reason: str | None = None
                _first_content: bool = True

                async for chunk in self._llm.stream(messages, tools):
                    # 首个实质 token — 通知思考中动画（Phase 2 §3.2）
                    # 需要同时检查 delta_content 和 tool_call_delta，
                    # 因为 LLM 首个 chunk 可能是 tool_call 而非文本内容。
                    if _first_content and (
                        chunk.delta_content or chunk.tool_call_delta
                    ):
                        _first_content = False
                        if _thinking is not None:
                            _thinking.first_token_received()

                    # 累积 tool call 增量
                    if chunk.tool_call_delta:
                        delta = chunk.tool_call_delta
                        idx = delta.get("index", 0)
                        if idx not in tool_call_acc:
                            tool_call_acc[idx] = {
                                "id": None, "name": "", "arguments": ""
                            }
                        acc = tool_call_acc[idx]
                        if delta.get("id"):
                            acc["id"] = delta["id"]
                        func = delta.get("function", {})
                        if func.get("name"):
                            acc["name"] += func["name"]
                        if func.get("arguments"):
                            acc["arguments"] += func["arguments"]

                    if chunk.finish_reason:
                        finish_reason = chunk.finish_reason

                    yield chunk

                # -- 流结束后重构 ToolCall 对象 -------------------------
                tool_calls: list[ToolCall] = []

                if finish_reason == "tool_calls":
                    for idx in sorted(tool_call_acc.keys()):
                        acc = tool_call_acc[idx]
                        try:
                            arguments = (
                                json.loads(acc["arguments"])
                                if acc["arguments"] else {}
                            )
                        except json.JSONDecodeError as exc:
                            logger.warning(
                                f"Tool call arguments JSON 解析失败: "
                                f"tool={acc['name']}, error={exc}"
                            )
                            arguments = {}
                        tool_calls.append(ToolCall(
                            id=acc["id"] or f"call_{idx}",
                            name=acc["name"],
                            arguments=arguments,
                        ))

                # 终端 chunk：通过 _stream_complete 属性传递结构化结果
                # （无法从 AsyncIterator 直接 return，故用属性传递）
                terminal = StreamChunk(finish_reason=finish_reason)
                terminal._stream_complete = tool_calls  # type: ignore[attr-defined]
                yield terminal
                return  # 成功，退出重试循环

            except Exception as exc:
                last_error = exc
                is_retriable = isinstance(exc, RetriableError)

                if is_retriable and attempt < _MAX_RETRIES:
                    delay = _RETRY_DELAY_BASE * (attempt + 1)
                    logger.warning(
                        f"LLM 流式调用失败 (可重试, attempt={attempt + 1}): "
                        f"{type(exc).__name__}: {exc} — "
                        f"{delay:.1f}s 后重试"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"LLM 流式调用失败 "
                        f"({'不可重试' if not is_retriable else '重试耗尽'}): "
                        f"{type(exc).__name__}: {exc}"
                    )
                    break

            finally:
                if _thinking is not None:
                    try:
                        _thinking.__exit__(None, None, None)
                    except Exception:
                        pass

        # 所有重试均失败 → yield 错误 chunk
        error_chunk = StreamChunk(
            finish_reason="error",
            delta_content=(
                "抱歉，LLM 服务暂时不可用，请稍后重试。"
                "如需帮助，请检查 API Key 和网络连接。"
            ),
        )
        error_chunk._stream_complete = []  # type: ignore[attr-defined]
        yield error_chunk

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------

    async def _execute_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[Message]:
        """执行工具调用列表，返回 tool role 消息列表。

        每个工具调用独立并行执行。工具执行失败时，
        将错误信息作为 tool result 返回给 LLM，
        使其能够基于错误反馈自行修正参数。

        Args:
            tool_calls: LLM 返回的工具调用列表。

        Returns:
            tool role 消息列表。
        """
        if not tool_calls:
            return []

        logger.info(
            f"开始执行 {len(tool_calls)} 个工具调用: "
            f"{[(tc.name, list(tc.arguments.keys())) for tc in tool_calls]}"
        )

        tasks = [
            self._execute_single_tool(tc, i)
            for i, tc in enumerate(tool_calls)
        ]
        results = await asyncio.gather(*tasks)
        messages = [msg for msg in results if msg is not None]

        logger.info(
            f"工具执行完成: {len(messages)}/{len(tool_calls)} 个完成"
        )
        return messages

    async def _execute_single_tool(
        self, tool_call: ToolCall, index: int
    ) -> Message:
        """执行单个工具调用并包装结果（永不抛异常）。

        执行流程:
            1. 在 ToolRegistry 中查找工具
            2. 使用 ToolValidator 校验参数
            3. 执行工具
            4. 将结果（或错误）包装为 Message(role="tool")

        Args:
            tool_call: 单个工具调用。
            index: 序号（用于日志）。

        Returns:
            Message(role="tool", content=json) — 始终返回。
        """
        t_start = time.time()
        name = tool_call.name
        args = tool_call.arguments

        logger.info(
            f"  [{index}] 执行工具: {name} "
            f"({', '.join(f'{k}={repr(v)[:50]}' for k, v in args.items())})"
        )

        try:
            # 1. 查找工具
            try:
                tool = self._tools.get(name)
            except ToolNotFoundError:
                elapsed = time.time() - t_start
                error_msg = (
                    f"工具 '{name}' 未注册。"
                    f"可用工具: {', '.join(t.name for t in self._tools.list_all())}"
                )
                logger.warning(f"  [{index}] {error_msg} ({elapsed:.2f}s)")
                return Message(
                    role="tool",
                    content=json.dumps(
                        {"success": False, "error": error_msg},
                        ensure_ascii=False,
                    ),
                    tool_call_id=tool_call.id,
                )

            # 2. 校验参数
            try:
                self._validator.validate(tool, args)
            except ToolValidationError as exc:
                elapsed = time.time() - t_start
                logger.warning(
                    f"  [{index}] 参数校验失败: {exc} ({elapsed:.2f}s)"
                )
                return Message(
                    role="tool",
                    content=json.dumps(
                        {
                            "success": False,
                            "error": f"参数校验失败: {exc}",
                            "param_name": exc.param_name,
                            "expected": exc.expected,
                        },
                        ensure_ascii=False,
                    ),
                    tool_call_id=tool_call.id,
                )

            # 3. 执行工具（Phase 2 §3.2: 加载动画）
            _progress = (
                self._renderer.render_tool_progress(
                    name, self._build_args_summary(args)
                )
                if self._renderer is not None
                else None
            )

            if _progress is not None:
                _progress.__enter__()

            try:
                result = await tool.execute(**args)
            finally:
                if _progress is not None:
                    try:
                        # 使用 sys.exc_info() 传递真实异常信息，
                        # 使 ToolProgressContext 在失败时显示"失败"而非"完成"
                        _progress.__exit__(*sys.exc_info())
                    except Exception:
                        pass

            elapsed = time.time() - t_start
            logger.info(
                f"  [{index}] 工具执行成功: {name} ({elapsed:.2f}s)"
            )

            if isinstance(result, str):
                content = result
            else:
                content = json.dumps(result, ensure_ascii=False, default=str)

            return Message(role="tool", content=content, tool_call_id=tool_call.id)

        except ToolExecutionError as exc:
            elapsed = time.time() - t_start
            logger.error(
                f"  [{index}] 工具执行异常: {name} — {exc} ({elapsed:.2f}s)"
            )
            return Message(
                role="tool",
                content=json.dumps(
                    {"success": False, "error": str(exc)},
                    ensure_ascii=False,
                ),
                tool_call_id=tool_call.id,
            )

        except Exception:
            elapsed = time.time() - t_start
            logger.exception(
                f"  [{index}] 工具执行未预期异常: {name} ({elapsed:.2f}s)"
            )
            return Message(
                role="tool",
                content=json.dumps(
                    {
                        "success": False,
                        "error": (
                            f"工具 '{name}' 执行时发生内部错误。"
                            f"请尝试其他方式完成当前任务。"
                        ),
                    },
                    ensure_ascii=False,
                ),
                tool_call_id=tool_call.id,
            )

    # ------------------------------------------------------------------
    # Memory 操作
    # ------------------------------------------------------------------

    async def _load_memory(self, history: list[Message]) -> None:
        """从 memory_manager 加载历史到 history 列表（原地修改）。"""
        try:
            stored = await self._memory.load()
            if stored and isinstance(stored, list):
                for msg_data in stored:
                    if isinstance(msg_data, dict):
                        history.append(Message(
                            role=msg_data.get("role", "user"),
                            content=msg_data.get("content", ""),
                            tool_call_id=msg_data.get("tool_call_id"),
                            tool_calls=msg_data.get("tool_calls"),
                        ))
                    elif isinstance(msg_data, Message):
                        history.append(msg_data)
                logger.info(f"从 memory 加载了 {len(history)} 条历史消息")
        except (OSError, ValueError, TypeError, AttributeError, EOFError):
            logger.exception("从 memory 加载历史失败，使用空历史")

    async def _save_memory(
        self, user_input: str, assistant_reply: str
    ) -> None:
        """将本轮对话保存到 memory_manager。"""
        try:
            await self._memory.save([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_reply},
            ])
            logger.info("对话已保存到 memory")
        except (OSError, ValueError, TypeError, AttributeError, EOFError):
            logger.exception("保存对话到 memory 失败")

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _build_args_summary(args: dict[str, Any], max_length: int = 60) -> str:
        """构建紧凑的参数摘要字符串（用于工具进度显示）。

        Args:
            args: 工具参数字典。
            max_length: 最大字符长度。

        Returns:
            紧凑字符串，如 ``{"path":"/tmp/test.py","query":"hello"}``，
            超长时截断并附加 "…"。
        """
        try:
            summary = json.dumps(
                args, ensure_ascii=False, separators=(",", ":")
            )
        except (TypeError, ValueError):
            summary = str(args)
        if len(summary) > max_length:
            summary = summary[:max_length] + "…"
        return summary

    def invalidate_project_cache(self) -> None:
        """使项目分析缓存失效（委托给 ContextBuilder）。"""
        self._ctx_builder.invalidate_cache()

    def get_project_cache(self) -> ProjectCache | None:
        """获取项目分析缓存（供 REPL 等上层模块使用）。

        Returns:
            ProjectCache 实例；若缓存未构建或不可用则返回 None。
        """
        try:
            cache = getattr(self._ctx_builder, "_project_cache", None)
            if cache is not None and isinstance(cache, ProjectCache):
                return cache
        except Exception:
            pass
        return None

    async def close(self) -> None:
        """关闭 LLM 适配器的连接池（供退出清理时调用）。"""
        if hasattr(self._llm, "close"):
            await self._llm.close()

    def get_status(self) -> dict[str, Any]:
        """获取执行器状态摘要。

        Returns:
            dict 包含 last_response, cache_status, tool_count, max_iterations。
        """
        last = self.last_response
        return {
            "last_response": (
                {
                    "content_preview": (
                        last.content[:100] + "..."
                        if len(last.content) > 100
                        else last.content
                    ),
                    "tool_calls_made": last.tool_calls_made,
                    "tokens_used": last.tokens_used,
                    "cost": last.cost,
                    "iterations": last.iterations,
                }
                if last else None
            ),
            "cache_status": self._ctx_builder.get_cache_status(),
            "tool_count": len(self._tools),
            "max_iterations": self.MAX_ITERATIONS,
            "run_timeout": self.RUN_TIMEOUT,
        }
