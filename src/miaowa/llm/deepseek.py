"""DeepSeek API 适配器 — 基于 OpenAI 兼容接口的 LLM 客户端封装。

基于 PRD §3.3 和 §5.2.4 设计，使用 AsyncOpenAI SDK 调用 DeepSeek Chat API。
DeepSeek 的 API 与 OpenAI Chat Completions 接口兼容（base_url + api_key 模式），
因此本适配器在 openai 库基础上做了一层薄封装，核心职责为:

    1. Message → OpenAI dict 线格式转换
    2. 非流式 & 流式 API 请求
    3. 响应 → ChatResponse / StreamChunk 解析
    4. 底层 SDK 异常 → LLMError 体系映射
    5. Token 近似计数 & 模型信息查询
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import RateLimitError as OpenAIRateLimitError

from miaowa.core.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMTimeoutError,
    LLMToolCallParseError,
)
from miaowa.core.logger import get_logger
from miaowa.llm.base import BaseLLMAdapter
from miaowa.llm.tokenizer import _count_chars_and_tokens
from miaowa.llm.types import (
    ChatResponse,
    FinishReason,
    Message,
    ModelInfo,
    StreamChunk,
    ToolCall,
    ToolDef,
)

if TYPE_CHECKING:
    from miaowa.core.config import LLMConfig

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 异常映射 — 将 openai SDK 异常映射为 miaowa LLMError 子类
# ---------------------------------------------------------------------------

def _map_openai_error(err: Exception) -> LLMError:
    """将 openai SDK 异常映射为 miaowa LLMError 体系。

    映射规则:
        - openai.AuthenticationError  → LLMAuthenticationError (401)
        - openai.RateLimitError       → LLMRateLimitError      (429, 可重试)
        - openai.APITimeoutError      → LLMTimeoutError        (可重试)
        - openai.APIConnectionError   → LLMConnectionError     (可重试)
        - openai.APIError             → 按 HTTP 状态码细分:
            - 401 → LLMAuthenticationError
            - 429 → LLMRateLimitError
            - 5xx → LLMConnectionError (可重试)
            - 其他 → LLMError (通用)
        - 其他 Exception              → LLMError (通用包装)

    Args:
        err: openai SDK 抛出的原始异常。

    Returns:
        对应的 miaowa LLMError 子类实例。
    """
    msg = str(err)

    # -- Authentication (401) ---------------------------------------------
    if isinstance(err, OpenAIAuthenticationError):
        return LLMAuthenticationError(
            f"DeepSeek API 认证失败: {msg}",
            status_code=err.status_code,
        )

    # -- Rate Limit (429, 可重试) -----------------------------------------
    if isinstance(err, OpenAIRateLimitError):
        return LLMRateLimitError(
            f"DeepSeek API 速率限制: {msg}",
            retry_after=_extract_retry_after(err),
        )

    # -- Timeout (可重试) -------------------------------------------------
    if isinstance(err, APITimeoutError):
        return LLMTimeoutError(
            f"DeepSeek API 请求超时: {msg}",
        )

    # -- Connection Error (DNS / TCP / TLS 失败, 可重试) ------------------
    if isinstance(err, APIConnectionError):
        return LLMConnectionError(
            f"DeepSeek API 连接失败: {msg}",
            host="api.deepseek.com",
        )

    # -- General API Error — 按 HTTP 状态码细分 --------------------------
    if isinstance(err, APIError):
        status: int | None = getattr(err, "code", None)
        if status == 401:
            return LLMAuthenticationError(
                f"DeepSeek API 认证失败 (HTTP {status}): {msg}",
                status_code=status,
            )
        if status == 429:
            return LLMRateLimitError(
                f"DeepSeek API 速率限制 (HTTP {status}): {msg}",
                retry_after=_extract_retry_after(err),
            )
        if status is not None and 500 <= status < 600:
            return LLMConnectionError(
                f"DeepSeek API 服务器错误 (HTTP {status}): {msg}",
                host="api.deepseek.com",
            )
        return LLMError(f"DeepSeek API 错误 (HTTP {status}): {msg}")

    # -- Catch-all --------------------------------------------------------
    return LLMError(f"DeepSeek API 未知错误: {msg}")


def _extract_retry_after(err: Exception) -> float | None:
    """从 openai RateLimitError 响应头中提取 Retry-After 值。

    Args:
        err: openai SDK 异常实例。

    Returns:
        Retry-After 秒数，无法提取时返回 None。
    """
    try:
        response = getattr(err, "response", None)
        if response is None:
            return None
        headers = getattr(response, "headers", None)
        if headers is None:
            return None
        value = headers.get("Retry-After") or headers.get("retry-after")
        if value is not None:
            return float(value)
    except (AttributeError, ValueError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# DeepSeekAdapter
# ---------------------------------------------------------------------------


class DeepSeekAdapter(BaseLLMAdapter):
    """DeepSeek API 适配器 — 使用 OpenAI 兼容接口。

    DeepSeek Chat API 的 base_url 为 ``https://api.deepseek.com/v1``，
    支持 Chat Completions 接口的全部特性:
        - 多轮对话（system / user / assistant / tool 消息）
        - 流式输出（SSE）
        - Function Calling（工具调用）
        - Token 用量统计

    Usage::

        from miaowa.core.config import LLMConfig
        config = LLMConfig(api_key="sk-...", model="deepseek-v4-flash")
        adapter = DeepSeekAdapter(config)

        response = await adapter.chat([Message(role="user", content="你好")])
        print(response.content)

        async for chunk in adapter.stream([Message(role="user", content="你好")]):
            print(chunk.delta_content, end="", flush=True)

        await adapter.close()  # 释放 httpx 连接池
    """

    # ------------------------------------------------------------------
    # 构造与生命周期
    # ------------------------------------------------------------------

    def __init__(self, config: LLMConfig) -> None:
        """初始化 DeepSeek 适配器。

        Args:
            config: LLMConfig 实例，包含 api_key、base_url、model、
                temperature、max_tokens、timeout 等字段。

        Raises:
            ConfigMissingError: 若 config.api_key 为空（由 ConfigManager.validate
                在更早阶段抛出，本构造器不做二次校验）。
        """
        super().__init__(model=config.model)

        self._temperature: float = config.temperature
        self._max_tokens: int = config.max_tokens
        self._timeout: float = float(config.timeout)

        self._client: AsyncOpenAI = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=self._timeout,
            # 不在 SDK 层重试，原因:
            # 1. Agent Executor 通过 RetriableError mixin 识别可重试异常并统一重试
            # 2. 若 SDK 层和 Executor 层同时重试，会产生 N×M 次指数级叠加调用
            # 3. 统一在 Executor 层重试可记录结构化日志 + 更新 UI 状态
            max_retries=0,
        )

    async def close(self) -> None:
        """关闭底层 httpx AsyncClient 连接池。

        应在 Agent 退出或切换模型时调用，详见 BaseLLMAdapter.close()。
        """
        try:
            await self._client.close()
        except Exception:
            logger.debug("关闭 DeepSeek AsyncOpenAI 客户端时发生异常", exc_info=True)

    # ------------------------------------------------------------------
    # Token 计数
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """近似估算文本的 Token 数量。

        委托 ``miaowa.llm.tokenizer._count_chars_and_tokens`` 进行统一的
        CJK 检测和 token 计算，算法与 ``TokenCounter.count_tokens()`` 完全一致。

        算法（中文优先）:
            - CJK 字符（含中文、日文汉字/假名、中文标点等）: ~1.5 tokens / 字符
            - 非 CJK 字符（英文、数字、空格等）: ~4 字符 → 1 token

        对于纯英文文本，误差约 ±15%；对于中英混合文本，误差约 ±20%。
        MVP 阶段使用此近似算法，后续可升级为 DeepSeek 专用 tokenizer。

        Args:
            text: 待估算的文本。

        Returns:
            int: 估算的 token 数量，始终 >= 0。空字符串返回 0。
        """
        if not text:
            return 0
        tokens, _ = _count_chars_and_tokens(text)
        return tokens

    # ------------------------------------------------------------------
    # 模型信息
    # ------------------------------------------------------------------

    def get_model_info(self) -> ModelInfo:
        """获取 DeepSeek 当前模型元信息。

        Returns:
            ModelInfo: 包含以下字段:
                - provider: "deepseek"
                - model: 当前模型名称（如 "deepseek-v4-flash"）
                - max_tokens: 上下文窗口 1,000,000 tokens (DeepSeek V4)
                - supports_streaming: True
                - supports_tools: True
        """
        return ModelInfo(
            provider="deepseek",
            model=self._model or "deepseek-v4-flash",
            max_tokens=1_000_000,
            supports_streaming=True,
            supports_tools=True,
        )

    # ------------------------------------------------------------------
    # 底层 API 实现（Template Method 模式）
    # ------------------------------------------------------------------

    async def _chat_impl(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
    ) -> ChatResponse:
        """实现 BaseLLMAdapter._chat_impl — 发送非流式 API 请求。

        由基类 chat() 骨架方法调用，本方法无需再调用 _format_messages / _format_tools。

        Args:
            messages: 已预处理的 Message 列表。
            tools: 已预处理的 ToolDef 列表（可能为 None）。

        Returns:
            ChatResponse: 完整响应（含 content、tool_calls、finish_reason、usage）。

        Raises:
            LLMError: API 调用失败、超时、鉴权错误等。
            LLMResponseParseError: 响应体格式异常。
            LLMToolCallParseError: tool_calls[*].arguments JSON 解析失败。
        """
        start = time.perf_counter()
        completion = await self._send_request(messages, tools, stream=False)
        elapsed = time.perf_counter() - start

        logger.debug(
            f"DeepSeek chat 请求完成 | "
            f"model={self._model} | "
            f"耗时={elapsed:.2f}s | "
            f"finish_reason={completion.choices[0].finish_reason if completion.choices else 'N/A'}"
        )

        return self._parse_completion(completion)

    async def _stream_impl(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
    ) -> AsyncIterator[StreamChunk]:
        """实现 BaseLLMAdapter._stream_impl — 发送流式 API 请求。

        由基类 stream() 骨架方法调用，本方法无需再调用 _format_messages / _format_tools。

        Args:
            messages: 已预处理的 Message 列表。
            tools: 已预处理的 ToolDef 列表（可能为 None）。

        Yields:
            StreamChunk: 每个数据块包含增量内容和/或工具调用增量。
                tool_call_delta 中的 arguments 为增量 JSON 片段，
                消费者需跨 chunk 按 index 累积拼接后再 json.loads() 解析。
        """
        start = time.perf_counter()
        stream = await self._send_request(messages, tools, stream=True)

        chunk_count = 0
        async for chunk in stream:
            chunk_count += 1
            yield self._parse_stream_chunk(chunk)

        elapsed = time.perf_counter() - start
        logger.debug(
            f"DeepSeek stream 请求完成 | "
            f"model={self._model} | "
            f"chunk_count={chunk_count} | "
            f"耗时={elapsed:.2f}s"
        )

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    async def _send_request(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
        *,
        stream: bool,
    ):
        """发送 API 请求并返回 SDK 原生响应对象。

        封装了 _chat_impl / _stream_impl 的公共逻辑:
        参数构建 → 请求发送 → 异常映射 → 错误日志。

        Args:
            messages: 已预处理的 Message 列表。
            tools: 已预处理的 ToolDef 列表（可能为 None）。
            stream: 是否启用流式输出。

        Returns:
            openai SDK 原生响应对象（ChatCompletion 或 Stream[ChatCompletionChunk]）。

        Raises:
            LLMError: 通过 _map_openai_error 映射后的异常。
        """
        request_params = self._build_request_params(messages, tools, stream=stream)
        label = "stream" if stream else "chat"

        try:
            return await self._client.chat.completions.create(**request_params)
        except Exception as exc:
            logger.debug(
                f"DeepSeek {label} 请求失败 | "
                f"model={self._model} | "
                f"异常={type(exc).__name__}: {exc}"
            )
            raise _map_openai_error(exc) from exc

    def _build_request_params(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
        *,
        stream: bool,
    ) -> dict:
        """构建 openai SDK chat.completions.create() 所需的请求参数字典。

        Args:
            messages: 已预处理的 Message 列表。
            tools: 已预处理的 ToolDef 列表（可能为 None）。
            stream: 是否启用流式输出。

        Returns:
            openai SDK 兼容的参数字典。
        """
        params: dict = {
            "model": self._model,
            "messages": self._messages_to_dicts(messages),
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": stream,
        }

        if tools:
            params["tools"] = tools  # ToolDef 本身即 OpenAI 线格式

        return params

    @staticmethod
    def _messages_to_dicts(messages: list[Message]) -> list[dict]:
        """将 Message dataclass 列表转换为 OpenAI API 所需的 dict 列表。

        Args:
            messages: Message 对象列表。

        Returns:
            OpenAI API 兼容的消息 dict 列表，包含 tool_call_id 和 tool_calls 等必要字段。
        """
        result: list[dict] = []
        for msg in messages:
            entry: dict = {"role": msg.role, "content": msg.content}

            # tool 消息必须附带 tool_call_id 以关联对应的 function call
            if msg.tool_call_id is not None:
                entry["tool_call_id"] = msg.tool_call_id

            # assistant 消息中的 tool_calls 必须保留，否则 API 不知道上一轮调用了哪个工具
            if msg.tool_calls is not None:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in msg.tool_calls
                ]

            result.append(entry)
        return result

    @staticmethod
    def _parse_completion(completion) -> ChatResponse:
        """解析 openai SDK 非流式 ChatCompletion 响应为 ChatResponse。

        Args:
            completion: openai.types.chat.ChatCompletion 实例。

        Returns:
            ChatResponse 实例。

        Raises:
            LLMResponseParseError: choices 为空或响应结构异常。
            LLMToolCallParseError: tool_calls[*].arguments JSON 解析失败。
        """
        try:
            choice = completion.choices[0]
        except (IndexError, AttributeError) as exc:
            raise LLMResponseParseError(
                f"DeepSeek API 响应不包含 choices 字段或为空: {exc}"
            ) from exc

        finish_reason = _normalize_finish_reason(
            getattr(choice, "finish_reason", None)
        )

        content = getattr(choice.message, "content", None)

        # -- 解析 tool_calls ---------------------------------------------
        raw_tool_calls = getattr(choice.message, "tool_calls", None)
        tool_calls: list[ToolCall] | None = None

        if raw_tool_calls:
            parsed: list[ToolCall] = []
            for tc in raw_tool_calls:
                tc_id = getattr(tc, "id", "")
                tc_name = getattr(tc.function, "name", "") if hasattr(tc, "function") else ""
                raw_args = getattr(tc.function, "arguments", "{}") if hasattr(tc, "function") else "{}"

                try:
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError as exc:
                    raise LLMToolCallParseError(
                        f"工具 {tc_name!r} 的 arguments 不是合法 JSON: {exc}",
                        tool_name=tc_name,
                        raw_arguments=raw_args if isinstance(raw_args, str) else str(raw_args),
                    ) from exc

                parsed.append(ToolCall(id=tc_id, name=tc_name, arguments=arguments))

            tool_calls = parsed if parsed else None

        # -- 解析 usage --------------------------------------------------
        usage: dict[str, int] | None = None
        raw_usage = getattr(completion, "usage", None)
        if raw_usage is not None:
            usage = {
                "prompt_tokens": getattr(raw_usage, "prompt_tokens", 0),
                "completion_tokens": getattr(raw_usage, "completion_tokens", 0),
                "total_tokens": getattr(raw_usage, "total_tokens", 0),
            }

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    @staticmethod
    def _parse_stream_chunk(chunk) -> StreamChunk:
        """解析 openai SDK 流式 ChatCompletionChunk 为 StreamChunk。

        tool_call_delta 中的 arguments 字段为增量 JSON 片段（如 ``{"na``、
        ``me": ``、``"foo"}``），不保证在单个 chunk 中完整。消费者需要跨
        chunk 按 index 累积 arguments 字符串，拼接完成后才能 json.loads()。

        Args:
            chunk: openai.types.chat.ChatCompletionChunk 实例。

        Returns:
            StreamChunk 实例。
        """
        delta_content: str | None = None
        tool_call_delta: dict | None = None
        finish_reason: str | None = None

        try:
            choice = chunk.choices[0]
        except (IndexError, AttributeError):
            return StreamChunk()

        # finish_reason — 统一通过 _normalize_finish_reason 规范化
        raw_finish = getattr(choice, "finish_reason", None)
        if raw_finish is not None:
            finish_reason = _normalize_finish_reason(raw_finish)

        # delta
        delta = getattr(choice, "delta", None)
        if delta is not None:
            # 文本增量
            if hasattr(delta, "content") and delta.content is not None:
                delta_content = delta.content

            # 工具调用增量
            raw_tc_deltas = getattr(delta, "tool_calls", None)
            if raw_tc_deltas:
                # 流式响应中每次通常只有一个 tool_call delta
                tc = raw_tc_deltas[0]
                tool_call_delta = {
                    "index": getattr(tc, "index", 0),
                    "id": getattr(tc, "id", None),
                    "function": {
                        "name": getattr(tc.function, "name", None) if hasattr(tc, "function") else None,
                        "arguments": getattr(tc.function, "arguments", None) if hasattr(tc, "function") else None,
                    },
                }

        return StreamChunk(
            delta_content=delta_content,
            tool_call_delta=tool_call_delta,
            finish_reason=finish_reason,
        )


# ---------------------------------------------------------------------------
# 模块级辅助函数
# ---------------------------------------------------------------------------


def _normalize_finish_reason(raw: str | None) -> FinishReason:
    """将 openai API 返回的 finish_reason 字符串规范化为 FinishReason 类型。

    同时用于非流式响应（ChatResponse.finish_reason）和流式响应
    （StreamChunk.finish_reason），确保两者输出一致。

    Args:
        raw: API 原始 finish_reason 字符串，可能为 None。

    Returns:
        规范化后的 FinishReason 值。
    """
    if raw is None:
        return "stop"

    mapping: dict[str, FinishReason] = {
        "stop": "stop",
        "tool_calls": "tool_calls",
        "length": "length",
        "content_filter": "content_filter",
        # DeepSeek 可能的其他变体
        "function_call": "tool_calls",
    }
    return mapping.get(raw, "stop")
