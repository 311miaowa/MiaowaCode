"""LLM 适配器抽象基类 — 定义统一的 LLM 调用接口。

基于 PRD §3.3 和 §5.2.4 设计，采用适配器模式封装不同 LLM 提供商的 API 差异。

架构说明:
    chat() / stream() 是本模块的骨架方法（Template Method 模式），
    它们在基类中统一调用 _format_messages / _format_tools 钩子后，
    再委托给子类实现的 _chat_impl / _stream_impl 发送底层 API 请求。

    子类只需实现四个抽象方法即可插入 Agent 执行流程:
        - _chat_impl(): 非流式 API 请求
        - _stream_impl(): 流式 API 请求
        - count_tokens(): Token 估算
        - get_model_info(): 模型元信息

典型子类: DeepSeekAdapter (deepseek.py)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from miaowa.llm.tokenizer import _MSG_OVERHEAD_TOKENS
from miaowa.llm.types import (
    ChatResponse,
    Message,
    ModelInfo,
    StreamChunk,
    ToolDef,
)


class BaseLLMAdapter(ABC):
    """LLM 适配器抽象基类 — 定义统一的 LLM 调用接口。

    适配器模式的核心角色：将不同 LLM 提供商（DeepSeek、OpenAI、Claude、
    Gemini 等）的 API 差异封装在子类中，Agent 层仅依赖本基类接口，
    无需关心底层实现。

    子类必须实现:
        - _chat_impl(): 非流式底层 API 请求
        - _stream_impl(): 流式底层 API 请求
        - count_tokens(): Token 估算
        - get_model_info(): 模型元信息

    子类可选覆盖:
        - _format_messages(): 消息预处理（如注入 system prompt、截断历史）
        - _format_tools(): 工具定义预处理（如过滤不兼容的参数格式）
        - count_message_tokens(): 消息列表级 token 计数
        - close(): 资源清理（如关闭 httpx 连接池）
    """

    # ------------------------------------------------------------------
    # 构造与生命周期
    # ------------------------------------------------------------------

    def __init__(self, *, model: str | None = None) -> None:
        """初始化适配器。

        子类应通过 super().__init__(model=...) 调用本构造器，
        以确保 self._model 被正确设置。

        Args:
            model: 模型名称。子类通常从 LLMConfig.model 传入，
                也可在子类构造器中自行指定默认值。
        """
        self._model: str | None = model

    async def close(self) -> None:
        """关闭适配器，释放底层资源（如 httpx AsyncClient 连接池）。

        默认实现为空操作。持有异步 HTTP 客户端的子类应覆盖本方法，
        在其中 await client.aclose()。

        Agent 层在退出或切换模型时调用本方法，确保连接正确释放。
        """
        pass

    # ------------------------------------------------------------------
    # 公共 API — 骨架方法（Template Method 模式）
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> ChatResponse:
        """发送非流式对话请求，返回完整响应。

        骨架方法（Template Method 模式）:
            1. 调用 _format_messages(messages) 预处理消息
            2. 调用 _format_tools(tools) 预处理工具定义
            3. 委托 _chat_impl() 发送底层 API 请求

        调用 LLM Chat Completions API（stream=False），等待模型生成完整的
        回复文本及可能的工具调用列表后一次性返回。

        Args:
            messages: 对话消息列表，按时间顺序排列。
                通常以 system 消息开头，以 user 消息结尾。
            tools: 可选的工具定义列表（OpenAI ToolDef 兼容格式）。
                传入 None 或空列表表示本轮对话不启用工具调用。

        Returns:
            ChatResponse: 包含回复文本、工具调用、终止原因和 token 用量。
                当模型请求调用工具时，content 为 None，tool_calls 非空；
                当模型直接回复时，tool_calls 为 None，content 非空。

        Raises:
            LLMError: API 调用失败、超时、鉴权错误等底层异常。
        """
        formatted_messages = self._format_messages(messages)
        formatted_tools = self._format_tools(tools)
        return await self._chat_impl(formatted_messages, formatted_tools)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """发送流式对话请求，逐步产出响应数据块。

        骨架方法（Template Method 模式）:
            1. 调用 _format_messages(messages) 预处理消息
            2. 调用 _format_tools(tools) 预处理工具定义
            3. 委托 _stream_impl() 发送底层流式 API 请求

        调用 LLM Chat Completions API（stream=True），以异步迭代器形式
        逐块返回增量内容，适用于需要实时展示回复的场景（如 REPL 打字效果）。

        Args:
            messages: 对话消息列表，格式同 chat()。
            tools: 可选的工具定义列表，格式同 chat()。

        Yields:
            StreamChunk: 每个数据块包含增量文本（delta_content）、
                工具调用增量（tool_call_delta）和终止原因（finish_reason）。
                最后一个 chunk 的 finish_reason 非 None，表示流结束。

        Raises:
            LLMError: API 调用失败、超时、鉴权错误等底层异常。
        """
        formatted_messages = self._format_messages(messages)
        formatted_tools = self._format_tools(tools)
        async for chunk in self._stream_impl(formatted_messages, formatted_tools):
            yield chunk

    # ------------------------------------------------------------------
    # 抽象方法 — 子类必须实现
    # ------------------------------------------------------------------

    @abstractmethod
    async def _chat_impl(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
    ) -> ChatResponse:
        """子类实现：发送非流式 API 请求并返回完整响应。

        本方法由 chat() 骨架方法在消息/工具预处理之后调用。
        子类无需再调用 _format_messages / _format_tools。

        Args:
            messages: 已通过 _format_messages 预处理的消息列表。
            tools: 已通过 _format_tools 预处理的工具定义列表（可能为 None）。

        Returns:
            ChatResponse: 完整响应（含 content、tool_calls、finish_reason、usage）。

        Raises:
            LLMError: 子类应将底层 SDK 异常统一转换为 LLMError 体系。
        """
        ...

    @abstractmethod
    async def _stream_impl(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
    ) -> AsyncIterator[StreamChunk]:
        """子类实现：发送流式 API 请求并逐步产出数据块。

        本方法由 stream() 骨架方法在消息/工具预处理之后调用。
        子类无需再调用 _format_messages / _format_tools。

        Args:
            messages: 已通过 _format_messages 预处理的消息列表。
            tools: 已通过 _format_tools 预处理的工具定义列表（可能为 None）。

        Yields:
            StreamChunk: 每个数据块包含增量内容和/或工具调用增量。

        Raises:
            LLMError: 子类应将底层 SDK 异常统一转换为 LLMError 体系。
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """估算给定文本的 Token 数量。

        用于上下文窗口管理、消息截断和 token 用量统计。
        子类应使用对应模型的 tokenizer 或近似算法，精度要求：
        - 精确 tokenizer（tiktoken / HuggingFace tokenizer）: 误差 < 1%
        - 近似估算（字符数 / N）: 误差 < 30%

        Args:
            text: 待估算的文本字符串。

        Returns:
            int: 估算的 token 数量，始终 >= 0。
        """
        ...

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """获取当前模型的元信息。

        供 Agent 层、日志系统和状态栏展示使用。

        Returns:
            ModelInfo: 包含 provider、model、max_tokens、
                supports_streaming、supports_tools 五个字段。
                - provider (str): LLM 提供商名称，如 "deepseek"、"openai"。
                - model (str): 当前使用的模型名称，如 "deepseek-chat"。
                - max_tokens (int): 模型上下文窗口上限。
                - supports_streaming (bool): 是否支持流式输出。
                - supports_tools (bool): 是否支持工具调用（Function Calling）。
        """
        ...

    # ------------------------------------------------------------------
    # 非抽象公共方法 — 子类可选覆盖
    # ------------------------------------------------------------------

    def count_message_tokens(self, messages: list[Message]) -> int:
        """估算消息列表的 Token 数量（含角色标签等格式开销）。

        供 ContextBuilder 做上下文截断决策时调用。
        算法与 ``TokenCounter.count_messages()`` 一致：对每条消息
        分别调用 ``count_tokens(content)`` 后累加固定格式开销，
        而非将消息拼接后再计数（避免 `role:` 前缀和换行符污染计数）。

        默认实现：每条消息附加 ``_MSG_OVERHEAD_TOKENS`` tokens
        的固定格式开销（OpenAI 经验值）。
        子类可覆盖以使用模型特定的精确计算（如 tiktoken 的
        num_tokens_from_messages）。

        Args:
            messages: 对话消息列表。

        Returns:
            int: 估算的 token 数量，始终 >= 0。
        """
        total = 0
        for msg in messages:
            total += self.count_tokens(msg.content) + _MSG_OVERHEAD_TOKENS
        return total

    # ------------------------------------------------------------------
    # 钩子方法 — 子类可选覆盖
    # ------------------------------------------------------------------

    def _format_messages(self, messages: list[Message]) -> list[Message]:
        """消息格式化钩子 — 子类可重写以执行预处理。

        由 chat() 和 stream() 骨架方法在发送请求前自动调用。
        允许子类对消息列表做任意变换，例如:
            - 注入默认 system prompt
            - 截断超出上下文窗口的历史消息
            - 合并连续同角色消息（某些 API 要求）
            - 转换为供应商特定格式（如 Gemini 的 parts 结构）

        默认实现原样返回，不做任何处理。

        Args:
            messages: 原始消息列表。

        Returns:
            list[Message]: 处理后的消息列表。
        """
        return messages

    def _format_tools(
        self, tools: list[ToolDef] | None
    ) -> list[ToolDef] | None:
        """工具格式化钩子 — 子类可重写以执行预处理。

        由 chat() 和 stream() 骨架方法在发送请求前自动调用。
        允许子类对工具定义列表做任意变换，例如:
            - 过滤不支持的参数类型
            - 为工具描述追加格式提示（如要求返回 JSON）
            - 注入内置工具（如 finish_task）
            - 转换为供应商特定格式

        默认实现原样返回，不做任何处理。

        Args:
            tools: 原始工具定义列表，可能为 None。

        Returns:
            list[ToolDef] | None: 处理后的工具定义列表。
        """
        return tools
