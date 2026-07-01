"""LLM 模块类型定义 — 消息、工具调用、响应、流式数据块等核心数据类型。

类型层级关系：
    Message / ToolCall → ChatResponse / StreamChunk
    ToolDefFunction → ToolDef（OpenAI API 线格式）

与 core.types 的关系：
    core.types.ToolDefinition（内部表示）→ 通过 BaseTool.to_openai_schema()
    转换为本模块的 ToolDef（线格式）后发送给 LLM API。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

# ---------------------------------------------------------------------------
# TypedDict — OpenAI API 线格式
# ---------------------------------------------------------------------------


class ToolDefFunction(TypedDict):
    """Tool function 描述（OpenAI / DeepSeek 兼容格式）。"""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolDef(TypedDict):
    """Tool 定义（OpenAI / DeepSeek 兼容格式）。

    与 core.types.ToolDefinition 的区别：
        ToolDefinition 是强类型的内部表示（parameters 为 list[ToolParameter]）；
        ToolDef 是发送给 LLM API 的线格式（parameters 为 JSON Schema dict）。
    """

    type: Literal["function"]
    function: ToolDefFunction


# ---------------------------------------------------------------------------
# ModelInfo — 模型元信息
# ---------------------------------------------------------------------------


class ModelInfo(TypedDict):
    """LLM 模型元信息。

    供 Agent 层、日志系统和状态栏展示使用。

    Attributes:
        provider: LLM 提供商名称，如 ``"deepseek"``、``"openai"``。
        model: 当前使用的模型名称，如 ``"deepseek-chat"``。
        max_tokens: 模型上下文窗口 token 上限。
        supports_streaming: 是否支持流式输出。
        supports_tools: 是否支持 Function Calling 工具调用。
    """

    provider: str
    model: str
    max_tokens: int
    supports_streaming: bool
    supports_tools: bool


# ---------------------------------------------------------------------------
# Role type alias — 约束 Message.role 的合法取值
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]
"""LLM 对话中的消息角色。

- system: 系统指令 / 提示词
- user: 用户输入
- assistant: 模型回复
- tool: 工具执行结果
"""

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """LLM 对话消息。

    Attributes:
        role: 角色标识，限定为 'system'、'user'、'assistant'、'tool' 之一。
        content: 消息文本内容。
        tool_call_id: 工具调用 ID。仅 role="tool" 时需要，
            用于将 tool 执行结果与对应的 function call 关联。
        tool_calls: 工具调用列表。仅 role="assistant" 且模型请求调用工具时需要。
            用于截断算法中的 tool-call ↔ tool-result 配对保护。
    """

    role: Role
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass
class ToolCall:
    """LLM 返回的工具调用。

    Attributes:
        id: 工具调用唯一标识（由 LLM 生成）。
        name: 工具名称。
        arguments: 工具参数字典。

    Note:
        OpenAI / DeepSeek API 原始响应中 arguments 字段为 JSON 字符串，
        需要先 json.loads() 解析后再赋值给本字段。
    """

    id: str
    name: str
    arguments: dict[str, Any]


FinishReason = Literal["stop", "tool_calls", "length", "content_filter"]
"""LLM 响应的终止原因。

- stop: 正常结束
- tool_calls: 模型请求调用工具
- length: 达到最大 token 限制
- content_filter: 内容被安全过滤
"""


@dataclass
class ChatResponse:
    """LLM 聊天响应（非流式）。

    Attributes:
        content: 模型文本回复。触发 tool_calls 时通常为 None。
        tool_calls: 工具调用列表，模型要求执行的工具操作。无工具调用时为 None。
        finish_reason: 终止原因，默认 "stop"。
        usage: Token 用量统计（prompt_tokens, completion_tokens, total_tokens）。
            部分流式响应中可能为 None。
    """

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: FinishReason = "stop"
    usage: dict[str, int] | None = None


@dataclass
class StreamChunk:
    """LLM 流式响应的单个数据块。

    Attributes:
        delta_content: 本次增量文本内容。首个 chunk 可能为 None（含 role 信息）。
        tool_call_delta: 工具调用的增量数据。
            常见键: index (int), id (str | None), function (dict: name + arguments 片段)。
        finish_reason: 终止原因，仅当流结束的最后 chunk 中非 None。
    """

    delta_content: str | None = None
    tool_call_delta: dict[str, Any] | None = None
    finish_reason: str | None = None
