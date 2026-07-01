"""Core 模块类型定义 — 工具参数、工具结果、上下文载荷、项目缓存等通用数据类型。

跨模块引用说明：
- AgentContext 的完整定义位于 agent/context.py（ContextBuilder 的运行时产物）。
  PRD §5.2.1 中 Planner.think() 接收该类型，它提供 build_messages() 方法和
  available_tools 属性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from miaowa.llm.types import Message, ToolCall

# ---------------------------------------------------------------------------
# Tool parameter & definition types (PRD §6.1)
# ---------------------------------------------------------------------------


@dataclass
class ToolParameter:
    """工具参数定义。

    Attributes:
        name: 参数名称（英文标识符）。
        type: 参数类型。支持 "string"、"integer"、"number"、"boolean"、"array"、"object"。
        description: 参数描述（中文，供 LLM 理解）。
        required: 是否必填，默认 True。
        default: 默认值，仅在 required=False 时有意义。
        enum: 可选枚举值列表，用于约束参数取值范围。
            枚举值比较**区分大小写**。None 表示无限制。
    """

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class ToolDefinition:
    """工具定义（内部表示）。

    与 llm.types.ToolDef（OpenAI API 线格式）不同，
    本类型使用强类型的 ToolParameter 列表描述参数，
    通过 BaseTool.to_openai_schema() 转换为线格式。

    Attributes:
        name: 工具名称（英文标识符）。
        description: 工具描述（中文，供 LLM 理解）。
        parameters: 工具参数列表。
    """

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """工具执行结果。

    Attributes:
        success: 是否执行成功。
        data: 执行成功时的返回数据。success=False 时必须为 None。
        error: 执行失败时的错误信息。success=True 时必须为 None。

    Note:
        推荐使用 ok() / fail() 工厂方法构造，
        直接调用构造器会触发 __post_init__ 互斥性校验。
    """

    success: bool
    data: Any | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        """校验 data / error 互斥性。"""
        if self.success and self.error is not None:
            raise ValueError(
                f"成功结果不能包含 error 字段: {self.error!r}"
            )
        if not self.success and self.data is not None:
            raise ValueError(
                f"失败结果不能包含 data 字段: {self.data!r}"
            )

    @classmethod
    def ok(cls, data: Any = None) -> ToolResult:
        """快捷创建成功结果。"""
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str) -> ToolResult:
        """快捷创建失败结果。"""
        return cls(success=False, error=error)


@dataclass
class ContextPayload:
    """Agent 上下文载荷，用于传递给 LLM 的完整会话信息。

    Attributes:
        messages: 对话消息列表（与 OpenAI Chat Completions API 格式兼容）。
        total_tokens: 上下文占用的近似 token 数，必须 ≥ 0。
    """

    messages: list[Message]
    total_tokens: int

    def __post_init__(self) -> None:
        """校验 token 计数合法性。"""
        if self.total_tokens < 0:
            raise ValueError(
                f"total_tokens 不能为负数: {self.total_tokens}"
            )


@dataclass
class ProjectCache:
    """项目分析缓存，存储对当前项目结构的分析结果。

    Attributes:
        tech_stack: 识别的技术栈信息。
            常见键: language, framework, build_tool, package_manager, runtime_version。
        structure: 项目目录结构摘要。
            常见键: tree (目录树文本), module_count (模块数), file_count (文件数)。
        key_files: 关键文件路径列表，如 ["pyproject.toml", "README.md", "src/"]。
    """

    tech_stack: dict[str, Any] = field(default_factory=dict)
    structure: dict[str, Any] = field(default_factory=dict)
    key_files: list[str] = field(default_factory=list)


@dataclass
class ThoughtResult:
    """Agent 思考过程产出。

    在一次推理-行动循环中，Agent 产生思考内容，
    可能需要调用工具、也可能直接给出最终回答。

    Attributes:
        thought: 模型的思考文本（推理过程描述）。
        tool_calls: 需要执行的工具调用列表，为 None 时表示无需调用工具，
            直接使用 thought 作为最终回复。
        needs_more_info: 是否需要更多信息才能完成任务。
    """

    thought: str | None = None
    tool_calls: list[ToolCall] | None = None
    needs_more_info: bool = False


# ---------------------------------------------------------------------------
# Application configuration
# ---------------------------------------------------------------------------


@dataclass
class AppConfig:
    """应用配置。

    对应多层配置加载（env → yaml → default）后的最终配置对象。
    被 Planner、ContextBuilder、DeepSeekAdapter 等模块共同依赖。

    Attributes:
        api_key: DeepSeek API Key（必填，来自环境变量 MIAOWA_API_KEY）。
        base_url: DeepSeek API 地址，默认 https://api.deepseek.com/v1。
        model: 默认模型名称，默认 deepseek-chat。
        max_turns: 最大对话轮数，默认 50。
        timeout: HTTP 请求超时时间（秒），默认 120。
        log_level: 日志级别，默认 "INFO"。
    """

    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    max_turns: int = 50
    timeout: int = 120
    log_level: str = "INFO"
