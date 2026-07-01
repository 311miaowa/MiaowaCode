"""LLM 模块 — 统一的 LLM 适配器接口与类型定义。

核心导出:
    - BaseLLMAdapter: LLM 适配器抽象基类
    - Message / ToolCall / ChatResponse / StreamChunk: 核心数据类型
    - ToolDef / ToolDefFunction / ModelInfo: API 线格式类型
"""

from miaowa.llm.base import BaseLLMAdapter
from miaowa.llm.types import (
    ChatResponse,
    FinishReason,
    Message,
    ModelInfo,
    Role,
    StreamChunk,
    ToolCall,
    ToolDef,
    ToolDefFunction,
)

__all__ = [
    "BaseLLMAdapter",
    "ChatResponse",
    "FinishReason",
    "Message",
    "ModelInfo",
    "Role",
    "StreamChunk",
    "ToolCall",
    "ToolDef",
    "ToolDefFunction",
]
