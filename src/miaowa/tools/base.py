"""Tool 基类定义 — BaseTool 抽象类、ToolParameter、ToolDefinition 数据结构。

提供 Tool 的统一接口规范（PRD §6.1），包括：
- 参数定义（ToolParameter）
- 工具定义（ToolDefinition）
- 抽象基类（BaseTool）及其到 OpenAI Function Calling 格式的转换

所有具体工具（read_file、write_file、search_code 等）均继承 BaseTool 实现。

Typical usage::

    from miaowa.tools.base import BaseTool, ToolParameter

    class ReadFileTool(BaseTool):
        name = "read_file"
        description = "读取指定路径的文件内容"
        parameters = [
            ToolParameter(
                name="file_path",
                type="string",
                description="要读取的文件路径（相对于项目根目录）",
                required=True,
            ),
        ]

        async def execute(self, **kwargs):
            file_path = kwargs["file_path"]
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from miaowa.core.types import ToolDefinition, ToolParameter  # 规范定义源（PRD §6.1）


# ============================================================================
# BaseTool 抽象基类
# ============================================================================


class BaseTool(ABC):
    """所有工具的抽象基类。

    子类必须定义以下抽象属性与方法：
        - name: 工具名称（英文标识符）
        - description: 工具描述（中文，供 LLM 理解）
        - parameters: 工具参数列表
        - execute(**kwargs): 工具执行逻辑

    工具注册时通过 isinstance(obj, BaseTool) 进行类型检查，
    ToolRegistry 中所有条目均为此类的子类实例。
    """

    # ------------------------------------------------------------------
    # 抽象属性 — 子类必须实现
    # ------------------------------------------------------------------
    #
    # 使用 @property + @abstractmethod 定义接口契约。
    # 子类可通过 class attribute 简写形式实现（PRD §6.2 示例用法）：
    #
    #     class ReadFileTool(BaseTool):
    #         name = "read_file"
    #         description = "读取指定路径的文件内容"
    #         parameters = [
    #             ToolParameter(name="file_path", type="string", ...),
    #         ]
    #
    # Python ABC 会在实例化时检查抽象方法是否已被覆盖；
    # class attribute 会遮蔽 abstract property 描述符，通过检查。

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称（英文标识符）。

        Returns:
            如 "read_file"、"search_code"、"execute_shell" 等唯一标识。
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（中文）。

        Returns:
            供 LLM 理解工具用途的自然语言描述。
        """
        ...

    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """工具参数列表。

        Returns:
            ToolParameter 对象列表，描述每个参数的名称、类型、是否必填等信息。
        """
        ...

    # ------------------------------------------------------------------
    # 抽象方法 — 工具执行入口
    # ------------------------------------------------------------------

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """执行工具逻辑。

        Args:
            **kwargs: 经过 ToolValidator 校验后的参数字典。
                保证类型正确、必填参数完整、枚举值合法。

        Returns:
            工具执行结果，类型由具体工具定义。常见返回类型：
            - str: 文件内容、搜索结果等文本数据
            - dict: 结构化数据（如项目分析结果）
            - list: 批量操作结果

        Raises:
            ToolExecutionError: 工具执行失败时抛出。
                例如：文件不存在、Shell 命令非零退出、网络请求失败等。
        """
        ...

    # ------------------------------------------------------------------
    # 公共方法 — Schema 转换与描述生成
    # ------------------------------------------------------------------

    def to_openai_schema(self) -> dict[str, Any]:
        """将工具定义转换为 OpenAI / DeepSeek Function Calling 线格式。

        转换规则：
            - ToolParameter.type → JSON Schema type 字段
            - ToolParameter.description → JSON Schema description 字段
            - required=True 的参数名收集到 required 数组
            - enum 值写入 enum 字段

        Returns:
            符合 OpenAI Function Calling 规范的 dict，结构如下::

                {
                    "type": "function",
                    "function": {
                        "name": "<tool_name>",
                        "description": "<tool_description>",
                        "parameters": {
                            "type": "object",
                            "properties": { ... },
                            "required": [ ... ],
                        },
                    },
                }

        Note:
            返回值为 dict 而非 llm.types.ToolDef TypedDict，
            便于调用方直接传给 openai SDK（SDK 接受 dict 参数）。
        """
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum is not None:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        parameters_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters_schema["required"] = required

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters_schema,
            },
        }

    def to_system_prompt(self) -> str:
        """生成人类可读的工具描述文本，用于系统提示词。

        输出格式::

            ## read_file
            读取指定路径的文件内容。

            参数:
            - file_path (string, 必填): 要读取的文件路径
            - max_lines (integer, 可选): 最大读取行数，默认 500

        Returns:
            格式化后的工具描述字符串。
        """
        lines: list[str] = []
        lines.append(f"## {self.name}")
        lines.append(self.description)
        lines.append("")
        lines.append("参数:")

        for param in self.parameters:
            req_label = "必填" if param.required else "可选"
            enum_note = ""
            if param.enum is not None:
                enum_note = f"，可选值: {', '.join(param.enum)}"
            default_note = ""
            if not param.required and param.default is not None:
                default_note = f"，默认 {param.default!r}"
            lines.append(
                f"- **{param.name}** ({param.type}, {req_label}){default_note}{enum_note}: "
                f"{param.description}"
            )

        return "\n".join(lines)
